#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from luckyloop.executor import execute
from luckyloop.defaults import CORE_TASK_PATHS
from luckyloop.loop import build_state
from luckyloop.planner import action_key, generate_candidates
from luckyloop.schemas import ExperimentTrace, ProposedAction, TaskSpec
from luckyloop.tasks import ROOT, load_task
from luckyloop.verifier import verify_sweep

from luckyloop.operator_trace import append_operator_event, write_operator_summary

from run_ablation_suite import _choose_classic_action


VERIFICATION_MODELS = {"verification_sweep", "top_model_verification", "protocol_probe"}


def load_traces(policy: str, task_id: str) -> list[ExperimentTrace]:
    run_dir = ROOT / "runs" / "ablations" / policy / task_id
    return [
        ExperimentTrace.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("run_*.json"))
    ]


def _same_action(a: ProposedAction, b: ProposedAction) -> bool:
    return action_key(a) == action_key(b)


def _find_verification_cases(task: TaskSpec, traces: list[ExperimentTrace], max_cases: int) -> list[dict]:
    cases = []
    for index, trace in enumerate(traces):
        if trace.proposed_action.model not in VERIFICATION_MODELS:
            continue
        if not trace.decision_trace or not trace.decision_trace.world_model_signal_used:
            continue
        prefix = traces[:index]
        state = build_state(task, prefix, index + 1, len(traces), "Counterfactual state reconstructed from Lucky Loop traces.")
        seen = {action_key(t.proposed_action) for t in prefix}
        candidates = generate_candidates(task, state, prefix, seen)
        classic = _choose_classic_action("classic_autoresearch", task, candidates, prefix)
        if _same_action(classic, trace.proposed_action):
            continue
        cases.append(
            {
                "case_id": f"case_{len(cases) + 1:03d}",
                "run_index": index + 1,
                "state": state,
                "prefix": prefix,
                "lucky_action": trace.proposed_action,
                "classic_action": classic,
                "lucky_trace": trace,
                "candidates": candidates,
            }
        )
        if len(cases) >= max_cases:
            break
    return cases


def _run_action(action: ProposedAction, out_path: Path) -> dict:
    result = execute(action.command, cwd=ROOT)
    verification = verify_sweep(result.raw) if result.raw.get("runs") else None
    metric = result.accuracy
    if metric is None:
        raw_metric = result.raw.get("metric", "accuracy")
        best = result.raw.get("best") or {}
        value = best.get(f"mean_{raw_metric}")
        metric = float(value) if value is not None else None
    payload = {
        "action": action.model_dump(),
        "actual_result": result.model_dump(),
        "actual_metric": metric,
        "verification": verification.model_dump() if verification else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _verdict(lucky: dict, classic: dict, state_needs_verification: bool) -> dict:
    """Honest paired verdict.

    A claim-safety lucky_win is credited ONLY when Lucky actually produced a verification result
    AND classic genuinely skipped verification the state required. Lucky running a verification
    action that yielded no verification, or classic not being score-chasing, is never a win.

    The raw-score comparison is only meaningful when both choices are the same KIND of experiment.
    Lucky's verification is a multi-seed sweep (mean metric) while classic is typically a single
    split — comparing those is apples-to-oranges, so it is reported as not_comparable.
    """
    lucky_verification = lucky.get("verification")
    classic_verification = classic.get("verification")
    lucky_ran_verification = lucky_verification is not None
    classic_ran_verification = classic_verification is not None
    lucky_trusted = bool(lucky_verification and lucky_verification.get("trustworthy", False))
    lucky_blocked = bool(lucky_verification and not lucky_verification.get("trustworthy", False))
    classic_is_score_chasing = (not classic_ran_verification) and state_needs_verification

    lucky_metric = lucky.get("actual_metric")
    classic_metric = classic.get("actual_metric")
    # Same KIND of experiment? Lucky's verification is a multi-seed sweep (mean metric); a classic
    # single-split run is not comparable to it. Compare raw scores only for like-for-like actions.
    lucky_is_sweep = lucky.get("action", {}).get("model") in VERIFICATION_MODELS
    classic_is_sweep = classic.get("action", {}).get("model") in VERIFICATION_MODELS
    comparable = (
        lucky_is_sweep == classic_is_sweep
        and lucky_metric is not None
        and classic_metric is not None
    )
    if not comparable:
        score_verdict = "not_comparable"
    elif abs(lucky_metric - classic_metric) < 1e-9:
        score_verdict = "tie"
    elif lucky_metric > classic_metric:
        score_verdict = "lucky_win"
    else:
        score_verdict = "classic_win"

    if lucky_trusted and classic_is_score_chasing:
        claim_safety_verdict = "lucky_win"
        reason = "Lucky ran verification and produced a trusted claim; classic skipped the verification the state required."
    elif lucky_blocked and classic_is_score_chasing:
        claim_safety_verdict = "lucky_win"
        reason = "Lucky ran verification and blocked an unsupported claim that classic score-chasing would have left."
    elif not lucky_ran_verification:
        claim_safety_verdict = "not_comparable"
        reason = "Lucky's verification action produced no verification result; no claim-safety advantage can be measured."
    elif lucky_trusted or lucky_blocked:
        claim_safety_verdict = "tie"
        reason = "Lucky verified, but classic was not score-chasing (state did not require verification); no safety advantage proven."
    else:
        claim_safety_verdict = "tie"
        reason = "No claim-safety advantage detected."

    # Overall is a win ONLY on a genuine claim-safety win; otherwise fall back to the honest
    # (often not_comparable) immediate-score comparison.
    overall = "lucky_win" if claim_safety_verdict == "lucky_win" else score_verdict
    return {
        "score_verdict": score_verdict,
        "claim_safety_verdict": claim_safety_verdict,
        "overall_verdict": overall,
        "reason": reason,
    }


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def run_task(task: TaskSpec, max_cases: int) -> list[dict]:
    lucky_traces = load_traces("lucky_loop_full", task.task_id)
    cases = _find_verification_cases(task, lucky_traces, max_cases)
    rows = []
    for case in cases:
        out_dir = ROOT / "runs" / "counterfactuals" / task.task_id / case["case_id"]
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lucky = _run_action(case["lucky_action"], out_dir / "lucky_choice.json")
        classic = _run_action(case["classic_action"], out_dir / "classic_choice.json")
        state_needs_verification = bool(
            case["state"].top_model_summary and case["state"].top_model_summary.needs_robustness_verification
        )
        verdict = _verdict(lucky, classic, state_needs_verification)
        row = {
            "task": task.task_id,
            "case_id": case["case_id"],
            "state_id": case["state"].state_id,
            "run_index": case["run_index"],
            "state_needs_verification": state_needs_verification,
            "lucky_action": case["lucky_action"].model_dump(),
            "classic_action": case["classic_action"].model_dump(),
            "lucky_actual_metric": lucky.get("actual_metric"),
            "classic_actual_metric": classic.get("actual_metric"),
            "lucky_verification": lucky.get("verification"),
            "classic_verification": classic.get("verification"),
            "verdict": verdict,
            "artifacts": {
                "lucky_choice": str((out_dir / "lucky_choice.json").relative_to(ROOT)),
                "classic_choice": str((out_dir / "classic_choice.json").relative_to(ROOT)),
            },
        }
        (out_dir / "case_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        print(f"{task.task_id}/{case['case_id']}: {verdict['overall_verdict']} - {verdict['reason']}")
    return rows


def write_reports(rows: list[dict]) -> None:
    out_dir = ROOT / "reports" / "counterfactuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    usable = len(rows)
    breakdown: dict[str, int] = {}
    for row in rows:
        v = row["verdict"]["overall_verdict"]
        breakdown[v] = breakdown.get(v, 0) + 1
    lucky_wins = breakdown.get("lucky_win", 0)
    # Claim-safety wins only — never counts not_comparable / score-only cases as wins.
    claim_safety_wins = sum(1 for row in rows if row["verdict"]["claim_safety_verdict"] == "lucky_win")
    win_rate = (lucky_wins / usable) if usable else None
    payload = {
        "schema_version": "1.1",
        "summary": {
            "cases": usable,
            "lucky_wins": lucky_wins,
            "claim_safety_wins": claim_safety_wins,
            "overall_verdict_breakdown": breakdown,
            "qwen_choice_usefulness": win_rate,  # honest: only genuine claim-safety wins / cases
        },
        "rows": rows,
    }
    (out_dir / "counterfactual_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Counterfactual Evaluation",
        "",
        "This report executes paired choices from the same reconstructed state: Lucky Loop's Qwen-guided action versus the action a classic score-chasing policy would have taken.",
        "",
        f"- Cases: {usable}",
        f"- Lucky wins: {lucky_wins}",
        f"- Qwen choice usefulness: {'' if win_rate is None else f'{win_rate:.2%}'}",
        "",
        "| Task | Case | Lucky action | Classic action | Lucky metric | Classic metric | Score verdict | Claim-safety verdict | Overall | Reason |",
        "|---|---|---|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['case_id']} | {row['lucky_action']['model']} | "
            f"{row['classic_action']['model']} | {_fmt(row['lucky_actual_metric'])} | "
            f"{_fmt(row['classic_actual_metric'])} | {row['verdict']['score_verdict']} | "
            f"{row['verdict']['claim_safety_verdict']} | {row['verdict']['overall_verdict']} | "
            f"{row['verdict']['reason']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A Lucky win means the Qwen-guided choice produced trusted evidence or prevented an unsupported claim when the classic policy would have continued score chasing. It does not necessarily mean Lucky won the immediate raw-score comparison.",
    ]
    (out_dir / "counterfactual_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=CORE_TASK_PATHS)
    parser.add_argument("--max-cases-per-task", type=int, default=1)
    args = parser.parse_args()
    task_ids = [Path(task_path).stem for task_path in args.tasks]
    append_operator_event(
        event_type="counterfactual_evaluation",
        goal="Evaluate states where Lucky Loop's world-model-guided choice differs from classic autoresearch.",
        action="run_counterfactual_evaluation",
        status="started",
        inputs={"tasks": args.tasks, "task_ids": task_ids, "max_cases_per_task": args.max_cases_per_task},
        rationale="Generate evidence that Qwen predictions can change the next experiment rather than merely narrating it.",
    )
    rows = []
    status = "completed"
    error = None
    try:
        for task_path in args.tasks:
            rows.extend(run_task(load_task(task_path), args.max_cases_per_task))
        write_reports(rows)
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        append_operator_event(
            event_type="counterfactual_evaluation",
            goal="Evaluate states where Lucky Loop's world-model-guided choice differs from classic autoresearch.",
            action="run_counterfactual_evaluation",
            status=status,
            inputs={"task_ids": task_ids},
            outputs={"rows": len(rows), "error": error},
            rationale="Counterfactual evaluation completed for the selected task subset.",
        )
        write_operator_summary()
    print("Wrote reports/counterfactuals/counterfactual_evaluation.md")


if __name__ == "__main__":
    main()
