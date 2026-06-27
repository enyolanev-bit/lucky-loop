#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from luckyloop.executor import execute
from luckyloop.loop import build_state
from luckyloop.planner import action_key, generate_candidates
from luckyloop.schemas import ExperimentTrace, ProposedAction, TaskSpec
from luckyloop.tasks import ROOT, load_task
from luckyloop.verifier import verify_sweep

from run_ablation_suite import TASK_PATHS, _choose_classic_action


VERIFICATION_MODELS = {"verification_sweep", "top_model_verification"}


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
    lucky_verification = lucky.get("verification")
    classic_verification = classic.get("verification")
    lucky_blocks_or_calibrates = bool(lucky_verification and not lucky_verification.get("trustworthy", False))
    lucky_trusted = bool(lucky_verification and lucky_verification.get("trustworthy", False))
    classic_is_score_chasing = not classic_verification and state_needs_verification
    lucky_metric = lucky.get("actual_metric")
    classic_metric = classic.get("actual_metric")
    if lucky_metric is not None and classic_metric is not None:
        if abs(lucky_metric - classic_metric) < 1e-12:
            score_verdict = "tie"
        elif lucky_metric > classic_metric:
            score_verdict = "lucky_win"
        else:
            score_verdict = "classic_win"
    else:
        score_verdict = "not_comparable"

    if lucky_trusted:
        claim_safety_verdict = "lucky_win"
        overall = "lucky_win"
        reason = "Lucky choice produced a trusted verifier claim."
    elif lucky_blocks_or_calibrates and classic_is_score_chasing:
        claim_safety_verdict = "lucky_win"
        overall = "lucky_win"
        reason = "Lucky choice ran verification and prevented a robust claim that classic score-chasing would leave unsupported."
    elif classic_is_score_chasing:
        claim_safety_verdict = "lucky_win"
        overall = "lucky_win"
        reason = "Lucky choice spent compute on claim risk while classic continued a non-verification run."
    else:
        claim_safety_verdict = "tie"
        overall = score_verdict
        reason = "No claim-safety advantage was detected; compare immediate score only."
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
    lucky_wins = sum(1 for row in rows if row["verdict"]["overall_verdict"] == "lucky_win")
    usable = len(rows)
    win_rate = (lucky_wins / usable) if usable else None
    payload = {
        "schema_version": "1.0",
        "summary": {
            "cases": usable,
            "lucky_wins": lucky_wins,
            "qwen_choice_usefulness": win_rate,
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
    parser.add_argument("--tasks", nargs="*", default=TASK_PATHS)
    parser.add_argument("--max-cases-per-task", type=int, default=1)
    args = parser.parse_args()
    rows = []
    for task_path in args.tasks:
        rows.extend(run_task(load_task(task_path), args.max_cases_per_task))
    write_reports(rows)
    print("Wrote reports/counterfactuals/counterfactual_evaluation.md")


if __name__ == "__main__":
    main()
