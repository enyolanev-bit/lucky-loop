#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

from luckyloop.benchmark_metrics import actual_metric as shared_actual_metric
from luckyloop.benchmark_metrics import best_single_run as shared_best_single_run
from luckyloop.benchmark_metrics import claimable_evidence_summary
from luckyloop.calibration import write_calibration_report
from luckyloop.claim_ledger import entries_from_verification, write_claim_ledger
from luckyloop.comparator import compare
from luckyloop.executor import execute
from luckyloop.loop import build_state, run as run_lucky_loop
from luckyloop.planner import action_key, generate_candidates, initial_hypothesis
from luckyloop.reporter import generate_report
from luckyloop.research_agent import prompt_hash
from luckyloop.schemas import (
    AgentDecision,
    CandidatePrediction,
    Comparison,
    DecisionTrace,
    ExperimentTrace,
    Prediction,
    ProposedAction,
    RejectedCandidate,
    SafetyValidation,
    TaskSpec,
)
from luckyloop.tasks import ROOT, load_task
from luckyloop.top_models import detect_top_models
from luckyloop.verifier import verify_sweep


TASK_PATHS = [
    "configs/tasks/breast_cancer_accuracy.json",
    "configs/tasks/wine_accuracy.json",
    "configs/tasks/digits_accuracy.json",
]

POLICIES = ["classic_autoresearch", "classic_verified", "lucky_loop_full"]


def _clean_namespace(namespace: str) -> tuple[Path, Path]:
    runs_dir = ROOT / "runs" / namespace
    reports_dir = ROOT / "reports" / namespace
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir, reports_dir


def _actual_metric(trace: ExperimentTrace) -> float | None:
    return shared_actual_metric(trace)


def _best_single_run(traces: Iterable[ExperimentTrace]) -> ExperimentTrace | None:
    return shared_best_single_run(traces)


def _nop_prediction(action: ProposedAction) -> Prediction:
    return Prediction(
        expected_metric="no pre-compute world-model prediction",
        expected_runtime_seconds="not predicted",
        risks=["classic autoresearch baseline does not simulate this action before compute"],
        recommendation="run",
        rationale="This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.",
        action_specific_signal="none",
        claim_risk="Without prospective simulation, claim risk is only discovered after execution or verification.",
    )


def _nop_comparison(actual_status: str) -> Comparison:
    unexpected = ["no world-model prediction was made before compute"]
    if actual_status != "success":
        unexpected.append(f"execution status was {actual_status}")
    return Comparison(
        metric_match=False,
        runtime_match=False,
        unexpected_events=unexpected,
        lesson="Classic baseline has no prediction-vs-reality evidence for this action.",
    )


def _agent_decision(
    policy: str,
    task: TaskSpec,
    state_id: str,
    candidates: list[ProposedAction],
    preferred: ProposedAction,
) -> AgentDecision:
    candidate_ids = [candidate.action_id or "" for candidate in candidates[:6]]
    if preferred.action_id and preferred.action_id not in candidate_ids:
        candidate_ids.insert(0, preferred.action_id)
    if policy == "classic_autoresearch":
        hypothesis = "Score-chasing autoresearch should spend budget on the next promising model family."
        evidence = "Collect single-run metrics and report the best observed score."
        risk = "A best single-run claim can be unsupported if top models are close."
    else:
        hypothesis = "Classic autoresearch should verify the top observed models before a robust winner claim."
        evidence = "Run matched multi-seed verification when observed leaders are close."
        risk = "No world model predicts the verification need before compute."
    return AgentDecision(
        research_question=task.goal,
        working_hypothesis=hypothesis,
        candidate_action_ids=candidate_ids,
        preferred_action_id=preferred.action_id or "",
        rationale=f"{policy} selected {preferred.action_id} from state {state_id}.",
        expected_evidence_needed=evidence,
        claim_risk=risk,
        stop_or_continue="continue",
    )


def _choose_classic_action(
    policy: str,
    task: TaskSpec,
    candidates: list[ProposedAction],
    traces: list[ExperimentTrace],
) -> ProposedAction:
    top_summary = detect_top_models(traces, metric=task.primary_metric)
    if policy == "classic_verified" and top_summary.needs_robustness_verification:
        for candidate in candidates:
            if candidate.model == "top_model_verification":
                return candidate

    if not traces:
        for candidate in candidates:
            if candidate.model == "logistic_regression" and not candidate.params.get("scale"):
                return candidate
        return candidates[0]

    if (
        traces[0].proposed_action.model == "logistic_regression"
        and not traces[0].proposed_action.params.get("scale")
    ):
        for candidate in candidates:
            if candidate.model == "logistic_regression" and candidate.params.get("scale"):
                return candidate

    tested_models = {trace.proposed_action.model for trace in traces}
    priority = ["svc", "random_forest", "gradient_boosting", "hist_gradient_boosting", "logistic_regression"]
    for model in priority:
        for candidate in candidates:
            if candidate.model == model and candidate.model not in tested_models:
                return candidate

    if policy == "classic_verified":
        for candidate in candidates:
            if candidate.model in {"top_model_verification", "verification_sweep"}:
                return candidate

    for candidate in candidates:
        if candidate.model not in {"top_model_verification", "verification_sweep"}:
            return candidate
    return candidates[0]


def _decision_trace(
    policy: str,
    preferred: ProposedAction,
    candidates: list[ProposedAction],
    decision: AgentDecision,
) -> DecisionTrace:
    rejected = [
        RejectedCandidate(
            action=candidate,
            reason=f"{policy} deferred this catalog action; no world-model score was available.",
            score_breakdown={"agent_preference": 0.0, "qwen_signal": 0.0},
        )
        for candidate in candidates
        if candidate.action_id != preferred.action_id
    ]
    signal = "score_chasing_policy" if policy == "classic_autoresearch" else "verification_policy"
    return DecisionTrace(
        selected_action=preferred,
        agent_signal_used=True,
        world_model_signal_used=False,
        selector_policy_signal_used=True,
        causal_signal_type="selector_policy",
        observed_state_signal=signal,
        selector_policy_signal=signal,
        selected_score=None,
        score_breakdown={"agent_preference": 1.0, "qwen_signal": 0.0},
        qwen_suggested_action=None,
        catalog_validation="accepted",
        agent_rationale=decision.rationale,
        preferred_action_id=decision.preferred_action_id,
        causal_reason=(
            f"{policy} selected {preferred.model} without pre-compute world-model simulation. "
            "This is the baseline Lucky Loop is compared against."
        ),
        rejected_candidates=rejected,
    )


def run_classic_policy(policy: str, task: TaskSpec, max_experiments: int | None = None) -> list[ExperimentTrace]:
    max_experiments = max_experiments or task.budget_runs
    namespace = f"ablations/{policy}/{task.task_id}"
    runs_dir, reports_dir = _clean_namespace(namespace)
    traces: list[ExperimentTrace] = []
    seen: set[str] = set()
    state_summary = (
        f"Goal: {task.goal}\n"
        f"Classic baseline on sklearn {task.dataset}; no world-model prediction before compute."
    )

    for i in range(1, max_experiments + 1):
        run_id = f"run_{i:03d}"
        state_before = build_state(task, traces, i, max_experiments, state_summary)
        candidates = generate_candidates(task, state_before, traces, seen)
        if not candidates:
            break
        action = _choose_classic_action(policy, task, candidates, traces)
        seen.add(action_key(action))
        decision = _agent_decision(policy, task, state_before.state_id, candidates, action)
        prediction = _nop_prediction(action)
        actual = execute(action.command, cwd=ROOT)
        comparison = _nop_comparison(actual.status)
        verification = verify_sweep(actual.raw) if actual.raw.get("runs") else None
        top_model_summary = detect_top_models(traces, metric=task.primary_metric)
        decision_trace = _decision_trace(policy, action, candidates, decision)
        next_decision = (
            "Stop and report from observed evidence."
            if i >= max_experiments
            else "Continue with the baseline policy; no world-model prediction is available."
        )

        trace = ExperimentTrace(
            run_id=run_id,
            goal=task.goal,
            hypothesis=decision.working_hypothesis if traces else initial_hypothesis(),
            proposed_action=action,
            world_model_prediction=prediction,
            actual_result=actual,
            comparison=comparison,
            next_decision=next_decision,
            verification=verification,
            schema_version="3.0",
            planner_mode=policy,
            agent_backend=policy,
            agent_model="classic-autoresearch-baseline",
            agent_prompt_hash=prompt_hash(json.dumps(state_before.model_dump(), sort_keys=True)),
            agent_decision=decision,
            safety_validation=SafetyValidation(
                valid_agent_action=True,
                selected_action_id=action.action_id or "",
                validation_notes=["catalog action selected by deterministic baseline policy"],
            ),
            research_hypothesis=decision.working_hypothesis,
            state_before=state_before,
            candidate_actions=candidates,
            candidate_predictions=[
                CandidatePrediction(action=candidate, prediction=_nop_prediction(candidate), source="unknown")
                for candidate in candidates
            ],
            selected_action=action,
            decision_trace=decision_trace,
            claim_ledger_updates=entries_from_verification(run_id, verification),
            top_model_summary=top_model_summary,
            artifacts={
                "task_id": task.task_id,
                "policy": policy,
                "trace_path": f"runs/{namespace}/{run_id}.json",
                "report_path": f"reports/{namespace}/final_report.md",
                "claim_ledger_path": f"reports/{namespace}/claim_ledger.json",
            },
        )
        traces.append(trace)
        (runs_dir / f"{run_id}.json").write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        state_summary += f"\n{run_id}: {action.model} actual={_actual_metric(trace)} status={actual.status}"
        print(f"{policy}/{task.task_id}/{run_id}: {action.model} metric={_actual_metric(trace)}")

    write_calibration_report(traces, reports_dir / "world_model_calibration.md")
    write_claim_ledger(traces, reports_dir / "claim_ledger.json")
    generate_report(task.goal, traces, reports_dir / "final_report.md")
    return traces


def run_policy(policy: str, task: TaskSpec, max_experiments: int | None, operator_agent: str) -> list[ExperimentTrace]:
    if policy == "lucky_loop_full":
        namespace = f"ablations/{policy}/{task.task_id}"
        _clean_namespace(namespace)
        return run_lucky_loop(
            task=task,
            output_namespace=namespace,
            max_experiments=max_experiments,
            planner_mode="operator_driven",
            agent_backend=operator_agent,
        )
    return run_classic_policy(policy, task, max_experiments=max_experiments)


def load_policy_traces(policy: str, task: TaskSpec) -> list[ExperimentTrace]:
    run_dir = ROOT / "runs" / "ablations" / policy / task.task_id
    return [
        ExperimentTrace.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("run_*.json"))
    ]


def summarize_traces(policy: str, task_id: str, traces: list[ExperimentTrace]) -> dict:
    best = _best_single_run(traces)
    claimable = claimable_evidence_summary(traces)
    top_verifications = [trace for trace in traces if trace.proposed_action.model == "top_model_verification"]
    verification_traces = [trace for trace in traces if trace.verification]
    blocked = sum(1 for trace in verification_traces if trace.verification and not trace.verification.trustworthy)
    supported = sum(1 for trace in verification_traces if trace.verification and trace.verification.trustworthy)
    qwen_predictions = sum(
        1
        for trace in traces
        for candidate_prediction in trace.candidate_predictions
        if candidate_prediction.source == "qwen_agentworld"
    )
    unsupported_claims = 0
    if policy == "classic_autoresearch" and best:
        unsupported_claims = 1
    elif best and not top_verifications and not supported:
        unsupported_claims = 1
    return {
        "task": task_id,
        "policy": policy,
        "runs": len(traces),
        "best_single_run_model": best.proposed_action.model if best else None,
        "best_single_run_metric": _actual_metric(best) if best else None,
        "top_model_verification_performed": bool(top_verifications),
        "top_model_verification_runs": len(top_verifications),
        "claims_blocked": blocked,
        "supported_claims": supported,
        "unsupported_best_model_claims": unsupported_claims,
        "prediction_misses_logged": sum(1 for trace in traces if trace.comparison.unexpected_events),
        "world_model_prediction_count": qwen_predictions,
        "useful_world_model_decisions": sum(
            1
            for trace in traces
            if trace.decision_trace and trace.decision_trace.world_model_signal_used
        ),
        "best_verified_mean_score": claimable["best_verified_mean_score"],
        "best_claimable_score": claimable["best_claimable_score"],
        "runs_to_first_verification": claimable["runs_to_first_verification"],
        "total_runtime_seconds": claimable["total_runtime_seconds"],
        "compute_per_claimable_claim": claimable["compute_per_claimable_claim"],
        "non_claimable_runs": claimable["non_claimable_runs"],
        "non_claimable_runtime_seconds": claimable["non_claimable_runtime_seconds"],
        "wasted_score_chasing_runs": claimable["wasted_score_chasing_runs"],
        "wasted_score_chasing_runtime_seconds": claimable["wasted_score_chasing_runtime_seconds"],
        "runs_after_verification_needed": claimable["runs_after_verification_needed"],
        "runtime_after_verification_needed_seconds": claimable["runtime_after_verification_needed_seconds"],
        "qwen_triggered_verification": claimable["qwen_triggered_verification"],
        "qwen_skip_or_stop_recommended": claimable["qwen_skip_or_stop_recommended"],
        "qwen_recommended_action": claimable["recommended_action"],
        "qwen_stop_after_run": claimable["stop_after_run"],
        "qwen_stop_saved_remaining_runs": claimable["saved_remaining_runs"],
        "qwen_stop_saved_remaining_runtime_seconds": claimable["saved_remaining_runtime_seconds"],
        "qwen_choice_usefulness": None,
        "action_sequence": [trace.proposed_action.model for trace in traces],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _fmt_nullable(value) -> str:
    if value is None:
        return "∞"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _counterfactual_usefulness_by_task() -> dict[str, float]:
    path = ROOT / "reports" / "counterfactuals" / "counterfactual_evaluation.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_task: dict[str, list[bool]] = {}
    for row in payload.get("rows", []):
        task = row.get("task")
        if not task:
            continue
        by_task.setdefault(task, []).append(row.get("verdict", {}).get("overall_verdict") == "lucky_win")
    return {task: sum(values) / len(values) for task, values in by_task.items() if values}


def write_ablation_reports(rows: list[dict]) -> None:
    out_dir = ROOT / "reports" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    usefulness = _counterfactual_usefulness_by_task()
    for row in rows:
        if row["policy"] == "lucky_loop_full" and row["task"] in usefulness:
            row["qwen_choice_usefulness"] = usefulness[row["task"]]
    _write_json(out_dir / "world_model_ablation.json", {"schema_version": "1.0", "rows": rows})

    lines = [
        "# World Model Ablation",
        "",
        "This suite runs real sklearn experiments under three backend policies.",
        "",
        "- `classic_autoresearch`: agent policy runs experiments and can report a best single-run score without prospective simulation.",
        "- `classic_verified`: same no-world-model planner, but with deterministic top-model verification.",
        "- `lucky_loop_full`: agent-in-repo planner plus Qwen-AgentWorld predictions before compute and deterministic claim verification.",
        "",
        "| Task | Policy | Runs | Best single-run | Best verified mean | Best claimable | Runs to verification | Compute / claimable claim | Wasted score-chasing | Unsupported claims | Claims blocked | Qwen predictions | Qwen triggered verification | Qwen choice usefulness |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        qwen_usefulness = "" if row["qwen_choice_usefulness"] is None else f"{row['qwen_choice_usefulness']:.2%}"
        lines.append(
            f"| {row['task']} | {row['policy']} | {row['runs']} | "
            f"{_fmt_metric(row['best_single_run_metric'])} | "
            f"{_fmt_metric(row['best_verified_mean_score'])} | "
            f"{_fmt_metric(row['best_claimable_score'])} | "
            f"{_fmt_nullable(row['runs_to_first_verification'])} | "
            f"{_fmt_nullable(row['compute_per_claimable_claim'])} | "
            f"{row['wasted_score_chasing_runs']} | "
            f"{row['unsupported_best_model_claims']} | {row['claims_blocked']} | "
            f"{row['world_model_prediction_count']} | "
            f"{'yes' if row['qwen_triggered_verification'] else 'no'} | "
            f"{qwen_usefulness} |"
        )
    lines += [
        "",
        "## Claimable Evidence Per Compute",
        "",
        "`best_claimable_score` is strict: inconclusive verifier outcomes do not count. `∞` means no claim reached the trust ladder threshold.",
        "",
        "| Task | Policy | Best claimable | Runtime | Compute / claimable claim |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['policy']} | {_fmt_metric(row['best_claimable_score'])} | "
            f"{row['total_runtime_seconds']:.2f}s | {_fmt_nullable(row['compute_per_claimable_claim'])} |"
        )
    (out_dir / "world_model_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    paired = [
        "# Classic Autoresearch vs Lucky Loop",
        "",
        "Lucky Loop is not evaluated only by final score. The comparison asks whether the system predicted before compute, verified top models before claims, and avoided unsupported best-model reporting.",
        "",
    ]
    for task in sorted({row["task"] for row in rows}):
        task_rows = {row["policy"]: row for row in rows if row["task"] == task}
        classic = task_rows.get("classic_autoresearch")
        full = task_rows.get("lucky_loop_full")
        verified = task_rows.get("classic_verified")
        paired.append(f"## {task}")
        if classic and full:
            full_usefulness = "" if full["qwen_choice_usefulness"] is None else f"{full['qwen_choice_usefulness']:.2%}"
            paired.append(
                f"- Classic autoresearch best single-run={_fmt_metric(classic['best_single_run_metric'])}, "
                f"best claimable={_fmt_metric(classic['best_claimable_score']) or 'none'}, "
                f"unsupported best-model claims={classic['unsupported_best_model_claims']}, "
                f"wasted score-chasing runs={classic['wasted_score_chasing_runs']}, "
                f"Qwen predictions={classic['world_model_prediction_count']}."
            )
            paired.append(
                f"- Lucky Loop full best single-run={_fmt_metric(full['best_single_run_metric'])}, "
                f"best verified mean={_fmt_metric(full['best_verified_mean_score'])}, "
                f"best claimable={_fmt_metric(full['best_claimable_score']) or 'none'}, "
                f"unsupported best-model claims={full['unsupported_best_model_claims']}, "
                f"Qwen predictions={full['world_model_prediction_count']}, "
                f"claims blocked={full['claims_blocked']}, supported claims={full['supported_claims']}, "
                f"Qwen triggered verification={'yes' if full['qwen_triggered_verification'] else 'no'}, "
                f"Qwen choice usefulness={full_usefulness}."
            )
        if verified:
            paired.append(
                f"- Classic verified isolates the verifier contribution: top-model verification="
                f"{'yes' if verified['top_model_verification_performed'] else 'no'}, "
                f"Qwen predictions={verified['world_model_prediction_count']}."
            )
        paired.append("")
    (out_dir / "classic_vs_lucky_loop.md").write_text("\n".join(paired), encoding="utf-8")

    pitch = [
        "# Backend Pitch Summary",
        "",
        "Lucky Loop is a world-model-guided autoresearch backend. The agent proposes catalog actions, Qwen-AgentWorld predicts outcomes before compute, real sklearn experiments run, prediction-vs-reality is logged, and a deterministic verifier gates claims.",
        "",
        "## What the ablation proves",
        "",
        "- Classic autoresearch can find good single-run scores, but has no pre-compute prediction trace.",
        "- Classic verified shows the trust gate alone can reduce overclaims, but still lacks prospective simulation.",
        "- Lucky Loop full adds the missing world-model layer: every candidate is predicted before compute and every miss is logged.",
        "- Claimable-score metrics are strict: inconclusive verifier outcomes block robust claims rather than becoming claimable wins.",
        "",
        "## Artifacts",
        "",
        "- `reports/ablations/world_model_ablation.md`",
        "- `reports/ablations/classic_vs_lucky_loop.md`",
        "- `reports/ablations/world_model_ablation.json`",
        "- `reports/counterfactuals/counterfactual_evaluation.md`",
        "- `reports/budgeted_compute/budgeted_compute_evaluation.md`",
        "- `runs/ablations/*/*/run_*.json`",
        "- `runs/counterfactuals/*/*/*.json`",
    ]
    (ROOT / "reports" / "pitch_backend_summary.md").write_text("\n".join(pitch) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=TASK_PATHS)
    parser.add_argument("--policies", nargs="*", choices=POLICIES, default=POLICIES)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--operator-agent", default="codex_operator")
    parser.add_argument("--reuse-existing", action="store_true", help="Recompute reports from existing runs/ablations traces.")
    parser.add_argument(
        "--world-model",
        choices=["auto", "heuristic"],
        default="auto",
        help="auto uses the configured Qwen-AgentWorld endpoint; heuristic temporarily disables it.",
    )
    args = parser.parse_args()

    old_base = os.environ.get("LUCKYWORLD_SIMULATOR_BASE_URL")
    old_model = os.environ.get("LUCKYWORLD_SIMULATOR_MODEL")
    if args.world_model == "heuristic":
        os.environ.pop("LUCKYWORLD_SIMULATOR_BASE_URL", None)
        os.environ.pop("LUCKYWORLD_SIMULATOR_MODEL", None)

    rows: list[dict] = []
    try:
        for task_path in args.tasks:
            task = load_task(task_path)
            for policy in args.policies:
                if args.reuse_existing:
                    traces = load_policy_traces(policy, task)
                else:
                    traces = run_policy(policy, task, args.max_experiments, args.operator_agent)
                rows.append(summarize_traces(policy, task.task_id, traces))
    finally:
        if old_base is not None:
            os.environ["LUCKYWORLD_SIMULATOR_BASE_URL"] = old_base
        if old_model is not None:
            os.environ["LUCKYWORLD_SIMULATOR_MODEL"] = old_model

    write_ablation_reports(rows)
    print("Wrote reports/ablations/world_model_ablation.md")
    print("Wrote reports/ablations/classic_vs_lucky_loop.md")
    print("Wrote reports/pitch_backend_summary.md")


if __name__ == "__main__":
    main()
