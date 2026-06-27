from __future__ import annotations

import argparse
import os
from pathlib import Path

from .calibration import write_calibration_report
from .comparator import compare
from .executor import execute
from .planner import action_key, generate_candidates, initial_hypothesis, predict_candidates, select_candidate
from .reporter import generate_report
from .schemas import ExperimentTrace, ResearchState
from .verifier import verify_sweep

ROOT = Path(__file__).resolve().parents[2]


def simulator_configured() -> bool:
    return bool(os.getenv("LUCKYWORLD_SIMULATOR_BASE_URL") and os.getenv("LUCKYWORLD_SIMULATOR_MODEL"))


def best_accuracy_results(traces: list[ExperimentTrace]) -> list[dict]:
    rows = []
    for t in traces:
        if t.actual_result.accuracy is not None:
            rows.append(
                {
                    "run_id": t.run_id,
                    "model": t.proposed_action.model,
                    "accuracy": t.actual_result.accuracy,
                    "f1": t.actual_result.f1,
                    "params": t.proposed_action.params,
                }
            )
        elif t.actual_result.raw.get("best"):
            rows.append(
                {
                    "run_id": t.run_id,
                    "model": t.proposed_action.model,
                    "best": t.actual_result.raw["best"],
                    "params": t.proposed_action.params,
                }
            )
    return rows


def open_questions_for(traces: list[ExperimentTrace]) -> list[str]:
    questions = ["Which candidate should receive the next compute budget?"]
    if any(t.actual_result.accuracy and t.actual_result.accuracy >= 0.98 for t in traces):
        questions.append("Is the best observed score robust across seeds or perturbations?")
    if not any(t.verification for t in traces):
        questions.append("Do apparent improvements survive an effect-vs-noise verifier?")
    if any(t.comparison.unexpected_events for t in traces):
        questions.append("What did the world model miss, and should the next action test that failure mode?")
    return questions


def risks_for(traces: list[ExperimentTrace]) -> list[str]:
    risks = ["single-run best score may not justify a scientific claim"]
    if any(t.actual_result.accuracy and t.actual_result.accuracy >= 0.98 for t in traces):
        risks.append("seed variance may exceed apparent model or hyperparameter effects")
    if any(t.comparison.unexpected_events for t in traces):
        risks.append("world-model predictions may miss quantitative details")
    return risks


def build_state(goal: str, traces: list[ExperimentTrace], run_index: int, max_experiments: int, summary: str) -> ResearchState:
    return ResearchState(
        state_id=f"state_{run_index:03d}",
        goal=goal,
        known_results=best_accuracy_results(traces),
        budget_remaining=max(max_experiments - run_index + 1, 0),
        open_questions=open_questions_for(traces),
        risks_to_check=risks_for(traces),
        summary=summary,
    )


def claim_updates_for(run_id: str, verification) -> list:
    if verification is None:
        return []
    entries = []
    for i, claim in enumerate(verification.supported_claims, start=1):
        entries.append(
            {
                "claim_id": f"{run_id}_claim_{i:03d}",
                "claim": claim,
                "status": "supported",
                "evidence_run_ids": [run_id],
                "metrics": {"effect_size": verification.effect_size, "seed_noise": verification.seed_noise},
            }
        )
    for i, finding in enumerate(verification.inconclusive_findings, start=len(entries) + 1):
        entries.append(
            {
                "claim_id": f"{run_id}_claim_{i:03d}",
                "claim": finding,
                "status": "blocked",
                "evidence_run_ids": [run_id],
                "blocked_reason": verification.rationale,
                "allowed_rewrite": finding,
                "metrics": {"effect_size": verification.effect_size, "seed_noise": verification.seed_noise},
            }
        )
    return entries


def run(goal: str, max_experiments: int = 5) -> list[ExperimentTrace]:
    runs_dir = ROOT / "runs"
    reports_dir = ROOT / "reports"
    runs_dir.mkdir(exist_ok=True)
    traces: list[ExperimentTrace] = []
    seen: set[str] = set()
    state = (
        f"Goal: {goal}\n"
        "No experiments have been run yet. Dataset: sklearn breast_cancer. "
        "Metric: validation accuracy and weighted F1."
    )

    for i in range(1, max_experiments + 1):
        run_id = f"run_{i:03d}"
        state_before = build_state(goal, traces, i, max_experiments, state)
        candidates = generate_candidates(state_before, traces, seen)
        if not candidates:
            break
        candidate_predictions = predict_candidates(state_before, candidates, simulator_configured())
        selected_candidate, decision_trace = select_candidate(state_before, candidate_predictions, traces)
        action = selected_candidate.action
        prediction = selected_candidate.prediction
        seen.add(action_key(action))
        actual = execute(action.command, cwd=ROOT)
        comparison = compare(prediction, actual)
        verification = verify_sweep(actual.raw) if actual.raw.get("runs") else None
        hypothesis = decision_trace.causal_reason if traces else initial_hypothesis()

        if i >= max_experiments:
            next_decision = "Stop and report the best observed model."
        elif actual.status != "success":
            next_decision = "Execution failed; prioritize repairing the environment before spending more experiments."
        else:
            next_decision = "Continue: regenerate candidates from the updated state and ask the world model to predict each option."

        trace = ExperimentTrace(
            run_id=run_id,
            goal=goal,
            hypothesis=hypothesis,
            proposed_action=action,
            world_model_prediction=prediction,
            actual_result=actual,
            comparison=comparison,
            next_decision=next_decision,
            verification=verification,
            schema_version="2.0",
            state_before=state_before,
            candidate_actions=candidates,
            candidate_predictions=candidate_predictions,
            selected_action=action,
            decision_trace=decision_trace,
            claim_ledger_updates=claim_updates_for(run_id, verification),
            artifacts={
                "trace_path": f"runs/{run_id}.json",
                "report_path": "reports/final_report.md",
            },
        )
        traces.append(trace)
        (runs_dir / f"{run_id}.json").write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        state += (
            f"\n{run_id}: {action.model} actual={actual.accuracy} "
            f"status={actual.status}; prediction={prediction.expected_metric}; "
            f"recommendation={prediction.recommendation}; lesson={comparison.lesson}; "
            f"next={next_decision}"
        )
        print(
            f"{run_id}: {action.model} acc={actual.accuracy} "
            f"predicted='{prediction.expected_metric}' rec={prediction.recommendation} "
            f"match={comparison.metric_match}"
        )
        if verification:
            print(
                f"{run_id}: verifier status={verification.status} "
                f"effect={verification.effect_size} seed_noise={verification.seed_noise} "
                f"trustworthy={verification.trustworthy}"
            )
        if i >= max_experiments:
            break

    write_calibration_report(traces, reports_dir / "world_model_calibration.md")
    generate_report(goal, traces, reports_dir / "final_report.md")
    return traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="Maximize validation accuracy on sklearn breast cancer dataset in five experiments.")
    ap.add_argument("--max-experiments", type=int, default=5)
    args = ap.parse_args()
    traces = run(args.goal, args.max_experiments)
    successful = [t for t in traces if t.actual_result.accuracy is not None]
    best = max(successful, key=lambda t: t.actual_result.accuracy or -1, default=None)
    print("Wrote", len(traces), "traces to runs/ and report to reports/final_report.md")
    if best:
        print(f"Best: {best.run_id} {best.proposed_action.model} accuracy={best.actual_result.accuracy:.4f}")


if __name__ == "__main__":
    main()
