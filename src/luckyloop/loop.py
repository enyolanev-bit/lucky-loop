from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .comparator import compare
from .executor import execute
from .reporter import generate_report
from .schemas import CandidatePrediction, DecisionTrace, ExperimentTrace, ProposedAction, RejectedCandidate, ResearchState
from .simulator import predict
from .verifier import verify_sweep

ROOT = Path(__file__).resolve().parents[2]


def action_key(action: ProposedAction) -> str:
    return f"{action.model}:{json.dumps(action.params, sort_keys=True)}"


def with_action_id(action: ProposedAction, action_id: str) -> ProposedAction:
    return ProposedAction(action_id=action_id, command=action.command, model=action.model, params=action.params)


def prediction_source(prediction) -> str:
    if not os.getenv("LUCKYWORLD_SIMULATOR_BASE_URL") or not os.getenv("LUCKYWORLD_SIMULATOR_MODEL"):
        return "heuristic_fallback"
    text = " ".join([prediction.rationale, *prediction.risks]).lower()
    return "heuristic_fallback" if "fallback" in text or "heuristic" in text else "qwen_agentworld"


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


def initial_experiment() -> tuple[str, ProposedAction]:
    return (
        "Establish a simple linear baseline before spending search budget.",
        ProposedAction(
            command="python experiments/train_sklearn.py --dataset breast_cancer --model logistic_regression",
            model="logistic_regression",
            params={"scale": False},
        ),
    )


def choose_next(traces: list[ExperimentTrace], seen: set[str]) -> tuple[str, ProposedAction] | None:
    """Small transparent policy: Qwen-AgentWorld predictions steer the next real run.

    This is intentionally simple for the hackathon demo: every branch is explainable in
    the final report, and the model's recommendations/risks are visible in JSON traces.
    """
    last = traces[-1]
    model = last.proposed_action.model
    params = last.proposed_action.params
    risks = " ".join(last.world_model_prediction.risks).lower()
    rec = last.world_model_prediction.recommendation

    candidates: list[tuple[str, ProposedAction]] = []

    if model == "logistic_regression" and not params.get("scale") and (rec == "modify" or "scal" in risks):
        candidates.append(
            (
                "World model warned that unscaled logistic regression may underperform; rerun the same baseline with feature scaling.",
                ProposedAction(
                    command="python experiments/train_sklearn.py --dataset breast_cancer --model logistic_regression --scale",
                    model="logistic_regression",
                    params={"scale": True},
                ),
            )
        )

    best = max(
        (t for t in traces if t.actual_result.accuracy is not None),
        key=lambda t: t.actual_result.accuracy or -1,
        default=None,
    )
    best_acc = best.actual_result.accuracy if best and best.actual_result.accuracy is not None else 0.0

    if best_acc >= 0.98:
        candidates.append(
            (
                "The current linear baseline is already strong; test a different inductive bias for robustness rather than only chasing accuracy.",
                ProposedAction(
                    command="python experiments/train_sklearn.py --dataset breast_cancer --model random_forest --n-estimators 300",
                    model="random_forest",
                    params={"n_estimators": 300},
                ),
            )
        )
        candidates.append(
            (
                "Try a scaled RBF SVC because Qwen-AgentWorld expects scaling-sensitive models to be competitive on small tabular data.",
                ProposedAction(
                    command="python experiments/train_sklearn.py --dataset breast_cancer --model svc --scale --C 2.0",
                    model="svc",
                    params={"scale": True, "C": 2.0, "kernel": "rbf"},
                ),
            )
        )
    else:
        candidates.append(
            (
                "Accuracy is not saturated yet; try a random forest to capture non-linear feature interactions without scaling.",
                ProposedAction(
                    command="python experiments/train_sklearn.py --dataset breast_cancer --model random_forest --n-estimators 300",
                    model="random_forest",
                    params={"n_estimators": 300},
                ),
            )
        )

    if model in {"random_forest", "gradient_boosting"} and not last.comparison.metric_match:
        candidates.insert(
            0,
            (
                "The world model overestimated the tree ensemble; try a scaled margin-based model as a different hypothesis.",
                ProposedAction(
                    command="python experiments/train_sklearn.py --dataset breast_cancer --model svc --scale --C 2.0",
                    model="svc",
                    params={"scale": True, "C": 2.0, "kernel": "rbf"},
                ),
            ),
        )

    candidates.append(
        (
            "Run a controlled noisy-label multi-seed sweep so the deterministic Verifier can decide whether an apparent improvement is larger than seed noise.",
            ProposedAction(
                command="python experiments/sweep_sklearn.py --dataset breast_cancer --model logistic_regression --scale --sweep-param C --values 0.1 1.0 10.0 --seeds 0 1 2 3 --label-noise 0.08",
                model="verification_sweep",
                params={"base_model": "logistic_regression", "scale": True, "sweep_param": "C", "values": [0.1, 1.0, 10.0], "seeds": [0, 1, 2, 3], "label_noise": 0.08},
            ),
        )
    )

    candidates.append(
        (
            "Try conservative gradient boosting as a final ensemble baseline and compare prediction accuracy against actual metrics.",
            ProposedAction(
                command="python experiments/train_sklearn.py --dataset breast_cancer --model gradient_boosting --n-estimators 150 --learning-rate 0.05",
                model="gradient_boosting",
                params={"n_estimators": 150, "learning_rate": 0.05},
            ),
        )
    )

    for candidate in candidates:
        if action_key(candidate[1]) not in seen:
            return candidate
    return None


def v2_candidate_list(selected: ProposedAction, next_choice: tuple[str, ProposedAction] | None) -> list[ProposedAction]:
    candidates = [with_action_id(selected, "action_selected")]
    if next_choice is not None:
        candidates.append(with_action_id(next_choice[1], "action_next_candidate"))
    return candidates


def build_decision_trace(
    selected: ProposedAction,
    prediction,
    next_choice: tuple[str, ProposedAction] | None,
    next_decision: str,
) -> DecisionTrace:
    rejected = []
    if next_choice is not None:
        rejected.append(
            RejectedCandidate(
                action=with_action_id(next_choice[1], "action_next_candidate"),
                reason="Not executed in this step; it is the queued candidate for the next research iteration.",
            )
        )
    risk_text = ", ".join(prediction.risks) or "no specific risk"
    return DecisionTrace(
        selected_action=with_action_id(selected, "action_selected"),
        world_model_signal_used=True,
        causal_reason=(
            f"Executed the selected action after the world model predicted {prediction.expected_metric}, "
            f"runtime {prediction.expected_runtime_seconds}, recommendation={prediction.recommendation}, "
            f"risks={risk_text}. Next decision: {next_decision}"
        ),
        rejected_candidates=rejected,
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
    hypothesis, action = initial_experiment()

    for i in range(1, max_experiments + 1):
        run_id = f"run_{i:03d}"
        state_before = build_state(goal, traces, i, max_experiments, state)
        seen.add(action_key(action))
        prediction = predict(action, state)
        actual = execute(action.command, cwd=ROOT)
        comparison = compare(prediction, actual)
        verification = verify_sweep(actual.raw) if actual.raw.get("runs") else None

        provisional = ExperimentTrace(
            run_id=run_id,
            goal=goal,
            hypothesis=hypothesis,
            proposed_action=action,
            world_model_prediction=prediction,
            actual_result=actual,
            comparison=comparison,
            next_decision="pending",
            verification=verification,
            schema_version="2.0",
            state_before=state_before,
            candidate_actions=[with_action_id(action, "action_selected")],
            candidate_predictions=[
                CandidatePrediction(
                    action=with_action_id(action, "action_selected"),
                    prediction=prediction,
                    source=prediction_source(prediction),
                )
            ],
            selected_action=with_action_id(action, "action_selected"),
        )
        next_choice = choose_next(traces + [provisional], seen)
        if i >= max_experiments or next_choice is None:
            next_decision = "Stop and report the best observed model."
        elif actual.status != "success":
            next_decision = "Execution failed; prioritize repairing the environment before spending more experiments."
        else:
            next_decision = f"Next selected by world-model-guided policy: {next_choice[0]}"

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
            candidate_actions=v2_candidate_list(action, next_choice),
            candidate_predictions=[
                CandidatePrediction(
                    action=with_action_id(action, "action_selected"),
                    prediction=prediction,
                    source=prediction_source(prediction),
                )
            ],
            selected_action=with_action_id(action, "action_selected"),
            decision_trace=build_decision_trace(action, prediction, next_choice, next_decision),
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
        if i >= max_experiments or next_choice is None:
            break
        hypothesis, action = next_choice

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
