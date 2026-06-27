from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparator import compare
from .executor import execute
from .reporter import generate_report
from .schemas import ExperimentTrace, ProposedAction
from .simulator import predict

ROOT = Path(__file__).resolve().parents[2]


def action_key(action: ProposedAction) -> str:
    return f"{action.model}:{json.dumps(action.params, sort_keys=True)}"


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
        seen.add(action_key(action))
        prediction = predict(action, state)
        actual = execute(action.command, cwd=ROOT)
        comparison = compare(prediction, actual)

        provisional = ExperimentTrace(
            run_id=run_id,
            goal=goal,
            hypothesis=hypothesis,
            proposed_action=action,
            world_model_prediction=prediction,
            actual_result=actual,
            comparison=comparison,
            next_decision="pending",
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
