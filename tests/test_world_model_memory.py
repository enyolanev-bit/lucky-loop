from __future__ import annotations

from luckyloop.schemas import ActualResult, Comparison, ExperimentTrace, Prediction, ProposedAction, TaskSpec
from luckyloop.world_model_memory import retrieve_similar_memories


def _trace(run_id: str, model: str, params: dict | None = None, dataset: str = "breast_cancer") -> ExperimentTrace:
    return ExperimentTrace(
        run_id=run_id,
        goal="g",
        hypothesis="h",
        proposed_action=ProposedAction(command="echo ok", model=model, params=params or {"dataset": dataset}),
        world_model_prediction=Prediction(
            expected_metric="accuracy around 0.95-0.98",
            expected_runtime_seconds="under 5",
            risks=["seed variance may exceed apparent model gap"],
            recommendation="run",
            claim_impact="low",
            compute_value="medium",
        ),
        actual_result=ActualResult(status="success", accuracy=0.95, runtime_seconds=1.0),
        comparison=Comparison(metric_match=False, runtime_match=True, unexpected_events=["accuracy miss"], lesson="model was overestimated"),
        next_decision="continue",
    )


def test_retrieve_similar_memories_prioritizes_same_model_different_params():
    """Same model with DIFFERENT params is legitimate context and should rank highest."""
    task = TaskSpec(task_id="breast_cancer_accuracy", dataset="breast_cancer")
    traces = [
        _trace("run_001", "random_forest", params={"dataset": "breast_cancer", "n_estimators": 300}),
        _trace("run_002", "svc", params={"dataset": "breast_cancer"}),
    ]
    action = ProposedAction(command="echo ok", model="random_forest", params={"dataset": "breast_cancer", "n_estimators": 100})
    memories = retrieve_similar_memories(task, action, traces, limit=1)
    assert len(memories) == 1
    assert memories[0]["action_model"] == "random_forest"
    assert memories[0]["actual_metric"] == 0.95


def test_excludes_identical_prior_action_no_leakage():
    """Calibration integrity: a candidate's OWN identical prior action must never appear in its
    retrieved memory, or its actual_metric would leak into the pre-compute prediction."""
    task = TaskSpec(task_id="breast_cancer_accuracy", dataset="breast_cancer")
    identical_params = {"dataset": "breast_cancer", "C": 1.0}
    traces = [
        _trace("run_001", "logistic_regression", params=identical_params),  # identical to the candidate
        _trace("run_002", "svc", params={"dataset": "breast_cancer"}),
    ]
    action = ProposedAction(command="echo ok", model="logistic_regression", params=dict(identical_params))
    memories = retrieve_similar_memories(task, action, traces, limit=5)
    keys = {(m["action_model"], tuple(sorted(m["action_params"].items()))) for m in memories}
    assert ("logistic_regression", tuple(sorted(identical_params.items()))) not in keys


def test_empty_traces_returns_empty():
    task = TaskSpec(task_id="breast_cancer_accuracy", dataset="breast_cancer")
    action = ProposedAction(command="echo ok", model="random_forest", params={"dataset": "breast_cancer"})
    assert retrieve_similar_memories(task, action, [], limit=3) == []
