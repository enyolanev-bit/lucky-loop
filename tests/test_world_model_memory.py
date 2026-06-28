from __future__ import annotations

from luckyloop.schemas import ActualResult, Comparison, ExperimentTrace, Prediction, ProposedAction, TaskSpec
from luckyloop.world_model_memory import retrieve_similar_memories


def _trace(run_id: str, model: str, dataset: str = "breast_cancer") -> ExperimentTrace:
    return ExperimentTrace(
        run_id=run_id,
        goal="g",
        hypothesis="h",
        proposed_action=ProposedAction(command="echo ok", model=model, params={"dataset": dataset}),
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


def test_retrieve_similar_memories_prioritizes_same_dataset_and_model():
    task = TaskSpec(task_id="breast_cancer_accuracy", dataset="breast_cancer")
    traces = [
        _trace("run_001", "random_forest"),
        _trace("run_002", "svc"),
    ]
    action = ProposedAction(command="echo ok", model="random_forest", params={"dataset": "breast_cancer"})
    memories = retrieve_similar_memories(task, action, traces, limit=1)
    assert len(memories) == 1
    assert memories[0]["action_model"] == "random_forest"
    assert memories[0]["actual_metric"] == 0.95
