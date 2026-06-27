from __future__ import annotations

from luckyloop.benchmark_metrics import (
    best_claimable_score,
    best_verified_mean_score,
    compute_per_claimable_claim,
    non_claimable_runs,
    qwen_triggered_verification,
    runs_after_verification_needed,
    runs_to_first_verification,
    stop_after_verification_opportunity,
    wasted_score_chasing_runs,
    wasted_score_chasing_runtime_seconds,
)
from luckyloop.schemas import (
    ActualResult,
    DecisionTrace,
    ExperimentTrace,
    Prediction,
    ProposedAction,
    ResearchState,
    TopModelSummary,
    Verification,
)


def _trace(
    run_id: str,
    model: str,
    *,
    metric: float | None = None,
    mean_metric: float | None = None,
    verification: Verification | None = None,
    needs_verification: bool = False,
    world_signal: bool = False,
) -> ExperimentTrace:
    raw = {}
    if mean_metric is not None:
        raw = {"metric": "accuracy", "best": {"mean_accuracy": mean_metric}}
    return ExperimentTrace(
        run_id=run_id,
        goal="g",
        hypothesis="h",
        proposed_action=ProposedAction(command="echo ok", model=model),
        world_model_prediction=Prediction(
            expected_metric="accuracy around 0.90-0.99",
            expected_runtime_seconds="under 10",
            risks=["seed variance may exceed apparent top-model gap"],
            action_specific_signal="verify top models before robust claim",
            claim_risk="single split claim risk",
        ),
        actual_result=ActualResult(status="success", accuracy=metric, runtime_seconds=1.0, raw=raw),
        comparison={"metric_match": True, "runtime_match": True, "unexpected_events": [], "lesson": "ok"},
        next_decision="continue",
        verification=verification,
        state_before=ResearchState(
            state_id=run_id,
            goal="g",
            top_model_summary=TopModelSummary(needs_robustness_verification=needs_verification),
        ),
        decision_trace=DecisionTrace(
            selected_action=ProposedAction(command="echo ok", model=model),
            world_model_signal_used=world_signal,
            causal_reason="world model predicted seed variance risk; verify top models",
        ),
    )


def test_claimable_score_is_strict_and_ignores_inconclusive():
    inconclusive = Verification(status="inconclusive", trustworthy=False)
    supported = Verification(status="supported", trustworthy=True)
    traces = [
        _trace("run_001", "top_model_verification", mean_metric=0.95, verification=inconclusive),
        _trace("run_002", "top_model_verification", mean_metric=0.93, verification=supported),
    ]
    assert best_verified_mean_score(traces) == 0.95
    assert best_claimable_score(traces) == 0.93


def test_compute_per_claimable_claim_is_none_without_trusted_claim():
    traces = [_trace("run_001", "top_model_verification", mean_metric=0.95, verification=Verification(status="inconclusive"))]
    assert compute_per_claimable_claim(traces) is None


def test_runs_to_first_verification_and_wasted_score_chasing():
    traces = [
        _trace("run_001", "svc", metric=0.97, needs_verification=False),
        _trace("run_002", "random_forest", metric=0.96, needs_verification=True),
        _trace("run_003", "top_model_verification", mean_metric=0.95, verification=Verification(status="inconclusive")),
    ]
    assert runs_to_first_verification(traces) == 3
    assert wasted_score_chasing_runs(traces) == 1
    assert wasted_score_chasing_runtime_seconds(traces) == 1.0
    assert runs_after_verification_needed(traces) == 1
    assert non_claimable_runs(traces) == 3


def test_qwen_triggered_verification_detects_world_model_signal():
    traces = [_trace("run_001", "top_model_verification", mean_metric=0.95, verification=Verification(status="inconclusive"), world_signal=True)]
    assert qwen_triggered_verification(traces) is True


def test_stop_after_verification_opportunity_recommends_stop_for_inconclusive_verifier():
    traces = [
        _trace("run_001", "svc", metric=0.97),
        _trace("run_002", "top_model_verification", mean_metric=0.95, verification=Verification(status="inconclusive")),
        _trace("run_003", "random_forest", metric=0.96),
    ]
    opportunity = stop_after_verification_opportunity(traces)
    assert opportunity["qwen_skip_or_stop_recommended"] is True
    assert opportunity["recommended_action"] == "stop_and_report"
    assert opportunity["stop_after_run"] == "run_002"
    assert opportunity["saved_remaining_runs"] == 1
