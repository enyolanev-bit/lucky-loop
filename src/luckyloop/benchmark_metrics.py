from __future__ import annotations

from math import isfinite
from typing import Iterable

from .schemas import ExperimentTrace


VERIFICATION_MODELS = {"verification_sweep", "top_model_verification"}


def actual_metric(trace: ExperimentTrace) -> float | None:
    if trace.actual_result.accuracy is not None:
        return trace.actual_result.accuracy
    metric = trace.actual_result.raw.get("metric", "accuracy")
    best = trace.actual_result.raw.get("best") or {}
    value = best.get(f"mean_{metric}")
    return float(value) if value is not None else None


def best_single_run(traces: Iterable[ExperimentTrace]) -> ExperimentTrace | None:
    singles = [
        trace
        for trace in traces
        if actual_metric(trace) is not None and trace.proposed_action.model not in VERIFICATION_MODELS
    ]
    return max(singles, key=lambda trace: actual_metric(trace) or -1, default=None)


def best_verified_mean_score(traces: Iterable[ExperimentTrace]) -> float | None:
    values = []
    for trace in traces:
        if not trace.verification:
            continue
        value = actual_metric(trace)
        if value is not None:
            values.append(value)
    return max(values, default=None)


def best_claimable_score(traces: Iterable[ExperimentTrace]) -> float | None:
    values = []
    for trace in traces:
        if not trace.verification or not trace.verification.trustworthy:
            continue
        value = actual_metric(trace)
        if value is not None:
            values.append(value)
    return max(values, default=None)


def runs_to_first_verification(traces: list[ExperimentTrace]) -> int | None:
    for index, trace in enumerate(traces, start=1):
        if trace.proposed_action.model in VERIFICATION_MODELS:
            return index
    return None


def total_runtime_seconds(traces: Iterable[ExperimentTrace]) -> float:
    values = [trace.actual_result.runtime_seconds for trace in traces if trace.actual_result.runtime_seconds is not None]
    return round(sum(values), 6)


def trusted_claim_count(traces: Iterable[ExperimentTrace]) -> int:
    return sum(1 for trace in traces if trace.verification and trace.verification.trustworthy)


def compute_per_claimable_claim(traces: list[ExperimentTrace]) -> float | None:
    count = trusted_claim_count(traces)
    if count <= 0:
        return None
    value = total_runtime_seconds(traces) / count
    return round(value, 6) if isfinite(value) else None


def wasted_score_chasing_runs(traces: Iterable[ExperimentTrace]) -> int:
    wasted = 0
    for trace in traces:
        if trace.proposed_action.model in VERIFICATION_MODELS:
            continue
        top_summary = trace.state_before.top_model_summary if trace.state_before else None
        if top_summary and top_summary.needs_robustness_verification:
            wasted += 1
    return wasted


def qwen_triggered_verification(traces: Iterable[ExperimentTrace]) -> bool:
    keywords = ["seed", "variance", "robust", "claim", "top", "single split", "verification"]
    for trace in traces:
        if trace.proposed_action.model not in VERIFICATION_MODELS or not trace.decision_trace:
            continue
        if not trace.decision_trace.world_model_signal_used:
            continue
        text = " ".join(
            [
                trace.decision_trace.world_model_signal,
                trace.decision_trace.causal_reason,
                trace.world_model_prediction.action_specific_signal,
                trace.world_model_prediction.claim_risk,
                " ".join(trace.world_model_prediction.risks),
            ]
        ).lower()
        if any(keyword in text for keyword in keywords):
            return True
    return False


def claimable_evidence_summary(traces: list[ExperimentTrace]) -> dict:
    return {
        "best_verified_mean_score": best_verified_mean_score(traces),
        "best_claimable_score": best_claimable_score(traces),
        "runs_to_first_verification": runs_to_first_verification(traces),
        "total_runtime_seconds": total_runtime_seconds(traces),
        "compute_per_claimable_claim": compute_per_claimable_claim(traces),
        "wasted_score_chasing_runs": wasted_score_chasing_runs(traces),
        "qwen_triggered_verification": qwen_triggered_verification(traces),
    }
