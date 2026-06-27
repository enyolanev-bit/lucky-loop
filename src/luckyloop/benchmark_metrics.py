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


def non_claimable_runs(traces: Iterable[ExperimentTrace]) -> int:
    return sum(1 for trace in traces if not (trace.verification and trace.verification.trustworthy))


def non_claimable_runtime_seconds(traces: Iterable[ExperimentTrace]) -> float:
    values = [
        trace.actual_result.runtime_seconds or 0.0
        for trace in traces
        if not (trace.verification and trace.verification.trustworthy)
    ]
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


def wasted_score_chasing_runtime_seconds(traces: Iterable[ExperimentTrace]) -> float:
    wasted = 0.0
    for trace in traces:
        if trace.proposed_action.model in VERIFICATION_MODELS:
            continue
        top_summary = trace.state_before.top_model_summary if trace.state_before else None
        if top_summary and top_summary.needs_robustness_verification:
            wasted += trace.actual_result.runtime_seconds or 0.0
    return round(wasted, 6)


def runs_after_verification_needed(traces: Iterable[ExperimentTrace]) -> int:
    return sum(
        1
        for trace in traces
        if trace.state_before
        and trace.state_before.top_model_summary
        and trace.state_before.top_model_summary.needs_robustness_verification
    )


def runtime_after_verification_needed_seconds(traces: Iterable[ExperimentTrace]) -> float:
    values = [
        trace.actual_result.runtime_seconds or 0.0
        for trace in traces
        if trace.state_before
        and trace.state_before.top_model_summary
        and trace.state_before.top_model_summary.needs_robustness_verification
    ]
    return round(sum(values), 6)


def stop_after_verification_opportunity(traces: list[ExperimentTrace]) -> dict:
    for index, trace in enumerate(traces):
        if trace.proposed_action.model not in VERIFICATION_MODELS:
            continue
        remaining = traces[index + 1 :]
        trustworthy = bool(trace.verification and trace.verification.trustworthy)
        if trustworthy:
            return {
                "qwen_skip_or_stop_recommended": False,
                "recommended_action": "continue",
                "reason": "first verifier result was claimable; no stop recommendation is needed",
                "stop_after_run": trace.run_id,
                "saved_remaining_runs": 0,
                "saved_remaining_runtime_seconds": 0.0,
            }
        return {
            "qwen_skip_or_stop_recommended": bool(remaining),
            "recommended_action": "stop_and_report" if remaining else "report",
            "reason": "strict best-model claim objective: verifier did not allow a robust claim, so remaining score-chasing budget should be skipped or reported as exploratory only",
            "stop_after_run": trace.run_id,
            "saved_remaining_runs": len(remaining),
            "saved_remaining_runtime_seconds": round(
                sum(t.actual_result.runtime_seconds or 0.0 for t in remaining),
                6,
            ),
        }
    return {
        "qwen_skip_or_stop_recommended": False,
        "recommended_action": "continue",
        "reason": "no verifier result has been observed yet",
        "stop_after_run": None,
        "saved_remaining_runs": 0,
        "saved_remaining_runtime_seconds": 0.0,
    }


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
        "non_claimable_runs": non_claimable_runs(traces),
        "non_claimable_runtime_seconds": non_claimable_runtime_seconds(traces),
        "wasted_score_chasing_runs": wasted_score_chasing_runs(traces),
        "wasted_score_chasing_runtime_seconds": wasted_score_chasing_runtime_seconds(traces),
        "runs_after_verification_needed": runs_after_verification_needed(traces),
        "runtime_after_verification_needed_seconds": runtime_after_verification_needed_seconds(traces),
        "qwen_triggered_verification": qwen_triggered_verification(traces),
        **stop_after_verification_opportunity(traces),
    }
