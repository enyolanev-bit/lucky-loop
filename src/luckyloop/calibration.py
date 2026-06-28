from __future__ import annotations

import re
from pathlib import Path

from .schemas import CalibrationMetrics, ExperimentTrace


def _range_from_text(text: str) -> tuple[float, float] | None:
    nums = [float(x) for x in re.findall(r"0\.\d+|1\.0+", text)]
    if len(nums) >= 2:
        return min(nums), max(nums)
    if len(nums) == 1:
        return max(0.0, nums[0] - 0.03), min(1.0, nums[0] + 0.03)
    return None


def _runtime_upper_bound(text: str) -> float | None:
    match = re.search(r"under\s+(\d+(?:\.\d+)?)", text.lower())
    if match:
        return float(match.group(1))
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if nums:
        return max(nums)
    return None


def _actual_metric(trace: ExperimentTrace) -> float | None:
    if trace.actual_result.accuracy is not None:
        return trace.actual_result.accuracy
    best = trace.actual_result.raw.get("best")
    metric = trace.actual_result.raw.get("metric", "accuracy")
    if best and best.get(f"mean_{metric}") is not None:
        return float(best[f"mean_{metric}"])
    return None


def _metric_error(trace: ExperimentTrace) -> float | None:
    actual = _actual_metric(trace)
    rng = (
        tuple(trace.world_model_prediction.expected_metric_range)
        if trace.world_model_prediction.expected_metric_range
        else _range_from_text(trace.world_model_prediction.expected_metric)
    )
    if actual is None or rng is None:
        return None
    lo, hi = rng
    if lo <= actual <= hi:
        return 0.0
    return min(abs(actual - lo), abs(actual - hi))


def _runtime_relative_error(trace: ExperimentTrace) -> float | None:
    actual = trace.actual_result.runtime_seconds
    upper = (
        max(trace.world_model_prediction.expected_runtime_range_seconds)
        if trace.world_model_prediction.expected_runtime_range_seconds
        else _runtime_upper_bound(trace.world_model_prediction.expected_runtime_seconds)
    )
    if actual is None or upper is None or upper == 0:
        return None
    if actual <= upper:
        return 0.0
    return (actual - upper) / upper


def _risk_hit(trace: ExperimentTrace) -> bool:
    predicted = " ".join(trace.world_model_prediction.risks).lower()
    observed = " ".join(trace.comparison.unexpected_events).lower()
    if trace.verification and trace.verification.status == "inconclusive":
        observed += " seed variance noise non-robust inconclusive"
    if trace.verification and trace.verification.rationale:
        observed += " " + trace.verification.rationale.lower()
    keywords = ["seed", "noise", "non-robust", "runtime", "overfit", "scal", "convergence"]
    keywords += ["leak", "protocol", "misleading", "imbalance", "balanced", "f1", "metric"]
    return any(k in predicted and k in observed for k in keywords)


def _useful_decision(trace: ExperimentTrace) -> bool:
    if not trace.decision_trace or not trace.decision_trace.world_model_signal_used:
        return False
    decision = trace.decision_trace.causal_reason.lower()
    keywords = [
        "scaling",
        "scaled",
        "robustness",
        "seed",
        "variance",
        "baseline",
        "inductive bias",
        "trust-ladder",
        "leakage",
        "protocol",
        "metric misuse",
        "accuracy-only",
    ]
    return any(k in decision for k in keywords)


def compute_world_model_calibration(traces: list[ExperimentTrace]) -> CalibrationMetrics:
    metric_hits = []
    runtime_hits = []
    metric_errors = []
    runtime_errors = []
    risk_hits = []

    for trace in traces:
        actual = _actual_metric(trace)
        metric_range = (
            tuple(trace.world_model_prediction.expected_metric_range)
            if trace.world_model_prediction.expected_metric_range
            else _range_from_text(trace.world_model_prediction.expected_metric)
        )
        if actual is not None and metric_range is not None:
            lo, hi = metric_range
            metric_hits.append(lo <= actual <= hi)
            err = _metric_error(trace)
            if err is not None:
                metric_errors.append(err)

        upper = (
            max(trace.world_model_prediction.expected_runtime_range_seconds)
            if trace.world_model_prediction.expected_runtime_range_seconds
            else _runtime_upper_bound(trace.world_model_prediction.expected_runtime_seconds)
        )
        if trace.actual_result.runtime_seconds is not None and upper is not None:
            runtime_hits.append(trace.actual_result.runtime_seconds <= upper)
            err = _runtime_relative_error(trace)
            if err is not None:
                runtime_errors.append(err)

        if trace.world_model_prediction.risks:
            risk_hits.append(_risk_hit(trace))

    prediction_miss_count = sum(1 for t in traces if t.comparison.unexpected_events)
    useful_decision_count = sum(1 for t in traces if _useful_decision(t))
    high_claim_impact_verification_count = sum(
        1
        for t in traces
        if t.world_model_prediction.claim_impact == "high"
        and t.proposed_action.model in {"verification_sweep", "top_model_verification", "stop_and_report"}
    )
    skip_or_stop_recommendation_count = sum(
        1 for t in traces if t.world_model_prediction.recommendation in {"skip", "stop_and_report"}
    )
    memory_augmented_prediction_count = sum(1 for t in traces if t.world_model_prediction.memory_example_ids)
    few_shot_augmented_prediction_count = sum(1 for t in traces if t.world_model_prediction.few_shot_example_ids)

    return CalibrationMetrics(
        metric_interval_coverage=(sum(metric_hits) / len(metric_hits)) if metric_hits else None,
        runtime_interval_coverage=(sum(runtime_hits) / len(runtime_hits)) if runtime_hits else None,
        metric_absolute_error=(sum(metric_errors) / len(metric_errors)) if metric_errors else None,
        runtime_relative_error=(sum(runtime_errors) / len(runtime_errors)) if runtime_errors else None,
        prediction_miss_count=prediction_miss_count,
        risk_recall=(sum(risk_hits) / len(risk_hits)) if risk_hits else None,
        recommendation_quality=None,
        useful_decision_count=useful_decision_count,
        high_claim_impact_verification_count=high_claim_impact_verification_count,
        skip_or_stop_recommendation_count=skip_or_stop_recommendation_count,
        memory_augmented_prediction_count=memory_augmented_prediction_count,
        few_shot_augmented_prediction_count=few_shot_augmented_prediction_count,
    )


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_calibration_report(traces: list[ExperimentTrace], path: Path) -> CalibrationMetrics:
    metrics = compute_world_model_calibration(traces)
    lines = [
        "# World Model Calibration",
        "",
        "Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.",
        "",
        "## Summary",
        "",
        f"- Metric interval coverage: {_fmt_ratio(metrics.metric_interval_coverage)}",
        f"- Runtime interval coverage: {_fmt_ratio(metrics.runtime_interval_coverage)}",
        f"- Mean metric absolute error outside interval: {_fmt_float(metrics.metric_absolute_error)}",
        f"- Mean runtime relative error above bound: {_fmt_ratio(metrics.runtime_relative_error)}",
        f"- Prediction miss count: {metrics.prediction_miss_count}",
        f"- Risk recall approximation: {_fmt_ratio(metrics.risk_recall)}",
        f"- Useful decision signals: {metrics.useful_decision_count}/{len(traces)}",
        f"- High claim-impact verification/stop decisions: {metrics.high_claim_impact_verification_count}",
        f"- Skip/stop recommendations: {metrics.skip_or_stop_recommendation_count}",
        f"- Memory-augmented predictions: {metrics.memory_augmented_prediction_count}/{len(traces)}",
        f"- Few-shot-augmented predictions: {metrics.few_shot_augmented_prediction_count}/{len(traces)}",
        "",
        "## Prediction vs Reality",
        "",
        "| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |",
        "|---|---|---|---:|---|---|---:|---|---|",
    ]

    for trace in traces:
        actual_metric = _actual_metric(trace)
        metric_hit = trace.comparison.metric_match if actual_metric is not None else False
        runtime_hit = trace.comparison.runtime_match
        misses = "; ".join(trace.comparison.unexpected_events) if trace.comparison.unexpected_events else ""
        actual_runtime = trace.actual_result.runtime_seconds
        lines.append(
            "| "
            f"{trace.run_id} | "
            f"{trace.proposed_action.model} | "
            f"{trace.world_model_prediction.expected_metric} | "
            f"{'' if actual_metric is None else f'{actual_metric:.4f}'} | "
            f"{'yes' if metric_hit else 'no'} | "
            f"{trace.world_model_prediction.expected_runtime_seconds} | "
            f"{'' if actual_runtime is None else f'{actual_runtime:.2f}s'} | "
            f"{'yes' if runtime_hit else 'no'} | "
            f"{misses} |"
        )

    lines += [
        "",
        "## Risk Signals",
        "",
        "| Run | Predicted risks | Observed evidence | Risk hit |",
        "|---|---|---|---|",
    ]
    for trace in traces:
        observed = list(trace.comparison.unexpected_events)
        if trace.verification and trace.verification.status == "inconclusive":
            observed.append(trace.verification.rationale)
        lines.append(
            "| "
            f"{trace.run_id} | "
            f"{'; '.join(trace.world_model_prediction.risks) or 'none'} | "
            f"{'; '.join(observed) or 'none'} | "
            f"{'yes' if _risk_hit(trace) else 'no'} |"
        )

    lines += [
        "",
        "## Decision Usefulness",
        "",
        "| Run | Selected action | World-model signal used | Decision useful | Reason |",
        "|---|---|---|---|---|",
    ]
    for trace in traces:
        decision = trace.decision_trace.causal_reason if trace.decision_trace else trace.next_decision
        lines.append(
            "| "
            f"{trace.run_id} | "
            f"{trace.proposed_action.model} | "
            f"{'yes' if trace.decision_trace and trace.decision_trace.world_model_signal_used else 'no'} | "
            f"{'yes' if _useful_decision(trace) else 'no'} | "
            f"{decision} |"
        )

    lines += [
        "",
        "## Prompt Context",
        "",
        "| Run | Prompt version | Schema version | Few-shot examples | Retrieved memory examples | Claim impact | Compute value | Recommendation |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for trace in traces:
        lines.append(
            "| "
            f"{trace.run_id} | "
            f"{trace.world_model_prediction.prompt_version or trace.prompt_version or 'legacy'} | "
            f"{trace.world_model_prediction.world_model_schema_version or trace.world_model_schema_version or 'legacy'} | "
            f"{len(trace.world_model_prediction.few_shot_example_ids)} | "
            f"{len(trace.world_model_prediction.memory_example_ids)} | "
            f"{trace.world_model_prediction.claim_impact} | "
            f"{trace.world_model_prediction.compute_value} | "
            f"{trace.world_model_prediction.recommendation} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return metrics
