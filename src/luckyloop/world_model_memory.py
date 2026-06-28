from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import ExperimentTrace, ProposedAction, TaskSpec


def _actual_metric(trace: ExperimentTrace) -> float | None:
    if trace.actual_result.accuracy is not None:
        return trace.actual_result.accuracy
    metric = trace.actual_result.raw.get("metric", "accuracy")
    best = trace.actual_result.raw.get("best") or {}
    value = best.get(f"mean_{metric}")
    return float(value) if value is not None else None


def _action_family(model: str) -> str:
    if model in {"verification_sweep", "top_model_verification"}:
        return "verification"
    if "boost" in model or model == "random_forest":
        return "tree_ensemble"
    if model in {"logistic_regression", "svc"}:
        return "scale_sensitive"
    return model


def memory_item_from_trace(task: TaskSpec, trace: ExperimentTrace) -> dict[str, Any]:
    verification = trace.verification
    lesson_parts = [trace.comparison.lesson]
    if verification:
        lesson_parts.append(f"verifier={verification.status}")
        if verification.allowed_claim:
            lesson_parts.append(verification.allowed_claim)
    item = {
        "memory_id": f"{task.task_id}:{trace.run_id}:{trace.proposed_action.model}",
        "task_id": task.task_id,
        "dataset": task.dataset,
        "action_model": trace.proposed_action.model,
        "action_family": _action_family(trace.proposed_action.model),
        "action_params": trace.proposed_action.params,
        "predicted_metric": trace.world_model_prediction.expected_metric,
        "actual_metric": _actual_metric(trace),
        "runtime_seconds": trace.actual_result.runtime_seconds,
        "risks": trace.world_model_prediction.risks,
        "recommendation": trace.world_model_prediction.recommendation,
        "claim_impact": trace.world_model_prediction.claim_impact,
        "compute_value": trace.world_model_prediction.compute_value,
        "comparison_miss": bool(trace.comparison.unexpected_events),
        "verification_status": verification.status if verification else None,
        "lesson": "; ".join(part for part in lesson_parts if part),
    }
    return item


def build_memory_items(task: TaskSpec, traces: list[ExperimentTrace]) -> list[dict[str, Any]]:
    return [memory_item_from_trace(task, trace) for trace in traces]


def retrieve_similar_memories(
    task: TaskSpec,
    action: ProposedAction,
    traces: list[ExperimentTrace],
    limit: int = 3,
) -> list[dict[str, Any]]:
    target_family = _action_family(action.model)
    target_risks = " ".join(
        risk
        for trace in traces[-3:]
        for risk in trace.world_model_prediction.risks
    ).lower()
    scored = []
    for item in build_memory_items(task, traces):
        # Calibration integrity: never feed a candidate's OWN identical prior action (same model +
        # params) into its pre-compute prediction — that would leak the answer. Makes the no-leakage
        # boundary explicit here rather than relying on the caller's candidate de-duplication.
        if item["action_model"] == action.model and item["action_params"] == action.params:
            continue
        score = 0
        if item["dataset"] == task.dataset:
            score += 5
        if item["action_model"] == action.model:
            score += 6
        if item["action_family"] == target_family:
            score += 3
        item_text = " ".join([item.get("lesson", ""), " ".join(item.get("risks", []))]).lower()
        for keyword in ["seed", "variance", "scal", "overfit", "claim", "runtime", "noise"]:
            if keyword in target_risks and keyword in item_text:
                score += 1
        if item.get("comparison_miss"):
            score += 1
        if item.get("verification_status"):
            score += 1
        scored.append((score, item))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [item for score, item in scored if score > 0][:limit]


def write_memory_jsonl(task: TaskSpec, traces: list[ExperimentTrace], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, sort_keys=True) for item in build_memory_items(task, traces)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
