from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class ProposedAction(BaseModel):
    command: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    expected_metric: str
    expected_runtime_seconds: str
    risks: list[str] = Field(default_factory=list)
    recommendation: Literal["run", "skip", "modify"] = "run"
    rationale: str = ""


class ActualResult(BaseModel):
    status: str
    accuracy: float | None = None
    f1: float | None = None
    runtime_seconds: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class Comparison(BaseModel):
    metric_match: bool
    runtime_match: bool
    unexpected_events: list[str] = Field(default_factory=list)
    lesson: str


class Verification(BaseModel):
    status: Literal["supported", "inconclusive", "missing_data"] = "missing_data"
    metric: str = "accuracy"
    effect_size: float | None = None
    seed_noise: float | None = None
    trustworthy: bool = False
    best_config: str | None = None
    supported_claims: list[str] = Field(default_factory=list)
    inconclusive_findings: list[str] = Field(default_factory=list)
    rationale: str = ""


class ExperimentTrace(BaseModel):
    run_id: str
    goal: str
    hypothesis: str
    proposed_action: ProposedAction
    world_model_prediction: Prediction
    actual_result: ActualResult
    comparison: Comparison
    next_decision: str
    verification: Verification | None = None
