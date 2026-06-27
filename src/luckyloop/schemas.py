from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class ProposedAction(BaseModel):
    action_id: str | None = None
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
    status: Literal["missing_data", "inconclusive", "weakly_supported", "supported", "strongly_supported"] = "missing_data"
    metric: str = "accuracy"
    effect_size: float | None = None
    seed_noise: float | None = None
    effect_to_noise_ratio: float | None = None
    min_seed_count: int | None = None
    low_n_warning: bool = False
    trustworthy: bool = False
    best_config: str | None = None
    worst_config: str | None = None
    blocked_claim: str | None = None
    allowed_claim: str | None = None
    supported_claims: list[str] = Field(default_factory=list)
    inconclusive_findings: list[str] = Field(default_factory=list)
    rationale: str = ""


class ResearchState(BaseModel):
    state_id: str
    goal: str
    known_results: list[dict[str, Any]] = Field(default_factory=list)
    budget_remaining: int | None = None
    open_questions: list[str] = Field(default_factory=list)
    risks_to_check: list[str] = Field(default_factory=list)
    summary: str = ""


class CandidatePrediction(BaseModel):
    action: ProposedAction
    prediction: Prediction
    source: Literal["qwen_agentworld", "heuristic_fallback", "unknown"] = "unknown"


class RejectedCandidate(BaseModel):
    action: ProposedAction
    reason: str


class DecisionTrace(BaseModel):
    selected_action: ProposedAction
    world_model_signal_used: bool = False
    causal_reason: str
    rejected_candidates: list[RejectedCandidate] = Field(default_factory=list)


class ClaimLedgerEntry(BaseModel):
    claim_id: str
    claim: str
    status: Literal["allowed", "weakly_supported", "supported", "strongly_supported", "blocked", "inconclusive"]
    evidence_run_ids: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None
    allowed_rewrite: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class CalibrationMetrics(BaseModel):
    metric_interval_coverage: float | None = None
    runtime_interval_coverage: float | None = None
    metric_absolute_error: float | None = None
    runtime_relative_error: float | None = None
    prediction_miss_count: int = 0
    risk_recall: float | None = None
    recommendation_quality: float | None = None
    useful_decision_count: int = 0


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
    schema_version: str = "1.0"
    state_before: ResearchState | None = None
    candidate_actions: list[ProposedAction] = Field(default_factory=list)
    candidate_predictions: list[CandidatePrediction] = Field(default_factory=list)
    selected_action: ProposedAction | None = None
    decision_trace: DecisionTrace | None = None
    claim_ledger_updates: list[ClaimLedgerEntry] = Field(default_factory=list)
    calibration_metrics: CalibrationMetrics | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
