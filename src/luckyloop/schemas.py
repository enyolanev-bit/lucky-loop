from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class ProposedAction(BaseModel):
    action_id: str | None = None
    command: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)


class SweepSpec(BaseModel):
    model: str
    param: str
    values: list[Any]
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    scale: bool = False
    label_noise: float = 0.0


class TaskSpec(BaseModel):
    task_id: str
    dataset: str
    problem_type: Literal["classification"] = "classification"
    primary_metric: str = "accuracy"
    secondary_metrics: list[str] = Field(default_factory=lambda: ["f1"])
    budget_runs: int = 6
    models: list[str] = Field(default_factory=list)
    candidate_space: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)
    sweeps: list[SweepSpec] = Field(default_factory=list)
    goal: str = ""
    notes: list[str] = Field(default_factory=list)


class TopModelCandidate(BaseModel):
    run_id: str
    model: str
    model_key: str
    metric: float
    params: dict[str, Any] = Field(default_factory=dict)


class TopModelSummary(BaseModel):
    best_observed_model: str | None = None
    best_observed_metric: float | None = None
    top_models: list[TopModelCandidate] = Field(default_factory=list)
    top_gap: float | None = None
    needs_robustness_verification: bool = False
    reason: str = ""
    verification_action_key: str | None = None


class Prediction(BaseModel):
    expected_metric: str
    expected_runtime_seconds: str
    risks: list[str] = Field(default_factory=list)
    recommendation: Literal["run", "skip", "modify"] = "run"
    rationale: str = ""
    action_specific_signal: str = ""
    claim_risk: str = ""


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
    top_model_summary: TopModelSummary | None = None
    summary: str = ""


class CandidatePrediction(BaseModel):
    action: ProposedAction
    prediction: Prediction
    source: Literal["qwen_agentworld", "heuristic_fallback", "unknown"] = "unknown"


class RejectedCandidate(BaseModel):
    action: ProposedAction
    reason: str
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class DecisionTrace(BaseModel):
    selected_action: ProposedAction
    world_model_signal_used: bool = False
    selector_policy_signal_used: bool = False
    causal_signal_type: Literal[
        "world_model_prediction",
        "selector_policy",
        "mixed",
        "demo_policy",
        "unknown",
    ] = "unknown"
    observed_state_signal: str = ""
    world_model_signal: str = ""
    selector_policy_signal: str = ""
    selected_score: float | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    qwen_suggested_action: str | None = None
    catalog_validation: str | None = None
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
    top_model_summary: TopModelSummary | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
