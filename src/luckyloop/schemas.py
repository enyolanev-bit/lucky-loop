from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class ProposedAction(BaseModel):
    action_id: str | None = None
    command: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)


class ResearchAction(BaseModel):
    action_id: str
    kind: Literal[
        "literature_review",
        "single_experiment",
        "multi_seed_verification",
        "protocol_probe",
        "ablation",
        "counterfactual",
        "stop_and_report",
    ]
    task_id: str | None = None
    description: str
    command: str | None = None
    expected_claim_change: str = ""
    cost_class: Literal["free", "cheap", "moderate", "expensive"] = "cheap"
    risk_focus: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)


class ResearchProgram(BaseModel):
    schema_version: str = "1.0"
    question: str
    goal: str
    selected_tasks: list[str]
    literature_gaps: list[dict[str, Any]] = Field(default_factory=list)
    success_metrics: list[str] = Field(default_factory=list)
    candidate_research_actions: list[ResearchAction] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)


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
    top_model_verification_min_single_runs: int = 5
    top_model_verification_seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
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


class PredictedNextState(BaseModel):
    likely_best_model: str | None = None
    expected_metric_delta: str = "unknown"
    uncertainty_reduction: Literal["low", "medium", "high"] = "medium"
    claim_status_after_action: Literal[
        "observation_only",
        "needs_verification",
        "likely_inconclusive",
        "likely_supported",
        "report_ready",
    ] = "observation_only"
    likely_next_open_questions: list[str] = Field(default_factory=list)
    recommended_followup: str = ""
    expected_compute_cost_seconds: float | None = None
    expected_research_value: Literal["low", "medium", "high"] = "medium"


class Prediction(BaseModel):
    expected_metric: str
    expected_runtime_seconds: str
    risks: list[str] = Field(default_factory=list)
    recommendation: Literal["run", "skip", "modify", "verify", "stop_and_report"] = "run"
    rationale: str = ""
    action_specific_signal: str = ""
    claim_risk: str = ""
    expected_metric_range: list[float] | None = None
    expected_runtime_range_seconds: list[float] | None = None
    claim_impact: Literal["low", "medium", "high"] = "medium"
    compute_value: Literal["low", "medium", "high"] = "medium"
    why_this_action_changes_claims: str = ""
    why_this_action_may_be_wasteful: str = ""
    risk_predictions: list[str] = Field(default_factory=list)
    stop_condition: str = ""
    predicted_next_state: PredictedNextState = Field(default_factory=PredictedNextState)
    expected_value_of_information: float = 0.5
    expected_claim_resolution: float = 0.5
    cost_aware_recommendation_reason: str = ""
    predicted_observation: str = ""
    expected_claim_delta: Literal[
        "none",
        "adds_observation",
        "reduces_uncertainty",
        "enables_claim",
        "blocks_or_rewrites_claim",
        "report_ready",
    ] = "adds_observation"
    protocol_risks: list[str] = Field(default_factory=list)
    compute_waste_risk: float = 0.5
    why_not_classic_autoresearch: str = ""
    memory_example_ids: list[str] = Field(default_factory=list)
    few_shot_example_ids: list[str] = Field(default_factory=list)
    prompt_version: str | None = None
    world_model_schema_version: str | None = None


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


class AgentDecision(BaseModel):
    research_question: str
    working_hypothesis: str
    candidate_action_ids: list[str] = Field(default_factory=list)
    preferred_action_id: str
    rationale: str
    expected_evidence_needed: str
    claim_risk: str
    stop_or_continue: Literal["continue", "stop_and_report"] = "continue"


class SafetyValidation(BaseModel):
    valid_agent_action: bool = True
    selected_action_id: str
    selection_overrode_agent: bool = False
    override_reason: str | None = None
    validation_notes: list[str] = Field(default_factory=list)


class RejectedCandidate(BaseModel):
    action: ProposedAction
    reason: str
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class DecisionTrace(BaseModel):
    selected_action: ProposedAction
    agent_signal_used: bool = False
    world_model_signal_used: bool = False
    selector_policy_signal_used: bool = False
    causal_signal_type: Literal[
        "agent_decision",
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
    agent_rationale: str = ""
    preferred_action_id: str | None = None
    selection_overrode_agent: bool = False
    override_reason: str | None = None
    world_model_decision_basis: str = ""
    classic_counterfactual_action_id: str | None = None
    why_world_model_mattered: str = ""
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


class LabQuestion(BaseModel):
    question: str
    study_id: str
    domain: str = "ml_research_validity"
    budget: int = 8
    require_qwen: bool = False
    success_criteria: list[str] = Field(default_factory=list)


class ResearchHypothesis(BaseModel):
    hypothesis_id: str
    claim_candidate: str
    why_it_matters: str
    literature_gap_ids: list[str] = Field(default_factory=list)
    falsification_condition: str
    minimum_evidence_needed: str


class ProtocolSpec(BaseModel):
    protocol_id: str
    hypothesis_id: str
    scientific_goal: str
    dataset: str
    conditions: list[str]
    controlled_variables: list[str] = Field(default_factory=list)
    manipulated_variable: str
    primary_metric: str = "balanced_accuracy"
    secondary_metrics: list[str] = Field(default_factory=lambda: ["accuracy", "f1"])
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    expected_artifacts: list[str] = Field(default_factory=list)
    protocol_risks: list[str] = Field(default_factory=list)
    claim_enabled_if: str = ""
    claim_blocked_if: str = ""


class DatasetCandidate(BaseModel):
    dataset_id: str
    source: Literal["huggingface", "openml", "sklearn"]
    name: str
    url: str = ""
    description: str = ""
    license: str | None = None
    task_tags: list[str] = Field(default_factory=list)
    score: float = 0.0
    score_reasons: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    fallback_rank: int = 0


class DatasetAudit(BaseModel):
    dataset_id: str
    source: Literal["huggingface", "openml", "sklearn"]
    status: Literal["accepted", "rejected"]
    reason: str = ""
    local_path: str | None = None
    target_column: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    n_rows: int = 0
    n_features: int = 0
    class_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class GeneratedResearchProtocol(BaseModel):
    protocol_id: str = "generated_ml_research_protocol"
    question: str
    hypothesis: str
    dataset_id: str
    dataset_source: Literal["huggingface", "openml", "sklearn"]
    target_column: str
    feature_columns: list[str] = Field(default_factory=list)
    task_type: Literal["classification"] = "classification"
    baseline_models: list[str] = Field(default_factory=lambda: ["logistic_regression"])
    candidate_models: list[str] = Field(default_factory=lambda: ["logistic_regression", "random_forest", "svc"])
    primary_metric: str = "balanced_accuracy"
    secondary_metrics: list[str] = Field(default_factory=lambda: ["accuracy", "f1_weighted"])
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    split_strategy: str = "stratified_train_test_split"
    ablations: list[str] = Field(default_factory=list)
    claim_enabled_if: str = "The best candidate model beats the strongest baseline by more than seed noise across repeated seeds."
    claim_blocked_if: str = "The best candidate effect is less than or equal to seed noise, or any protocol warning invalidates the comparison."
    risk_controls: list[str] = Field(default_factory=list)


class LiteratureQueryPlan(BaseModel):
    domain_queries: list[str] = Field(default_factory=list)
    method_queries: list[str] = Field(default_factory=list)
    dataset_queries: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)
    rationale: str = ""


class LiteratureBundle(BaseModel):
    bundle_id: str
    purpose: Literal["domain", "method"]
    queries: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    gap_ids: list[str] = Field(default_factory=list)
    summary: str = ""


class DomainGap(BaseModel):
    gap_id: str
    claim: str
    source_ids: list[str] = Field(default_factory=list)
    why_it_matters: str = ""
    testable_question: str = ""
    dataset_requirements: list[str] = Field(default_factory=list)
    suggested_metrics: list[str] = Field(default_factory=list)


class HypothesisCandidate(BaseModel):
    hypothesis_id: str
    claim_candidate: str
    motivation: str
    literature_gap_ids: list[str] = Field(default_factory=list)
    dataset_requirements: list[str] = Field(default_factory=list)
    expected_signal: str = ""
    falsification_condition: str
    minimum_evidence_needed: str
    scientific_value: float = 0.5
    compute_risk: float = 0.5
    priority_score: float = 0.5


class ResearchAgenda(BaseModel):
    question: str
    domain_summary: str = ""
    method_summary: str = ""
    domain_gaps: list[DomainGap] = Field(default_factory=list)
    hypotheses: list[HypothesisCandidate] = Field(default_factory=list)
    selected_hypothesis_id: str = ""
    selection_rationale: str = ""


class DatasetSearchPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)
    required_properties: list[str] = Field(default_factory=list)
    preferred_sources: list[str] = Field(default_factory=lambda: ["huggingface", "openml"])
    rejection_criteria: list[str] = Field(default_factory=list)
    rationale: str = ""


class DatasetSelectionRationale(BaseModel):
    selected_dataset_id: str
    selected_source: str
    selected_reason: str
    rejected_summary: list[str] = Field(default_factory=list)
    fit_to_hypothesis: str = ""
    risks: list[str] = Field(default_factory=list)


class VerificationPlan(BaseModel):
    claim_enabled_if: str
    claim_blocked_if: str
    statistical_tests: list[str] = Field(default_factory=list)
    support_threshold: str = ""
    weak_support_threshold: str = ""
    invalid_protocol_conditions: list[str] = Field(default_factory=list)
    allowed_rewrites: list[str] = Field(default_factory=list)


class NextDecision(BaseModel):
    decision: Literal[
        "replicate",
        "ablate",
        "inspect_failure",
        "revise_hypothesis",
        "revise_protocol",
        "search_better_dataset",
        "verify_claim",
        "stop_and_report",
    ]
    rationale: str
    next_action_goal: str = ""
    expected_evidence_gain: str = ""
    stop_reason: str = ""


class CodeValidationResult(BaseModel):
    status: Literal["accepted", "rejected"]
    reason: str = ""
    blocked_nodes: list[str] = Field(default_factory=list)
    blocked_imports: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LabAction(BaseModel):
    action_id: str
    kind: Literal[
        "search_datasets",
        "audit_dataset",
        "draft_protocol",
        "generate_code",
        "dry_run",
        "inspect_dataset",
        "run_baseline",
        "run_protocol",
        "run_replication",
        "run_ablation",
        "analyze_results",
        "verify_claims",
        "stop_and_report",
    ]
    hypothesis_id: str | None = None
    protocol_id: str | None = None
    scientific_goal: str
    command: str
    expected_artifacts: list[str] = Field(default_factory=list)
    primary_metric: str = "balanced_accuracy"
    manipulated_variables: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    protocol_risks: list[str] = Field(default_factory=list)
    claim_delta_target: Literal[
        "none",
        "adds_observation",
        "reduces_uncertainty",
        "enables_claim",
        "blocks_or_rewrites_claim",
        "report_ready",
    ] = "adds_observation"
    estimated_cost_class: Literal["free", "cheap", "moderate", "expensive"] = "cheap"


class LabScientistDecision(BaseModel):
    source: Literal["llm", "handoff", "deterministic"] = "deterministic"
    research_question: str
    working_hypothesis: str
    candidate_action_ids: list[str] = Field(default_factory=list)
    preferred_action_id: str
    rationale: str
    expected_evidence_needed: str
    claim_risk: str
    stop_or_continue: Literal["continue", "stop_and_report"] = "continue"
    prompt_version: str | None = None
    model: str | None = None


class LabPrediction(BaseModel):
    source: Literal["qwen_agentworld", "plumbing_not_called"] = "plumbing_not_called"
    predicted_terminal_observation: str = ""
    expected_result_pattern: str = ""
    predicted_artifacts: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    protocol_risks: list[str] = Field(default_factory=list)
    runtime_risk: str = ""
    expected_runtime_seconds: float | None = None
    expected_runtime_range_seconds: list[float] = Field(default_factory=list)
    compute_waste_risk: float = 0.0
    value_of_information: float = 0.5
    suggested_modification: str = ""
    decision_threshold: str = ""
    claim_support_probability: float = 0.5
    expected_best_model: str = ""
    preferred_next_action_if_blocked: str = ""
    what_would_change_my_mind: str = ""
    discriminative_reason: str = ""
    expected_claim_delta: Literal[
        "none",
        "adds_observation",
        "reduces_uncertainty",
        "enables_claim",
        "blocks_or_rewrites_claim",
        "report_ready",
    ] = "adds_observation"
    recommendation: Literal["run", "skip", "modify", "verify", "stop_and_report"] = "run"
    why_not_score_chasing: str = ""
    rationale: str = ""
    prompt_version: str | None = None
    world_model_schema_version: str | None = None


class LabObservation(BaseModel):
    status: str
    action_id: str
    protocol_id: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""
    artifacts: list[str] = Field(default_factory=list)
    runtime_seconds: float | None = None


class LabAnalysis(BaseModel):
    analysis_id: str
    protocol_id: str
    primary_metric: str = "balanced_accuracy"
    condition_means: dict[str, float] = Field(default_factory=dict)
    condition_stds: dict[str, float] = Field(default_factory=dict)
    effect_size: float | None = None
    seed_noise: float | None = None
    effect_to_noise_ratio: float | None = None
    best_condition: str | None = None
    protocol_warnings: list[str] = Field(default_factory=list)
    summary: str = ""


class LabClaim(BaseModel):
    claim_id: str
    hypothesis_id: str | None = None
    claim: str
    verdict: Literal["supported", "weakly_supported", "inconclusive", "blocked", "observation_only"]
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    failure_category: str | None = None
    diagnostic: str | None = None
    next_action: str | None = None
    allowed_rewrite: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    # Provenance from the EXECUTED protocol, machine-derived. The `claim` prose
    # comes from the hypothesis and may drift from what actually ran; these
    # fields always reflect reality (the flaw published in paper §5.1).
    executed_dataset: str | None = None
    executed_conditions: list[str] = Field(default_factory=list)
    executed_metric: str | None = None
    provenance_note: str | None = None


class LabNotebookEntry(BaseModel):
    step: int
    hypothesis_id: str | None = None
    state_before: str
    candidate_actions: list[LabAction] = Field(default_factory=list)
    scientist_decision: LabScientistDecision | None = None
    qwen_predictions: list[dict[str, Any]] = Field(default_factory=list)
    selected_action: LabAction
    why_world_model_mattered: str = ""
    actual_observation: LabObservation
    prediction_comparison: dict[str, Any] = Field(default_factory=dict)
    analysis: LabAnalysis | None = None
    claim_updates: list[LabClaim] = Field(default_factory=list)
    next_decision: str = ""


class LabStudyState(BaseModel):
    lab_question: LabQuestion
    hypotheses: list[ResearchHypothesis] = Field(default_factory=list)
    protocols: list[ProtocolSpec] = Field(default_factory=list)
    completed_actions: list[str] = Field(default_factory=list)
    observations: list[LabObservation] = Field(default_factory=list)
    analyses: list[LabAnalysis] = Field(default_factory=list)
    claims: list[LabClaim] = Field(default_factory=list)
    summary: str = ""


class LabStudyResult(BaseModel):
    workspace: str
    lab_question: LabQuestion
    hypotheses: list[ResearchHypothesis]
    protocols: list[ProtocolSpec]
    claims: list[LabClaim]
    final_report: str


class CalibrationMetrics(BaseModel):
    metric_interval_coverage: float | None = None
    runtime_interval_coverage: float | None = None
    metric_absolute_error: float | None = None
    runtime_relative_error: float | None = None
    prediction_miss_count: int = 0
    risk_recall: float | None = None
    recommendation_quality: float | None = None
    useful_decision_count: int = 0
    high_claim_impact_verification_count: int = 0
    skip_or_stop_recommendation_count: int = 0
    memory_augmented_prediction_count: int = 0
    few_shot_augmented_prediction_count: int = 0


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
    planner_mode: str | None = None
    agent_backend: str | None = None
    agent_model: str | None = None
    agent_prompt_hash: str | None = None
    agent_decision: AgentDecision | None = None
    safety_validation: SafetyValidation | None = None
    research_hypothesis: str | None = None
    state_before: ResearchState | None = None
    candidate_actions: list[ProposedAction] = Field(default_factory=list)
    candidate_predictions: list[CandidatePrediction] = Field(default_factory=list)
    selected_action: ProposedAction | None = None
    decision_trace: DecisionTrace | None = None
    claim_ledger_updates: list[ClaimLedgerEntry] = Field(default_factory=list)
    calibration_metrics: CalibrationMetrics | None = None
    top_model_summary: TopModelSummary | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    memory_examples_used: list[dict[str, Any]] = Field(default_factory=list)
    prompt_version: str | None = None
    world_model_schema_version: str | None = None
