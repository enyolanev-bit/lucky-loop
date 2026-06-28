from __future__ import annotations

import re
import shlex
from pathlib import Path

from .schemas import LabAction, LabQuestion, LabStudyState, ProtocolSpec, ResearchHypothesis


SUPPORTED_STUDIES = {
    "split_validity_sensor",
    "leakage_trap",
    "metric_misuse_imbalance",
    "seed_variance_claim",
    "small_data_complexity",
}


def slugify(text: str, max_len: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].strip("-") or "lab-study")


def infer_study_id(question: str, requested: str | None = None) -> str:
    if requested:
        if requested not in SUPPORTED_STUDIES:
            raise ValueError(f"unsupported study: {requested}")
        return requested
    text = question.lower()
    if any(word in text for word in ["split", "sequential", "sensor", "time", "group"]):
        return "split_validity_sensor"
    if any(word in text for word in ["leak", "preprocessing", "pipeline"]):
        return "leakage_trap"
    if any(word in text for word in ["imbalance", "accuracy", "minority", "balanced"]):
        return "metric_misuse_imbalance"
    if any(word in text for word in ["seed", "robust", "winner", "variance"]):
        return "seed_variance_claim"
    if any(word in text for word in ["complex", "small data", "sample size", "overfit"]):
        return "small_data_complexity"
    return "split_validity_sensor"


def make_lab_question(question: str, study_id: str | None, budget: int, require_qwen: bool) -> LabQuestion:
    resolved = infer_study_id(question, study_id)
    return LabQuestion(
        question=question,
        study_id=resolved,
        budget=budget,
        require_qwen=require_qwen,
        success_criteria=[
            "literature review, gaps, hypotheses, protocols, real runs, analyses, notebook, claim ledger, and final report exist",
            "Qwen-AgentWorld predicts each Lucky Loop lab action before execution when --require-qwen is used",
            "claims in the final report are backed by deterministic lab verifier entries",
        ],
    )


def hypotheses_for(study_id: str) -> list[ResearchHypothesis]:
    if study_id == "split_validity_sensor":
        return [
            ResearchHypothesis(
                hypothesis_id="H1_split_overstates_generalization",
                claim_candidate="Random train/test splits overstate generalization on sequential sensor data compared with blocked splits.",
                why_it_matters="Sensor samples can be temporally or subject correlated; random splits can mix near-duplicates across train and test.",
                literature_gap_ids=["gap_split_validity_sensor"],
                falsification_condition="Random and blocked split performance are similar within seed noise.",
                minimum_evidence_needed="Repeated-seed comparison where random split exceeds blocked split by more than seed noise.",
            )
        ]
    if study_id == "leakage_trap":
        return [
            ResearchHypothesis(
                hypothesis_id="H1_leakage_invalidates_claim",
                claim_candidate="A high score from leaky preprocessing is protocol-invalid and must not be reported as model generalization.",
                why_it_matters="Autonomous ML agents can accidentally run preprocessing outside the train fold and produce impressive but invalid results.",
                literature_gap_ids=["gap_leakage_trap"],
                falsification_condition="Proper and intentionally leaky preprocessing produce comparable performance.",
                minimum_evidence_needed="Controlled proper-vs-leaky protocol with a protocol warning attached to the leaky branch.",
            )
        ]
    if study_id == "metric_misuse_imbalance":
        return [
            ResearchHypothesis(
                hypothesis_id="H1_accuracy_misleads_under_imbalance",
                claim_candidate="Accuracy alone is misleading on imbalanced classification when balanced accuracy or F1 reveals weaker minority behavior.",
                why_it_matters="Score-chasing agents can optimize the wrong metric and write a false success claim.",
                literature_gap_ids=["gap_metric_misuse_imbalance"],
                falsification_condition="Accuracy, balanced accuracy, and F1 agree across conditions.",
                minimum_evidence_needed="Repeated runs on an imbalanced protocol showing material divergence between accuracy and balanced accuracy/F1.",
            )
        ]
    if study_id == "seed_variance_claim":
        return [
            ResearchHypothesis(
                hypothesis_id="H1_single_run_winner_not_robust",
                claim_candidate="A single-run winner is not a robust best-model claim when the effect is smaller than seed noise.",
                why_it_matters="Auto-research reports often turn leaderboard observations into unsupported claims.",
                literature_gap_ids=["gap_seed_variance_claim"],
                falsification_condition="Repeated seeds show an effect-to-noise ratio above the support threshold.",
                minimum_evidence_needed="Multi-seed comparison of candidate models with effect/noise reported.",
            )
        ]
    return [
        ResearchHypothesis(
            hypothesis_id="H1_complexity_not_claimably_better",
            claim_candidate="Complex models do not produce claimable improvement over simple baselines under limited data unless gains exceed seed noise.",
            why_it_matters="Compute-heavy models can look attractive while adding little claimable evidence.",
            literature_gap_ids=["gap_small_data_complexity"],
            falsification_condition="Complex models beat simple baselines by more than seed noise across sample sizes.",
            minimum_evidence_needed="Repeated sample-size ablation comparing simple and complex model families.",
        )
    ]


def protocols_for(study_id: str) -> list[ProtocolSpec]:
    if study_id == "split_validity_sensor":
        return [
            ProtocolSpec(
                protocol_id="random_vs_blocked_split",
                hypothesis_id="H1_split_overstates_generalization",
                scientific_goal="Compare random train/test splitting against blocked sequential splitting on real sensor-style data.",
                dataset="eeg_eye_state",
                conditions=["random_split", "blocked_split"],
                controlled_variables=["model_family", "feature_set", "test_fraction", "seeds"],
                manipulated_variable="split_strategy",
                protocol_risks=["temporal_correlation", "single_split_overclaim"],
                claim_enabled_if="random_split mean exceeds blocked_split mean by more than seed noise",
                claim_blocked_if="split effects are within seed noise",
                expected_artifacts=["runs/experiment_<step>.json", "analyses/analysis_<step>.json"],
            )
        ]
    if study_id == "leakage_trap":
        return [
            ProtocolSpec(
                protocol_id="proper_vs_leaky_preprocessing",
                hypothesis_id="H1_leakage_invalidates_claim",
                scientific_goal="Compare a proper train-fold preprocessing pipeline against an intentionally leaky preprocessing probe.",
                dataset="breast_cancer",
                conditions=["proper_pipeline", "leaky_preprocessing"],
                controlled_variables=["model_family", "seeds", "test_fraction"],
                manipulated_variable="preprocessing_scope",
                protocol_risks=["data_leakage"],
                claim_enabled_if="leaky preprocessing raises the score but emits a protocol warning",
                claim_blocked_if="no score inflation or warning is observed",
                expected_artifacts=["runs/experiment_<step>.json", "analyses/analysis_<step>.json"],
            )
        ]
    if study_id == "metric_misuse_imbalance":
        return [
            ProtocolSpec(
                protocol_id="accuracy_vs_balanced_metrics",
                hypothesis_id="H1_accuracy_misleads_under_imbalance",
                scientific_goal="Create a controlled imbalance protocol and compare accuracy with balanced accuracy and F1.",
                dataset="breast_cancer",
                conditions=["accuracy_objective", "balanced_metrics_audit"],
                controlled_variables=["model_family", "seeds", "imbalance_fraction"],
                manipulated_variable="reported_metric",
                protocol_risks=["metric_misuse", "class_imbalance"],
                claim_enabled_if="accuracy remains high while balanced accuracy or F1 materially drops",
                claim_blocked_if="all metrics agree",
                expected_artifacts=["runs/experiment_<step>.json", "analyses/analysis_<step>.json"],
            )
        ]
    if study_id == "seed_variance_claim":
        return [
            ProtocolSpec(
                protocol_id="single_run_vs_repeated_seeds",
                hypothesis_id="H1_single_run_winner_not_robust",
                scientific_goal="Compare single-run model winner against repeated-seed evidence.",
                dataset="wine",
                conditions=["single_run_winner", "repeated_seed_comparison"],
                controlled_variables=["dataset", "candidate_models", "preprocessing"],
                manipulated_variable="seed_replication",
                protocol_risks=["seed_variance", "single_split_overclaim"],
                claim_enabled_if="best model effect exceeds seed noise",
                claim_blocked_if="effect is less than or equal to seed noise",
                expected_artifacts=["runs/experiment_<step>.json", "analyses/analysis_<step>.json"],
            )
        ]
    return [
        ProtocolSpec(
            protocol_id="simple_vs_complex_small_data",
            hypothesis_id="H1_complexity_not_claimably_better",
            scientific_goal="Compare simple and complex model families under limited sample sizes.",
            dataset="wine",
            conditions=["simple_baseline", "complex_model"],
            controlled_variables=["seeds", "sample_fraction", "metric"],
            manipulated_variable="model_complexity",
            protocol_risks=["overfitting", "small_sample_variance"],
            claim_enabled_if="complex model gain exceeds seed noise",
            claim_blocked_if="complex model gain is within seed noise",
            expected_artifacts=["runs/experiment_<step>.json", "analyses/analysis_<step>.json"],
        )
    ]


def build_action(protocol: ProtocolSpec, step: int, workspace: Path) -> LabAction:
    return build_protocol_action(protocol, "run_main_protocol", step, workspace, "run_protocol")


def build_protocol_action(
    protocol: ProtocolSpec,
    lab_action: str,
    step: int,
    workspace: Path,
    kind: str = "run_protocol",
) -> LabAction:
    seeds = " ".join(str(seed) for seed in protocol.seeds)
    command = " ".join(
        [
            "python",
            "experiments/ml_validity_lab.py",
            "--action",
            shlex.quote(lab_action),
            "--study",
            shlex.quote(protocol.protocol_id),
            "--dataset",
            shlex.quote(protocol.dataset),
            "--seeds",
            seeds,
            "--out-dir",
            shlex.quote(str(workspace)),
            "--step",
            str(step),
        ]
    )
    kind_map = {
        "inspect_dataset": "inspect_dataset",
        "run_random_split_baseline": "run_baseline",
        "run_baseline": "run_baseline",
        "run_main_protocol": "run_protocol",
        "run_replication": "run_replication",
        "run_negative_control": "run_ablation",
        "run_ablation": "run_ablation",
    }
    action_kind = kind_map.get(lab_action, kind)
    claim_delta = "adds_observation"
    if action_kind in {"run_protocol", "run_replication", "run_ablation"}:
        claim_delta = "reduces_uncertainty"
    return LabAction(
        action_id=f"action_{step:03d}_{lab_action}_{protocol.protocol_id}",
        kind=action_kind,
        hypothesis_id=protocol.hypothesis_id,
        protocol_id=protocol.protocol_id,
        scientific_goal=protocol.scientific_goal,
        command=command,
        expected_artifacts=protocol.expected_artifacts,
        primary_metric=protocol.primary_metric,
        manipulated_variables=[protocol.manipulated_variable],
        controls=protocol.controlled_variables,
        protocol_risks=protocol.protocol_risks,
        claim_delta_target=claim_delta,
        estimated_cost_class="moderate",
    )


def candidate_actions_for_state(state: LabStudyState, step: int, workspace: Path) -> list[LabAction]:
    completed = set(state.completed_actions)
    actions: list[LabAction] = []
    for protocol in state.protocols:
        if f"inspect_dataset:{protocol.dataset}" not in completed:
            actions.append(build_protocol_action(protocol, "inspect_dataset", step, workspace, "inspect_dataset"))
        if f"baseline:{protocol.protocol_id}" not in completed:
            baseline_action = "run_random_split_baseline" if protocol.protocol_id == "random_vs_blocked_split" else "run_baseline"
            actions.append(build_protocol_action(protocol, baseline_action, step, workspace, "run_baseline"))
        if f"main:{protocol.protocol_id}" not in completed:
            actions.append(build_protocol_action(protocol, "run_main_protocol", step, workspace, "run_protocol"))
        if f"replication:{protocol.protocol_id}" not in completed and f"main:{protocol.protocol_id}" in completed:
            actions.append(build_protocol_action(protocol, "run_replication", step, workspace, "run_replication"))
        if f"control:{protocol.protocol_id}" not in completed and protocol.protocol_id in {
            "proper_vs_leaky_preprocessing",
            "random_vs_blocked_split",
            "accuracy_vs_balanced_metrics",
        }:
            actions.append(build_protocol_action(protocol, "run_negative_control", step, workspace, "run_ablation"))
    if not actions:
        actions.append(stop_action(step))
    return actions[:8]


def stop_action(step: int) -> LabAction:
    return LabAction(
        action_id=f"action_{step:03d}_stop_and_report",
        kind="stop_and_report",
        scientific_goal="Stop running experiments and write the evidence-bounded report.",
        command="python -c \"print('stop_and_report: no further lab compute requested')\"",
        claim_delta_target="report_ready",
        estimated_cost_class="free",
    )
