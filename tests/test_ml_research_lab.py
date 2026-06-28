from __future__ import annotations

import json
import subprocess
import sys

from luckyloop.lab import run_lab
from luckyloop.lab_protocols import build_action, candidate_actions_for_state, hypotheses_for, make_lab_question, protocols_for
from luckyloop.lab_scientist import decide_next_action
from luckyloop.lab_verifier import analyze_observation, verify_lab_claims
from luckyloop.lab_world_model import _prediction_quality_failures, normalize_prediction, predict_lab_action
from luckyloop.open_lab import _make_dry_run_dataset, _script_action
from luckyloop.code_safety import validate_generated_code, write_validated_code
from luckyloop.dataset_discovery import audit_candidate
from luckyloop.dataset_discovery import _infer_target
from luckyloop.dataset_discovery import _contains_term
from luckyloop.dataset_discovery import _openml_query_variants
from luckyloop.dataset_discovery import discover_dataset_candidates, select_and_materialize_dataset
from luckyloop.lab_scientist import template_experiment_code
from luckyloop.lab_scientist import _agenda_from_data
from luckyloop.literature import derive_literature_query_plan, write_split_context, ResearchContext
from luckyloop.open_lab import _dataset_search_plan
from luckyloop.lab_reporter import write_final_report
from luckyloop.schemas import (
    DatasetAudit,
    DatasetCandidate,
    DatasetSearchPlan,
    GeneratedResearchProtocol,
    HypothesisCandidate,
    LabQuestion,
    LabStudyState,
    ProtocolSpec,
    ResearchHypothesis,
)
from luckyloop.tasks import ROOT


def test_protocol_compiler_uses_safe_ml_validity_runner(tmp_path):
    protocol = protocols_for("split_validity_sensor")[0]
    action = build_action(protocol, 1, tmp_path)
    assert action.command.startswith("python experiments/ml_validity_lab.py")
    assert "--study random_vs_blocked_split" in action.command
    assert "&&" not in action.command
    assert ";" not in action.command


def test_lab_exposes_multiple_candidate_actions(tmp_path):
    lab_question = make_lab_question("Do random splits overstate sensor performance?", "split_validity_sensor", 4, False)
    protocols = protocols_for(lab_question.study_id)
    hypotheses = hypotheses_for(lab_question.study_id)
    state = LabStudyState(lab_question=lab_question, protocols=protocols, hypotheses=hypotheses)
    actions = candidate_actions_for_state(state, 1, tmp_path)
    assert len(actions) >= 3
    assert {action.kind for action in actions} >= {"inspect_dataset", "run_baseline", "run_protocol"}


def test_real_runner_returns_standard_json(tmp_path):
    cmd = [
        sys.executable,
        "experiments/ml_validity_lab.py",
        "--study",
        "single_run_vs_repeated_seeds",
        "--dataset",
        "wine",
        "--seeds",
        "0",
        "1",
        "--out-dir",
        str(tmp_path),
        "--step",
        "1",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.find("{") :])
    assert payload["status"] == "success"
    assert payload["runs"]
    assert payload["primary_metric"] == "balanced_accuracy"
    assert (tmp_path / "runs" / "experiment_001.json").exists()


def test_lab_verifier_blocks_leakage_claim():
    protocol = protocols_for("leakage_trap")[0]
    hypothesis = hypotheses_for("leakage_trap")[0]
    raw = {
        "protocol_id": "proper_vs_leaky_preprocessing",
        "primary_metric": "balanced_accuracy",
        "runs": [
            {"condition": "proper_pipeline", "balanced_accuracy": 0.7},
            {"condition": "leaky_preprocessing", "balanced_accuracy": 1.0},
        ],
        "effect_size": 0.3,
        "seed_noise": 0.01,
        "effect_to_noise_ratio": 30,
        "protocol_warnings": ["data_leakage: label-derived feature was intentionally included"],
    }
    analysis = analyze_observation(raw, "analysis_001")
    claims = verify_lab_claims(protocol, analysis, hypothesis, "experiment_001")
    assert any(claim.verdict == "blocked" and "leaky" in claim.claim.lower() for claim in claims)


def test_require_qwen_fails_without_simulator_env(monkeypatch, tmp_path):
    monkeypatch.delenv("LUCKYWORLD_SIMULATOR_BASE_URL", raising=False)
    monkeypatch.delenv("LUCKYWORLD_SIMULATOR_MODEL", raising=False)
    lab_question = make_lab_question("Do random splits overstate sensor performance?", "split_validity_sensor", 4, True)
    protocols = protocols_for(lab_question.study_id)
    hypotheses = hypotheses_for(lab_question.study_id)
    state = LabStudyState(lab_question=lab_question, protocols=protocols, hypotheses=hypotheses)
    action = build_action(protocols[0], 1, tmp_path)
    try:
        predict_lab_action(action, state, require_qwen=True)
    except RuntimeError as exc:
        assert "Qwen-AgentWorld is required" in str(exc)
    else:
        raise AssertionError("require_qwen should fail when simulator env is missing")


def test_require_agent_fails_without_agent_env(monkeypatch, tmp_path):
    monkeypatch.delenv("LUCKYLOOP_AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("LUCKYLOOP_AGENT_MODEL", raising=False)
    monkeypatch.delenv("LUCKYLOOP_AGENT_API_KEY", raising=False)
    lab_question = make_lab_question("Do random splits overstate sensor performance?", "split_validity_sensor", 4, False)
    protocols = protocols_for(lab_question.study_id)
    hypotheses = hypotheses_for(lab_question.study_id)
    state = LabStudyState(lab_question=lab_question, protocols=protocols, hypotheses=hypotheses)
    actions = candidate_actions_for_state(state, 1, tmp_path)
    try:
        decide_next_action(state, actions, {}, planner="llm", require_agent=True)
    except RuntimeError as exc:
        assert "Scientist planner is required" in str(exc)
    else:
        raise AssertionError("require_agent should fail when planner env is missing")


def test_require_agent_requires_llm_planner():
    try:
        run_lab("Can leaky preprocessing create invalid ML claims?", "leakage_trap", 1, False, "lucky_loop_lab", "deterministic", True)
    except RuntimeError as exc:
        assert "--require-agent requires --planner llm" in str(exc)
    else:
        raise AssertionError("require_agent should require planner llm")


def test_dataset_audit_accepts_sklearn_fallback(tmp_path):
    candidate = DatasetCandidate(
        dataset_id="wine",
        source="sklearn",
        name="wine",
        description="sklearn fallback",
        license="bsd-3-clause",
    )
    audit = audit_candidate(candidate, tmp_path)
    assert audit.status == "accepted"
    assert audit.local_path
    assert audit.target_column == "target"
    assert audit.n_rows > 80
    assert (tmp_path / "datasets" / "selected_dataset.csv").exists()


def test_dataset_target_inference_prefers_semantic_target_over_binary_feature():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "Sex": ["male", "female", "male", "female"],
            "Age": [20, 30, 40, 50],
            "Survived": [0, 1, 0, 1],
        }
    )
    assert _infer_target(frame) == "Survived"


def test_dataset_domain_term_matching_uses_word_boundaries_for_short_terms():
    assert _contains_term("human activity recognition har benchmark", "har")
    assert not _contains_term("monthly charges customer churn", "har")


def test_openml_query_variants_add_domain_specific_short_names():
    variants = _openml_query_variants("human activity recognition with wearable sensor features")
    assert "har" in variants
    assert "eeg-eye-state" in variants
    assert "sonar" in variants


def test_dataset_selection_prefers_dynamic_candidate_over_curated_fallback(monkeypatch, tmp_path):
    dynamic = DatasetCandidate(dataset_id="dynamic_hf", source="huggingface", name="dynamic_hf", score=1)
    fallback = DatasetCandidate(
        dataset_id="fallback_openml",
        source="openml",
        name="fallback_openml",
        score=99,
        risk_flags=["curated_fallback"],
    )

    def fake_audit(candidate, workspace, max_rows=50000):
        dataset = workspace / "datasets" / f"{candidate.dataset_id}.csv"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_text("x,y,target\n1,2,a\n2,3,b\n3,4,a\n4,5,b\n", encoding="utf-8")
        return DatasetAudit(
            dataset_id=candidate.dataset_id,
            source=candidate.source,
            status="accepted",
            reason="usable supervised classification table",
            local_path=str(dataset),
            target_column="target",
            feature_columns=["x", "y"],
            n_rows=4,
            n_features=2,
            class_counts={"a": 2, "b": 2},
        )

    monkeypatch.setattr("luckyloop.dataset_discovery.audit_candidate", fake_audit)
    selected, _audit = select_and_materialize_dataset([fallback, dynamic], tmp_path)
    assert selected.dataset_id == "dynamic_hf"
    rationale = json.loads((tmp_path / "datasets" / "selection_rationale.json").read_text())
    assert rationale["selected_dataset_id"] == "dynamic_hf"


def test_dataset_selection_rejects_dynamic_domain_mismatch_when_sensor_required(monkeypatch, tmp_path):
    titanic = DatasetCandidate(dataset_id="titanic", source="huggingface", name="titanic", description="passenger survival", score=99)
    har = DatasetCandidate(
        dataset_id="har",
        source="openml",
        name="har",
        description="human activity recognition sensor benchmark",
        score=1,
        risk_flags=["curated_fallback"],
    )

    def fake_audit(candidate, workspace, max_rows=50000):
        dataset = workspace / "datasets" / f"{candidate.dataset_id}.csv"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_text("x,y,target\n1,2,a\n2,3,b\n3,4,a\n4,5,b\n", encoding="utf-8")
        return DatasetAudit(
            dataset_id=candidate.dataset_id,
            source=candidate.source,
            status="accepted",
            reason="usable supervised classification table",
            local_path=str(dataset),
            target_column="target",
            feature_columns=["accel_x", "gyro_y"] if candidate.dataset_id == "har" else ["fare", "age"],
            n_rows=4,
            n_features=2,
            class_counts={"a": 2, "b": 2},
        )

    monkeypatch.setattr("luckyloop.dataset_discovery.audit_candidate", fake_audit)
    plan = DatasetSearchPlan(queries=["human activity recognition sensor classification"], required_properties=["sensor features"])
    selected, _audit = select_and_materialize_dataset([titanic, har], tmp_path, search_plan=plan)
    assert selected.dataset_id == "har"


def test_dataset_discovery_logs_search_errors(monkeypatch, tmp_path):
    def fail_hf(query, limit=8, errors=None):
        if errors is not None:
            errors.append({"source": "huggingface", "query": query, "error": "boom"})
        return []

    monkeypatch.setattr("luckyloop.dataset_discovery.search_huggingface", fail_hf)
    monkeypatch.setattr("luckyloop.dataset_discovery.search_openml", lambda query, limit=10, errors=None: [])
    monkeypatch.setattr("luckyloop.dataset_discovery.external_registry_candidates", lambda query_text: [])
    candidates = discover_dataset_candidates("sensor classification", {}, tmp_path)
    assert candidates == []
    errors = json.loads((tmp_path / "datasets" / "search_errors.json").read_text())
    assert errors and errors[0]["source"] == "huggingface"


def test_generated_code_validator_blocks_shell_access():
    result = validate_generated_code("import subprocess\nsubprocess.run(['echo', 'bad'])\n")
    assert result.status == "rejected"
    assert "subprocess" in result.blocked_imports


def test_template_generated_experiment_runs_on_materialized_dataset(tmp_path):
    candidate = DatasetCandidate(dataset_id="wine", source="sklearn", name="wine")
    audit = audit_candidate(candidate, tmp_path)
    protocol = GeneratedResearchProtocol(
        question="Is a complex model robustly better than a linear baseline?",
        hypothesis="Complex models must beat a linear baseline beyond seed noise.",
        dataset_id=audit.dataset_id,
        dataset_source=audit.source,
        target_column=audit.target_column or "target",
        feature_columns=audit.feature_columns,
        candidate_models=["logistic_regression", "random_forest"],
        seeds=[0, 1],
    )
    code = template_experiment_code().replace("__PROTOCOL_JSON__", json.dumps(protocol.model_dump(), sort_keys=True))
    script = tmp_path / "generated" / "experiment.py"
    validation = write_validated_code(code, script)
    assert validation.status == "accepted", validation.model_dump()
    cmd = [
        sys.executable,
        str(script),
        "--dataset-csv",
        audit.local_path,
        "--target-column",
        audit.target_column,
        "--out-dir",
        str(tmp_path),
        "--step",
        "1",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.find("{") :])
    assert payload["status"] == "success"
    assert payload["runs"]
    assert (tmp_path / "runs" / "experiment_001.json").exists()


def test_template_generated_experiment_accepts_deepseek_model_aliases(tmp_path):
    candidate = DatasetCandidate(dataset_id="wine", source="sklearn", name="wine")
    audit = audit_candidate(candidate, tmp_path)
    protocol = GeneratedResearchProtocol(
        question="Can nonlinear sklearn models beat logistic regression?",
        hypothesis="Nonlinear aliases must map to deterministic sklearn classes.",
        dataset_id=audit.dataset_id,
        dataset_source=audit.source,
        target_column=audit.target_column or "target",
        feature_columns=audit.feature_columns,
        candidate_models=["logisticregression", "gradientboostingclassifier", "mlpclassifier", "randomforestclassifier", "svc"],
        seeds=[0],
    )
    code = template_experiment_code().replace("__PROTOCOL_JSON__", json.dumps(protocol.model_dump(), sort_keys=True))
    script = tmp_path / "generated" / "experiment.py"
    validation = write_validated_code(code, script)
    assert validation.status == "accepted", validation.model_dump()
    cmd = [
        sys.executable,
        str(script),
        "--dataset-csv",
        audit.local_path,
        "--target-column",
        audit.target_column,
        "--out-dir",
        str(tmp_path),
        "--step",
        "2",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.find("{") :])
    assert {row["model"] for row in payload["runs"]} >= {"logisticregression", "gradientboostingclassifier", "mlpclassifier"}
    assert (tmp_path / "runs" / "experiment_002.json").exists()


def test_open_lab_dry_run_uses_reduced_dataset(tmp_path):
    dataset = tmp_path / "selected_dataset.csv"
    dataset.write_text("x,y,target\n" + "\n".join(f"{i},{i + 1},{i % 2}" for i in range(1000)) + "\n", encoding="utf-8")
    audit = DatasetAudit(
        dataset_id="external",
        source="openml",
        status="accepted",
        reason="test",
        local_path=str(dataset),
        target_column="target",
        feature_columns=["x", "y"],
        n_rows=1000,
        n_features=2,
        class_counts={"0": 500, "1": 500},
    )

    reduced = _make_dry_run_dataset(audit, tmp_path, max_rows=50)
    assert sum(1 for _ in open(reduced, encoding="utf-8")) == 51

    action = _script_action(5, "dry_run", ROOT / "reports" / "test_generated" / "experiment.py", audit, tmp_path, dry_run=True, dataset_csv=reduced)
    assert "--dry-run" in action.command
    assert str(reduced) in action.command
    assert action.estimated_cost_class == "cheap"


def test_lab_world_model_prediction_keeps_compute_fields():
    prediction = normalize_prediction(
        {
            "predicted_terminal_observation": "short schema check succeeds",
            "expected_result_pattern": "writes experiment json",
            "predicted_artifacts": ["runs/experiment_005.json"],
            "failure_modes": ["dry-run accidentally trains full SVC"],
            "protocol_risks": ["heavy_model_runtime"],
            "runtime_risk": "low if dataset is reduced, high otherwise",
            "expected_runtime_seconds": 12,
            "expected_runtime_range_seconds": [5, 30],
            "compute_waste_risk": 0.7,
            "value_of_information": 0.9,
            "suggested_modification": "cap dry-run to 300 rows and one seed",
            "decision_threshold": "run only if schema artifacts are produced",
            "expected_claim_delta": "reduces_uncertainty",
            "recommendation": "modify",
            "why_not_score_chasing": "prevents wasting full compute on invalid generated code",
            "rationale": "dry-run should be a smoke test",
        },
        "qwen_agentworld",
    )
    assert prediction.expected_runtime_seconds == 12
    assert prediction.expected_runtime_range_seconds == [5.0, 30.0]
    assert prediction.compute_waste_risk == 0.7
    assert prediction.value_of_information == 0.9
    assert prediction.recommendation == "modify"
    assert prediction.claim_support_probability == 0.5


def test_lab_world_model_prediction_keeps_discriminative_fields():
    prediction = normalize_prediction(
        {
            "predicted_terminal_observation": "full run may finish but produce a fragile winner",
            "compute_waste_risk": 0.4,
            "value_of_information": 0.8,
            "claim_support_probability": 0.35,
            "expected_best_model": "randomforestclassifier",
            "preferred_next_action_if_blocked": "search_better_dataset",
            "what_would_change_my_mind": "effect-to-noise above 2",
            "discriminative_reason": "full repeated-seed fit can settle the claim; stop/report cannot",
            "recommendation": "verify",
        },
        "qwen_agentworld",
    )
    assert prediction.claim_support_probability == 0.35
    assert prediction.expected_best_model == "randomforestclassifier"
    assert prediction.preferred_next_action_if_blocked == "search_better_dataset"
    assert "settle the claim" in prediction.discriminative_reason


def test_lab_world_model_quality_gate_rejects_neutral_claim_probability(tmp_path):
    action = build_action(protocols_for("seed_variance_claim")[0], 7, tmp_path)
    prediction = normalize_prediction(
        {
            "predicted_terminal_observation": "full protocol probably runs",
            "failure_modes": ["claim may remain unsupported"],
            "runtime_risk": "moderate",
            "claim_support_probability": 0.5,
            "preferred_next_action_if_blocked": "revise_hypothesis",
            "what_would_change_my_mind": "effect-to-noise above 2",
            "discriminative_reason": "full run can test the claim while stop/report cannot",
            "rationale": "the action produces verifier evidence",
        },
        "qwen_agentworld",
    )
    failures = _prediction_quality_failures(action, prediction)
    assert "claim_support_probability must not be neutral 0.5" in failures


def test_literature_query_plan_separates_domain_from_method():
    plan = derive_literature_query_plan(
        "Can a nonlinear model outperform logistic regression on human activity recognition sensor data?"
    )
    assert any("human activity recognition" in query.lower() or "sensor" in query.lower() for query in plan.domain_queries)
    assert any("qwen-agentworld" in query.lower() or "ai scientist" in query.lower() for query in plan.method_queries)
    assert not any("qwen-agentworld" in query.lower() for query in plan.domain_queries)


def test_research_agenda_selects_best_priority_hypothesis():
    agenda = _agenda_from_data(
        "Which model is robust?",
        {
            "domain_summary": "Domain evidence",
            "method_summary": "Method safeguards",
            "domain_gaps": [
                {
                    "gap_id": "g1",
                    "claim": "Need robust model comparison",
                    "source_ids": ["s1"],
                    "why_it_matters": "Avoid trivial claims",
                    "testable_question": "Does nonlinear help?",
                    "dataset_requirements": ["sensor classification"],
                    "suggested_metrics": ["balanced_accuracy"],
                }
            ],
            "hypotheses": [
                {
                    "hypothesis_id": "H_low",
                    "claim_candidate": "Low value claim",
                    "falsification_condition": "fails",
                    "minimum_evidence_needed": "evidence",
                    "scientific_value": 0.2,
                    "compute_risk": 0.8,
                },
                {
                    "hypothesis_id": "H_best",
                    "claim_candidate": "High value claim",
                    "falsification_condition": "fails",
                    "minimum_evidence_needed": "evidence",
                    "scientific_value": 0.9,
                    "compute_risk": 0.3,
                },
            ],
        },
    )
    assert agenda.selected_hypothesis_id == "H_best"
    assert agenda.domain_gaps[0].gap_id == "g1"


def test_research_agenda_does_not_split_string_lists_or_invert_question():
    agenda = _agenda_from_data(
        "Can a nonlinear model robustly outperform logistic regression on public sensor data?",
        {
            "domain_gaps": [
                {
                    "gap_id": "g1",
                    "claim": "Need comparison",
                    "source_ids": "paper_1",
                    "dataset_requirements": "public sensor classification",
                    "suggested_metrics": "balanced_accuracy",
                }
            ],
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "claim_candidate": "Logistic regression achieves comparable or higher balanced accuracy than nonlinear models.",
                    "literature_gap_ids": "g1",
                    "dataset_requirements": "EEG Eye State",
                    "falsification_condition": "fails",
                    "minimum_evidence_needed": "evidence",
                    "scientific_value": 0.8,
                    "compute_risk": 0.4,
                }
            ],
        },
    )
    assert agenda.domain_gaps[0].source_ids == ["paper_1"]
    assert agenda.domain_gaps[0].dataset_requirements == ["public sensor classification"]
    assert agenda.hypotheses[0].literature_gap_ids == ["g1"]
    assert agenda.hypotheses[0].dataset_requirements == ["EEG Eye State"]
    assert agenda.hypotheses[0].claim_candidate.startswith("A nonlinear model can robustly outperform")


def test_dataset_search_plan_uses_hypothesis_requirements():
    plan = _dataset_search_plan(
        ["sensor classification dataset"],
        HypothesisCandidate(
            hypothesis_id="H1",
            claim_candidate="Nonlinear models improve robust sensor classification.",
            motivation="domain literature",
            dataset_requirements=["human activity recognition", "wearable sensor features"],
            falsification_condition="no improvement",
            minimum_evidence_needed="repeated seed effect",
        ),
    )
    joined = " ".join(plan.queries).lower()
    assert "human activity recognition" in joined
    assert "wearable sensor features" in joined
    assert plan.required_properties == ["human activity recognition", "wearable sensor features"]


def test_split_literature_report_links_domain_and_method_sections(tmp_path):
    empty_context = ResearchContext(
        question="Q",
        queries=["Q"],
        papers=[],
        excluded_papers=[],
        gap_findings=[],
        recommended_metrics=[],
        recommended_baselines=[],
        recommended_experiment_plan=[],
    )
    query_plan = derive_literature_query_plan("Q")
    write_split_context(empty_context, empty_context, tmp_path / "literature", query_plan)
    lab_question = LabQuestion(question="Q", study_id="open_generated_ml_research")
    hypothesis = ResearchHypothesis(
        hypothesis_id="H1",
        claim_candidate="Claim",
        why_it_matters="Because",
        falsification_condition="Fails",
        minimum_evidence_needed="Evidence",
    )
    protocol = ProtocolSpec(
        protocol_id="p1",
        hypothesis_id="H1",
        scientific_goal="Goal",
        dataset="d",
        conditions=["a", "b"],
        manipulated_variable="model",
    )
    report = write_final_report(tmp_path, lab_question, [hypothesis], [protocol], [])
    text = report.read_text(encoding="utf-8")
    assert "Domain Related Work" in text
    assert "Method Safeguards" in text
    assert "Lucky Loop Audit Trail" in text


def test_generated_protocol_blocks_negative_effect_claim():
    protocol = protocols_for("seed_variance_claim")[0].model_copy(
        update={"protocol_id": "generated_ml_research_protocol", "hypothesis_id": "H1_generated"}
    )
    hypothesis = hypotheses_for("seed_variance_claim")[0].model_copy(
        update={
            "hypothesis_id": "H1_generated",
            "claim_candidate": "A generated model comparison claim is supported.",
        }
    )
    raw = {
        "protocol_id": "generated_ml_research_protocol",
        "primary_metric": "balanced_accuracy",
        "runs": [
            {"condition": "baseline", "balanced_accuracy": 0.7},
            {"condition": "candidate", "balanced_accuracy": 0.8},
        ],
        "effect_size": -0.1,
        "seed_noise": 0.01,
        "effect_to_noise_ratio": 10.0,
        "protocol_warnings": [],
    }
    analysis = analyze_observation(raw, "analysis_generated")
    claims = verify_lab_claims(protocol, analysis, hypothesis, "experiment_generated")
    assert claims[0].verdict == "blocked"


def test_generated_protocol_blocks_baseline_claim_when_baseline_not_best():
    protocol = protocols_for("seed_variance_claim")[0].model_copy(
        update={"protocol_id": "generated_ml_research_protocol", "hypothesis_id": "H1_generated"}
    )
    hypothesis = hypotheses_for("seed_variance_claim")[0].model_copy(
        update={
            "hypothesis_id": "H1_generated",
            "claim_candidate": "Logistic regression is robustly comparable or better than complex models.",
        }
    )
    raw = {
        "protocol_id": "generated_ml_research_protocol",
        "primary_metric": "balanced_accuracy",
        "runs": [
            {"condition": "logistic_regression", "balanced_accuracy": 0.7},
            {"condition": "random_forest", "balanced_accuracy": 0.9},
        ],
        "effect_size": 0.2,
        "seed_noise": 0.01,
        "effect_to_noise_ratio": 20.0,
        "best_condition": "random_forest",
        "protocol_warnings": [],
    }
    analysis = analyze_observation(raw, "analysis_generated")
    claims = verify_lab_claims(protocol, analysis, hypothesis, "experiment_generated")
    assert claims[0].verdict == "blocked"
    assert any(claim.verdict == "supported" and "should be rejected" in claim.claim for claim in claims)


def test_generated_protocol_does_not_misread_nonlinear_beats_logistic_claim():
    protocol = protocols_for("seed_variance_claim")[0].model_copy(
        update={"protocol_id": "generated_ml_research_protocol", "hypothesis_id": "H1_generated"}
    )
    hypothesis = hypotheses_for("seed_variance_claim")[0].model_copy(
        update={
            "hypothesis_id": "H1_generated",
            "claim_candidate": "A nonlinear model will outperform logistic regression across seeds.",
        }
    )
    raw = {
        "protocol_id": "generated_ml_research_protocol",
        "primary_metric": "balanced_accuracy",
        "runs": [
            {"condition": "logistic_regression", "balanced_accuracy": 0.7},
            {"condition": "random_forest", "balanced_accuracy": 0.72},
        ],
        "effect_size": 0.02,
        "seed_noise": 0.1,
        "effect_to_noise_ratio": 0.2,
        "best_condition": "random_forest",
        "protocol_warnings": [],
    }
    analysis = analyze_observation(raw, "analysis_generated")
    claims = verify_lab_claims(protocol, analysis, hypothesis, "experiment_generated")
    assert claims[0].verdict == "blocked"
    assert claims[0].failure_category == "effect_within_seed_noise"
    assert any(claim.verdict == "supported" and "effect_within_seed_noise" in claim.reason for claim in claims)


def test_generated_protocol_does_not_misread_transformer_beats_logistic_claim():
    protocol = protocols_for("seed_variance_claim")[0].model_copy(
        update={"protocol_id": "generated_ml_research_protocol", "hypothesis_id": "H1_generated"}
    )
    hypothesis = hypotheses_for("seed_variance_claim")[0].model_copy(
        update={
            "hypothesis_id": "H1_generated",
            "claim_candidate": "A Transformer-based model robustly outperforms logistic regression across seeds.",
        }
    )
    raw = {
        "protocol_id": "generated_ml_research_protocol",
        "primary_metric": "balanced_accuracy",
        "runs": [
            {"condition": "logisticregression", "balanced_accuracy": 0.7},
            {"condition": "candidate", "balanced_accuracy": 0.8},
        ],
        "effect_size": 0.1,
        "seed_noise": 0.2,
        "effect_to_noise_ratio": 0.5,
        "best_condition": "candidate",
        "protocol_warnings": [],
    }
    analysis = analyze_observation(raw, "analysis_generated")
    claims = verify_lab_claims(protocol, analysis, hypothesis, "experiment_generated")
    assert claims[0].verdict == "blocked"
    assert claims[0].failure_category == "effect_within_seed_noise"
