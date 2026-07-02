"""Claim-ledger provenance tests (paper §5.1 flaw, fixed forward).

Every LabClaim must carry machine-derived provenance from the EXECUTED
protocol (dataset, conditions, metric), and the verifier must flag claim
prose that names a different dataset than the one that actually ran —
the exact mismatch published as a kept-visible flaw in the M1 workspace
(prose said breast_cancer, evidence ran on eeg_eye_state).

Run standalone (pytest is not installed in .venv):
    PYTHONPATH=src .venv/bin/python tests/test_claim_provenance.py
"""
from __future__ import annotations

from luckyloop.lab_verifier import _dataset_mismatch_note, verify_lab_claims
from luckyloop.schemas import LabAnalysis, ProtocolSpec, ResearchHypothesis


def _protocol(dataset: str = "eeg_eye_state") -> ProtocolSpec:
    return ProtocolSpec(
        protocol_id="generated_ml_research_protocol",
        hypothesis_id="H1",
        scientific_goal="Test claim provenance.",
        dataset=dataset,
        conditions=["logisticregression", "randomforestclassifier"],
        manipulated_variable="model_family",
        primary_metric="balanced_accuracy",
    )


def _analysis(effect: float = 0.12, noise: float = 0.011) -> LabAnalysis:
    return LabAnalysis(
        analysis_id="analysis_001",
        protocol_id="generated_ml_research_protocol",
        primary_metric="balanced_accuracy",
        condition_means={"logisticregression": 0.57, "randomforestclassifier": 0.92},
        effect_size=effect,
        seed_noise=noise,
        effect_to_noise_ratio=effect / noise,
        best_condition="randomforestclassifier",
    )


def _hypothesis(claim: str) -> ResearchHypothesis:
    return ResearchHypothesis(
        hypothesis_id="H1",
        claim_candidate=claim,
        why_it_matters="test",
        falsification_condition="no effect",
        minimum_evidence_needed="multi-seed run",
    )


def test_claims_carry_executed_provenance() -> None:
    claims = verify_lab_claims(
        _protocol(), _analysis(), _hypothesis("Random forest is robustly best."), "experiment_001"
    )
    assert claims, "verifier returned no claims"
    for claim in claims:
        assert claim.executed_dataset == "eeg_eye_state"
        assert claim.executed_conditions == ["logisticregression", "randomforestclassifier"]
        assert claim.executed_metric == "balanced_accuracy"


def test_mismatch_note_on_the_published_m1_flaw() -> None:
    # The exact published case: prose inherited "breast cancer" from the
    # question while the agent's autonomous selection ran eeg_eye_state.
    claims = verify_lab_claims(
        _protocol(dataset="eeg_eye_state"),
        _analysis(),
        _hypothesis("Feature scaling improves logistic regression accuracy on breast cancer classification."),
        "experiment_001",
    )
    flagged = [c for c in claims if c.provenance_note]
    assert flagged, "prose/executed dataset mismatch was not flagged"
    note = flagged[0].provenance_note
    assert "breast_cancer" in note and "eeg_eye_state" in note


def test_no_note_when_prose_matches_executed_dataset() -> None:
    claims = verify_lab_claims(
        _protocol(dataset="eeg_eye_state"),
        _analysis(),
        _hypothesis("Random forest beats logistic regression on eeg_eye_state."),
        "experiment_001",
    )
    assert all(c.provenance_note is None for c in claims)


def test_no_note_when_prose_names_no_dataset() -> None:
    claims = verify_lab_claims(
        _protocol(dataset="wine"),
        _analysis(),
        _hypothesis("The single-run winner is not robust under seed variance."),
        "experiment_001",
    )
    assert all(c.provenance_note is None for c in claims)


def test_word_boundaries_prevent_substring_false_positives() -> None:
    # "har" must not match inside words like "harder" or "sharpness".
    assert _dataset_mismatch_note("The harder split makes sharpness worse.", "wine") is None
    # Space alias must still match ("breast cancer" for breast_cancer).
    note = _dataset_mismatch_note("scaling helps on breast cancer data", "wine")
    assert note is not None and "breast_cancer" in note


def test_old_ledgers_without_provenance_still_validate() -> None:
    from luckyloop.schemas import LabClaim

    legacy = {
        "claim_id": "experiment_001:robustness",
        "claim": "old entry",
        "verdict": "blocked",
    }
    claim = LabClaim(**legacy)
    assert claim.executed_dataset is None and claim.provenance_note is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all claim-provenance tests passed")
