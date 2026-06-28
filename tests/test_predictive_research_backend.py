from __future__ import annotations

from luckyloop.schemas import ResearchAction, ResearchProgram
from luckyloop.simulator import predict
from luckyloop.schemas import ProposedAction
from luckyloop.verifier import verify_sweep


def test_research_program_schema_accepts_protocol_actions():
    action = ResearchAction(
        action_id="task:protocol_probe",
        kind="protocol_probe",
        task_id="task",
        description="Probe whether a metric should be claim-blocked.",
        expected_claim_change="Blocks or rewrites fragile claims.",
        risk_focus=["metric_misuse"],
        produces=["runs/task/run_001.json"],
    )
    program = ResearchProgram(
        question="Can a world model improve claim quality?",
        goal="Predict before compute.",
        selected_tasks=["task"],
        candidate_research_actions=[action],
        baselines=["classic_autoresearch", "classic_verified", "lucky_loop_full"],
    )
    assert program.candidate_research_actions[0].kind == "protocol_probe"


def test_prediction_fallback_populates_research_action_fields(monkeypatch):
    monkeypatch.delenv("LUCKYWORLD_SIMULATOR_BASE_URL", raising=False)
    monkeypatch.delenv("LUCKYWORLD_SIMULATOR_MODEL", raising=False)
    prediction = predict(
        ProposedAction(
            action_id="action_protocol_probe",
            command="echo ok",
            model="protocol_probe",
            params={"dataset": "breast_cancer"},
        ),
        state="needs_robustness_verification",
    )
    assert prediction.expected_claim_delta in {"blocks_or_rewrites_claim", "enables_claim"}
    assert prediction.compute_waste_risk <= 0.5
    assert prediction.why_not_classic_autoresearch


def test_protocol_warning_blocks_claim_even_with_large_effect():
    verification = verify_sweep(
        {
            "type": "protocol_probe",
            "metric": "accuracy",
            "protocol_warning": "metric-only result is protocol-fragile",
            "runs": [
                {"config_key": "A", "label": "A", "accuracy": 0.99, "seed": 0},
                {"config_key": "A", "label": "A", "accuracy": 0.98, "seed": 1},
                {"config_key": "B", "label": "B", "accuracy": 0.75, "seed": 0},
                {"config_key": "B", "label": "B", "accuracy": 0.76, "seed": 1},
            ],
        }
    )
    assert verification.status == "inconclusive"
    assert verification.trustworthy is False
    assert "protocol warning blocks" in verification.rationale.lower()
