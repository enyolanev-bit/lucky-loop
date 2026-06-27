"""Tests for external-agent planner handoff: safety-selector override + fake backend contract.

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_handoff.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from luckyloop.research_agent import validate_agent_decision
from luckyloop.schemas import AgentDecision, ProposedAction

ROOT = Path(__file__).resolve().parents[1]


def _catalog() -> list[ProposedAction]:
    return [
        ProposedAction(action_id="action_logistic_regression_C-1p0_scale-False",
                       command="train", model="logistic_regression", params={"scale": False}),
        ProposedAction(action_id="action_logistic_regression_C-1p0_scale-True",
                       command="train", model="logistic_regression", params={"scale": True}),
        ProposedAction(action_id="action_top_model_verification",
                       command="verify", model="top_model_verification", params={}),
    ]


def _decision(preferred: str, candidates: list[str]) -> AgentDecision:
    return AgentDecision(
        research_question="q", working_hypothesis="h", candidate_action_ids=candidates,
        preferred_action_id=preferred, rationale="r", expected_evidence_needed="e", claim_risk="risk",
    )


def test_valid_decision_passes_unchanged():
    cat = _catalog()
    dec = _decision(cat[0].action_id, [cat[0].action_id, cat[1].action_id])
    out = validate_agent_decision(dec, cat)
    assert out.preferred_action_id == cat[0].action_id


def test_strict_mode_rejects_invalid():
    cat = _catalog()
    bad = _decision("action_DOES_NOT_EXIST", [cat[0].action_id])
    with pytest.raises(ValueError):
        validate_agent_decision(bad, cat)


def test_override_replaces_invalid_preferred_with_safe_catalog_action():
    cat = _catalog()
    bad = _decision("rm -rf /", ["rm -rf /", cat[1].action_id])
    out = validate_agent_decision(bad, cat, override_on_invalid=True)
    valid_ids = {c.action_id for c in cat}
    assert out.preferred_action_id in valid_ids          # safety selector picked a safe action
    assert all(a in valid_ids for a in out.candidate_action_ids)  # unknowns sanitized
    assert "safety-selector override" in out.rationale    # override is recorded for audit


def test_override_keeps_valid_preferred_but_sanitizes_candidates():
    cat = _catalog()
    dec = _decision(cat[1].action_id, [cat[1].action_id, "ghost_action"])
    out = validate_agent_decision(dec, cat, override_on_invalid=True)
    assert out.preferred_action_id == cat[1].action_id
    assert "ghost_action" not in out.candidate_action_ids


def test_fake_agent_backend_produces_valid_response(tmp_path):
    """The reference backend must return a response that passes the safe-catalog validator."""
    cat = _catalog()
    request = {
        "task": {"dataset": "breast_cancer", "primary_metric": "accuracy", "goal": "max acc"},
        "state": {"state_id": "s0", "top_model_summary": {}},
        "candidate_catalog": [c.model_dump() for c in cat],
        "prior_traces_summary": [],
    }
    req_path = tmp_path / "s0.request.json"
    resp_path = tmp_path / "s0.response.json"
    req_path.parent.mkdir(parents=True, exist_ok=True)
    req_path.write_text(json.dumps(request), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fake_agent_backend.py"), str(req_path), str(resp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(resp_path.read_text(encoding="utf-8"))
    decision = AgentDecision(**data)
    out = validate_agent_decision(decision, cat, override_on_invalid=True)
    assert out.preferred_action_id in {c.action_id for c in cat}
