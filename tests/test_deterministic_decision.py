"""Regression test for the deterministic planner used by the fixed-dataset
ablation path. Previously `deterministic_decision` raised
'no candidate actions available' whenever the only candidate action had a kind
outside its hard-coded preferred list (notably `run_baseline`), which broke
`lab.py --study <study> --planner deterministic --no-require-agent`.
"""
from __future__ import annotations

import pytest

from luckyloop.lab_scientist import deterministic_decision
from luckyloop.schemas import LabAction, LabQuestion, LabStudyState


def _state() -> LabStudyState:
    return LabStudyState(lab_question=LabQuestion(question="q", study_id="seed_variance_claim"))


def _action(kind: str) -> LabAction:
    return LabAction(action_id=f"a_{kind}", kind=kind, scientific_goal="g", command="python -c \"print(1)\"")


def test_run_baseline_only_does_not_crash():
    # The exact case that raised before the fix.
    d = deterministic_decision(_state(), [_action("run_baseline")])
    assert d.preferred_action_id == "a_run_baseline"


def test_unhandled_kind_falls_back_to_first_action():
    d = deterministic_decision(_state(), [_action("analyze_results")])
    assert d.preferred_action_id == "a_analyze_results"


def test_preferred_priority_unchanged_when_protocol_present():
    # run_protocol must still win over run_baseline (no behaviour change).
    d = deterministic_decision(_state(), [_action("run_baseline"), _action("run_protocol")])
    assert d.preferred_action_id == "a_run_protocol"


def test_stop_action_sets_stop():
    d = deterministic_decision(_state(), [_action("stop_and_report")])
    assert d.stop_or_continue == "stop_and_report"


def test_empty_actions_raises():
    with pytest.raises(ValueError):
        deterministic_decision(_state(), [])
