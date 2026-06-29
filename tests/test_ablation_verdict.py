"""Verdict-logic tests for scripts/analyze_ablation.py.

Guards the integrity rule: only behavioural deltas (experiment runs, claim
verdicts) may trigger 'EFFECT'. Runtime jitter and Qwen free-text differences
must NEVER turn a null into an effect.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_ablation", ROOT / "scripts" / "analyze_ablation.py")
analyze_ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_ablation)
build_report = analyze_ablation.build_report


def run(*, runs, runtime, supported, blocked, source, fingerprint):
    return {
        "workspace": "ws",
        "wm_source": source,
        "steps": 6,
        "experiment_runs": runs,
        "total_runtime_s": runtime,
        "claims_total": supported + blocked,
        "claims_supported": supported,
        "claims_blocked": blocked,
        "claims_observation_inconclusive": 0,
        "runs_to_first_supported": 2 if supported else None,
        "nontrivial_signal": False,
        "signal_fingerprint": fingerprint,
    }


def test_identical_metrics_with_runtime_jitter_and_text_diff_is_null():
    # Same runs, same claims; only runtime differs (jitter) and Qwen text differs.
    on = [run(runs=4, runtime=0.052, supported=1, blocked=0, source="qwen_agentworld",
              fingerprint=["('run', 0.0, 0.5, 'adds_observation')", "('run', 0.1, 0.7, 'reduces_uncertainty')"])]
    off = [run(runs=4, runtime=0.041, supported=1, blocked=0, source="plumbing_not_called",
               fingerprint=["('run', 0.0, 0.5, 'adds_observation')"])]
    v = build_report(on, off)["verdict"]
    assert v.startswith("NULL"), v


def test_big_runtime_delta_alone_is_still_null():
    on = [run(runs=3, runtime=200.0, supported=0, blocked=1, source="qwen_agentworld", fingerprint=["x"])]
    off = [run(runs=3, runtime=0.2, supported=0, blocked=1, source="plumbing_not_called", fingerprint=["x"])]
    v = build_report(on, off)["verdict"]
    assert v.startswith("NULL"), v


def test_claims_delta_triggers_effect():
    on = [run(runs=4, runtime=0.05, supported=1, blocked=0, source="qwen_agentworld", fingerprint=["x"])]
    off = [run(runs=4, runtime=0.05, supported=0, blocked=1, source="plumbing_not_called", fingerprint=["x"])]
    v = build_report(on, off)["verdict"]
    assert v.startswith("EFFECT MEASURED"), v


def test_compute_runs_delta_triggers_effect():
    on = [run(runs=2, runtime=0.05, supported=1, blocked=0, source="qwen_agentworld", fingerprint=["x"])]
    off = [run(runs=4, runtime=0.05, supported=1, blocked=0, source="plumbing_not_called", fingerprint=["x"])]
    rep = build_report(on, off)
    assert rep["verdict"].startswith("EFFECT MEASURED"), rep["verdict"]
    assert rep["world_model_effect"]["compute_runs_saved_by_wm"] == 2.0


def test_runtime_never_in_verdict_drivers():
    rep = build_report(
        [run(runs=4, runtime=1.0, supported=1, blocked=0, source="qwen_agentworld", fingerprint=["x"])],
        [run(runs=4, runtime=9.0, supported=1, blocked=0, source="plumbing_not_called", fingerprint=["x"])],
    )
    assert rep["verdict"].startswith("NULL")
