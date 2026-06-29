"""Verdict-logic tests for scripts/analyze_ablation.py.

Guards the integrity rule: only behavioural deltas (experiment runs, claim
verdicts) may trigger 'EFFECT'. Runtime jitter and Qwen free-text differences
must NEVER turn a null into an effect.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_ablation", ROOT / "scripts" / "analyze_ablation.py")
analyze_ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_ablation)
build_report = analyze_ablation.build_report
metrics_for_workspace = analyze_ablation.metrics_for_workspace


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


def arm(*, runs, runtime, supported, blocked, source, fingerprint, n=3):
    """A well-powered arm: the same run repeated n>=MIN_RUNS times."""
    return [run(runs=runs, runtime=runtime, supported=supported, blocked=blocked,
                source=source, fingerprint=fingerprint) for _ in range(n)]


def test_claims_delta_triggers_effect():
    on = arm(runs=4, runtime=0.05, supported=1, blocked=0, source="qwen_agentworld", fingerprint=["x"])
    off = arm(runs=4, runtime=0.05, supported=0, blocked=1, source="plumbing_not_called", fingerprint=["x"])
    v = build_report(on, off)["verdict"]
    assert v.startswith("EFFECT MEASURED"), v


def test_compute_runs_delta_triggers_effect():
    on = arm(runs=2, runtime=0.05, supported=1, blocked=0, source="qwen_agentworld", fingerprint=["x"])
    off = arm(runs=4, runtime=0.05, supported=1, blocked=0, source="plumbing_not_called", fingerprint=["x"])
    rep = build_report(on, off)
    assert rep["verdict"].startswith("EFFECT MEASURED"), rep["verdict"]
    assert rep["world_model_effect"]["compute_runs_saved_by_wm"] == 2.0


def test_runtime_never_in_verdict_drivers():
    rep = build_report(
        arm(runs=4, runtime=1.0, supported=1, blocked=0, source="qwen_agentworld", fingerprint=["x"]),
        arm(runs=4, runtime=9.0, supported=1, blocked=0, source="plumbing_not_called", fingerprint=["x"]),
    )
    assert rep["verdict"].startswith("NULL")


def test_underpowered_effect_is_flagged_not_measured():
    # A real behavioural delta but only 1 run/arm — must be flagged UNDER-POWERED, not "MEASURED".
    on = [run(runs=2, runtime=0.05, supported=1, blocked=0, source="qwen_agentworld", fingerprint=["x"])]
    off = [run(runs=4, runtime=0.05, supported=1, blocked=0, source="plumbing_not_called", fingerprint=["x"])]
    rep = build_report(on, off)
    assert rep["verdict"].startswith("EFFECT (UNDER-POWERED"), rep["verdict"]
    assert rep["world_model_effect"]["underpowered"] is True
    assert rep["world_model_effect"]["min_runs_required"] == 3


def test_well_powered_effect_is_not_flagged_underpowered():
    on = arm(runs=2, runtime=0.05, supported=1, blocked=0, source="qwen_agentworld", fingerprint=["x"])
    off = arm(runs=4, runtime=0.05, supported=1, blocked=0, source="plumbing_not_called", fingerprint=["x"])
    rep = build_report(on, off)
    assert rep["world_model_effect"]["underpowered"] is False
    assert "UNDER-POWERED" not in rep["verdict"], rep["verdict"]


def test_mislabeled_arm_sources_warn_in_verdict():
    # ON arm carries the fallback source (arms swapped) — verdict must be prefixed with a warning.
    on = arm(runs=4, runtime=0.05, supported=1, blocked=0, source="plumbing_not_called", fingerprint=["x"])
    off = arm(runs=4, runtime=0.05, supported=0, blocked=1, source="qwen_agentworld", fingerprint=["x"])
    rep = build_report(on, off)
    assert rep["world_model_effect"]["arm_sources_ok"] is False
    assert rep["verdict"].startswith("⚠ ARM SOURCES UNEXPECTED"), rep["verdict"]


def test_correct_arm_sources_pass():
    on = arm(runs=4, runtime=0.05, supported=1, blocked=0, source="qwen_agentworld", fingerprint=["x"])
    off = arm(runs=4, runtime=0.05, supported=1, blocked=0, source="plumbing_not_called", fingerprint=["x"])
    rep = build_report(on, off)
    assert rep["world_model_effect"]["arm_sources_ok"] is True
    assert not rep["verdict"].startswith("⚠"), rep["verdict"]


def test_duplicate_claim_rows_counted_once():
    # A claim_ledger with the same row written twice must count as ONE claim,
    # so a duplicated 'supported' row cannot inflate the delta into a spurious effect.
    dup = {"claim_id": "c1", "verdict": "supported", "claim": "scaling helps"}
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        ws = Path(d)
        (ws / "notebook.jsonl").write_text(
            json.dumps({"step": 1, "selected_action": {"kind": "run_protocol"},
                        "actual_observation": {"status": "success", "runtime_seconds": 0.1}}) + "\n",
            encoding="utf-8",
        )
        (ws / "claim_ledger.json").write_text(json.dumps([dup, dup]), encoding="utf-8")
        m = metrics_for_workspace(ws)
    assert m["claims_total"] == 1, m
    assert m["claims_supported"] == 1, m
