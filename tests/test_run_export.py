"""Conformance test for the static run.json exporter.

Asserts the projected `Run` dict matches the frontend type field-for-field, that
metrics trace to the source artifacts (not placeholders), and that the
anti-fabrication rule holds (no findings artifact -> empty findings).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from luckyloop.run_export import build_run


def _write(ws: Path, rel: str, obj) -> None:
    (ws / rel).parent.mkdir(parents=True, exist_ok=True)
    (ws / rel).write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "study_result.json", {"state_id": "s_TEST123"})
    _write(ws, "dataset_audit.json", {"dataset_id": "breast_cancer", "n_rows": 569, "n_features": 30})
    _write(
        ws,
        "literature/study_inference.json",
        {
            "included_sources": [
                {"title": "The AI Scientist", "url": "https://arxiv.org/abs/2408.06292", "authors": ["Sakana AI"]},
                {"title": "MLE-bench", "url": "https://arxiv.org/abs/2410.07095", "authors": ["OpenAI"]},
            ]
        },
    )
    _write(
        ws,
        "analyses/analysis_000.json",
        {
            "condition_means": {"logreg (raw)": 0.9579, "+ StandardScaler": 0.9807},
            "condition_stds": {"logreg (raw)": 0.012, "+ StandardScaler": 0.0066},
            "effect_size": 0.0228,
            "seed_noise": 0.0066,
            "best_condition": "+ StandardScaler",
            "summary": "scaling beats raw beyond noise",
        },
    )
    _write(
        ws,
        "claim_ledger.json",
        [{"claim_id": "c0", "claim": "scaling helps", "verdict": "weakly_supported", "reason": "effect > noise"}],
    )
    _write(
        ws,
        "top_model_summary.json",
        {
            "top_models": [
                {"config": "logistic_regression + scale", "metric": 0.9807, "metric_std": 0.0066},
                {"config": "svc (rbf)", "metric": 0.9719, "metric_std": 0.0102},
                {"config": "random_forest", "metric": 0.9649, "metric_std": 0.0147},
            ]
        },
    )
    return ws


def test_run_has_exact_top_level_shape(workspace):
    run = build_run(workspace)
    run.pop("_warnings", None)
    assert set(run) == {
        "currentState", "predictions", "pipeline", "verdict", "papers", "traces", "diffs", "findings",
    }


def test_current_state_fields(workspace):
    cs = build_run(workspace)["currentState"]
    assert set(cs) == {"id", "dataset", "context"}
    assert cs["id"] == "s_TEST123"
    assert cs["dataset"] == "breast_cancer"
    assert "4 papers" not in cs["context"] and "2 papers" in cs["context"]


def test_predictions_shape_and_real_values(workspace):
    preds = build_run(workspace)["predictions"]
    assert len(preds) == 3
    assert [p["rank"] for p in preds] == [1, 2, 3]
    top = preds[0]
    assert top["config"] == "logistic_regression + scale"
    # accLow/High are mean ± real std from the artifact (not placeholders).
    assert top["accLow"] == pytest.approx(0.9807 - 0.0066, abs=1e-4)
    assert top["accHigh"] == pytest.approx(0.9807 + 0.0066, abs=1e-4)
    # probability is a softmax distribution -> sums to 1, ranked descending.
    assert math.isclose(sum(p["probability"] for p in preds), 1.0, abs_tol=1e-3)
    assert preds[0]["probability"] >= preds[1]["probability"] >= preds[2]["probability"]
    for p in preds:
        assert set(p) >= {"rank", "id", "config", "accLow", "accHigh", "probability"}
        assert set(p["tag"]) == {"label", "tone"}


def test_verdict_traces_to_analysis(workspace):
    v = build_run(workspace)["verdict"]
    assert set(v) == {"state", "title", "titleLab", "reason", "reasonLab", "effect", "noise"}
    assert v["effect"] == 0.0228 and v["noise"] == 0.0066  # straight from analysis
    assert v["state"] == "CONFIRMED"  # weakly_supported + effect > noise


def test_diffs_progression_real(workspace):
    diffs = build_run(workspace)["diffs"]
    assert [d["step"] for d in diffs] == ["logreg (raw)", "+ StandardScaler"]
    scaled = diffs[1]
    assert scaled["before"] == 0.9579 and scaled["after"] == 0.9807
    assert scaled["delta"] == pytest.approx(0.0228, abs=1e-4)


def test_papers_exact_titles_and_refs(workspace):
    papers = build_run(workspace)["papers"]
    assert {p["ref"] for p in papers} == {"arXiv:2408.06292", "arXiv:2410.07095"}
    for p in papers:
        assert set(p) == {"title", "authors", "ref", "url"}
        assert p["url"].startswith("https://arxiv.org/abs/")


def test_traces_include_real_dataset_dims(workspace):
    traces = build_run(workspace)["traces"]
    load = [t for t in traces if t["source"].startswith("sklearn:")]
    assert load and load[0]["event"] == "Loaded 569×30"
    for t in traces:
        assert set(t) == {"time", "source", "event"}


def test_pipeline_seven_steps_real_numbers(workspace):
    pipeline = build_run(workspace)["pipeline"]
    assert len(pipeline) == 7
    for step in pipeline:
        assert set(step) == {"key", "title", "detail", "speak", "log"}
        for line in step["log"]:
            assert "tag" in line and "text" in line
            assert line.get("tone") is not None or "tone" not in line  # never null
    cross = next(s for s in pipeline if s["key"] == "cross")
    assert "effect=0.0228 vs seed_noise=0.0066" in cross["log"][0]["text"]


def test_findings_empty_without_real_artifact(workspace):
    # Anti-fabrication: no findings.json -> empty, never the fixture's stats.
    run = build_run(workspace)
    assert run["findings"] == []
    assert any("findings" in w for w in run["_warnings"])


def test_findings_read_from_real_artifact(workspace):
    _write(workspace, "findings.json", [{"stat": "r = 0.45", "label": "calibration → compute saved"}])
    findings = build_run(workspace)["findings"]
    assert findings == [{"stat": "r = 0.45", "label": "calibration → compute saved"}]
