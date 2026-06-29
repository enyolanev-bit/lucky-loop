#!/usr/bin/env python3
"""Produce a lab workspace whose numbers are REALLY computed by sklearn.

No completed run exists in the repo, and the full loop can't run here (no
`openai` dep, no Qwen-AgentWorld simulator, no network). This script bypasses
the LLM orchestration and runs the COMPUTATIONAL core for real: it fits sklearn
models on breast_cancer across seeds and writes the resulting metrics as
workspace artifacts in the same shapes a real run emits (LabAnalysis, LabClaim,
DatasetAudit, top_model_summary, literature/study_inference). Every accuracy,
effect size, seed-noise value and verdict below is measured, not invented.

Provenance is stamped in `_provenance.json` as `offline_real_metrics_no_llm`
so nobody mistakes it for a full end-to-end live run.

    PYTHONPATH=src .venv/bin/python scripts/make_demo_workspace.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luckyloop.literature import CURATED_REFERENCES, _extract_arxiv_id  # noqa: E402
from luckyloop.schemas import DatasetAudit, LabAnalysis, LabClaim  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEEDS = [0, 1, 2, 3, 4]
DATASET = "breast_cancer"
SCENARIO = "scaling"  # overridden by --scenario in __main__
# The 4 references shown by the frontend demo — all real, already verified.
DEMO_REF_IDS = {"2606.24597", "2408.06292", "2501.04227", "2410.07095"}


def _fit_score(make_clf, X, y, seed: int) -> float:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    clf = make_clf(seed)
    clf.fit(X_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(X_te)))


def _mean_std(make_clf, X, y) -> tuple[float, float, list[float]]:
    scores = [_fit_score(make_clf, X, y, s) for s in SEEDS]
    mean = float(statistics.fmean(scores))
    std = float(statistics.pstdev(scores))
    return mean, std, scores


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    n_rows, n_features = int(X.shape[0]), int(X.shape[1])

    # --- candidate MODELS (drive predictions: ranked, with real spread) ------
    candidates = {
        "logistic_regression + scale": lambda s: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, C=1.0, random_state=s)
        ),
        "random_forest": lambda s: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=-1),
        "svc (rbf)": lambda s: make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=s)),
    }
    model_stats = {name: _mean_std(fn, X, y) for name, fn in candidates.items()}
    top_models = [
        {
            "run_id": f"s_{DATASET[:3].upper()}{i}",
            "model": name.split(" ")[0],
            "model_key": name,
            "config": name,
            "metric": round(mean, 4),
            "metric_std": round(std, 4),
            "params": {},
        }
        for i, (name, (mean, std, _)) in enumerate(sorted(model_stats.items(), key=lambda kv: kv[1][0], reverse=True))
    ]
    best_model = top_models[0]
    top_model_summary = {
        "best_observed_model": best_model["model_key"],
        "best_observed_metric": best_model["metric"],
        "top_models": top_models,
        "top_gap": round(top_models[0]["metric"] - top_models[1]["metric"], 4) if len(top_models) > 1 else None,
        "needs_robustness_verification": True,
        "reason": "single-split observation; robustness not yet established",
    }

    # --- ablation that drives diffs + verdict (effect vs seed noise) ----------
    if SCENARIO == "scaling":
        # Manipulated variable = feature scaling. On breast_cancer the effect
        # really clears seed noise -> a genuinely CONFIRMED claim.
        protocol_id = "breast_cancer_scaling_ablation"
        raw_mean, raw_std, _ = _mean_std(
            lambda s: LogisticRegression(max_iter=5000, C=1.0, random_state=s), X, y
        )
        scaled_mean, scaled_std, scaled_scores = _mean_std(
            lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0, random_state=s)), X, y
        )
        tuned_best = max(
            (
                (C, _mean_std(lambda s, C=C: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C, random_state=s)), X, y))
                for C in (0.1, 1.0, 10.0)
            ),
            key=lambda kv: kv[1][0],
        )
        tuned_C, (tuned_mean, tuned_std, tuned_scores) = tuned_best
        condition_means = {
            "logreg (raw)": round(raw_mean, 4),
            "+ StandardScaler": round(scaled_mean, 4),
            f"tune C={tuned_C}": round(tuned_mean, 4),
        }
        condition_stds = {
            "logreg (raw)": round(raw_std, 4),
            "+ StandardScaler": round(scaled_std, 4),
            f"tune C={tuned_C}": round(tuned_std, 4),
        }
        best_condition = max(condition_means, key=condition_means.get)
        baseline_mean = condition_means["logreg (raw)"]
        effect_size = round(condition_means[best_condition] - baseline_mean, 4)
        seed_noise = round(float(statistics.pstdev(tuned_scores if best_condition.startswith("tune") else scaled_scores)), 4)
    else:
        # close_models: a genuinely tied pair. On breast_cancer the two weaker
        # candidates (svc vs random_forest) lead each other by less than their
        # seed noise, so the verifier BLOCKS the "B beats C" claim. Real, not
        # rigged: the top model (logreg+scale) really is best and is NOT the
        # subject here — this claim is about the close pair only.
        protocol_id = "breast_cancer_close_pair_within_noise"
        ranked = sorted(model_stats.items(), key=lambda kv: kv[1][0], reverse=True)
        (hi_name, (hi_mean, hi_std, hi_scores)), (lo_name, (lo_mean, lo_std, _)) = ranked[1], ranked[2]
        condition_means = {lo_name: round(lo_mean, 4), hi_name: round(hi_mean, 4)}
        condition_stds = {lo_name: round(lo_std, 4), hi_name: round(hi_std, 4)}
        best_condition = hi_name
        baseline_mean = round(lo_mean, 4)
        effect_size = round(hi_mean - lo_mean, 4)
        # Conservative, honest noise: the larger seed std of the two candidates.
        seed_noise = round(max(hi_std, lo_std), 4)

    supported = effect_size > seed_noise
    analysis = LabAnalysis(
        analysis_id="analysis_000",
        protocol_id=protocol_id,
        primary_metric="accuracy",
        condition_means=condition_means,
        condition_stds=condition_stds,
        effect_size=effect_size,
        seed_noise=seed_noise,
        effect_to_noise_ratio=round(effect_size / seed_noise, 3) if seed_noise else None,
        best_condition=best_condition,
        protocol_warnings=["single stratified split per seed; no nested CV"],
        summary=(
            f"Best condition '{best_condition}' mean={condition_means[best_condition]} vs raw {baseline_mean}; "
            f"effect={effect_size}, seed_noise={seed_noise} over seeds {SEEDS}."
        ),
    )

    verdict_word = "weakly_supported" if supported else "blocked"
    if SCENARIO == "scaling":
        claim_text = "Feature scaling improves logistic regression accuracy on breast_cancer beyond seed noise."
    else:
        # close_models compares the two CLOSEST candidates (not the overall top model),
        # so the claim is scoped to that pair — never an overall "best model" claim.
        other = next((k for k in condition_means if k != best_condition), "the other candidate")
        claim_text = (
            f"Between the two closest candidates, '{best_condition}' beats '{other}' on "
            f"breast_cancer beyond seed noise (close-pair comparison, not a best-model-overall claim)."
        )
    claim = LabClaim(
        claim_id="claim_000",
        hypothesis_id="hyp_scaling" if SCENARIO == "scaling" else "hyp_best_model",
        claim=claim_text,
        verdict=verdict_word,
        evidence_ids=["analysis_000"],
        reason=(
            f"Best mean accuracy at {best_condition}, but the observed effect ({effect_size}) is "
            f"{'greater than' if supported else 'within'} the across-seed std ({seed_noise}). "
            + ("Claim weakly supported." if supported else "On a single split this is within noise — no best-model claim is supported.")
        ),
        metrics={"effect_size": effect_size, "seed_noise": seed_noise, "best_condition": best_condition},
    )

    audit = DatasetAudit(
        dataset_id=DATASET,
        source="sklearn",
        status="accepted",
        target_column="target",
        feature_columns=[str(c) for c in data.feature_names],
        n_rows=n_rows,
        n_features=n_features,
        class_counts={str(int(c)): int((y == c).sum()) for c in np.unique(y)},
    )

    # --- literature (real curated refs the demo cites) -----------------------
    included_sources = []
    for ref in CURATED_REFERENCES:
        arxiv_id, _ = _extract_arxiv_id(ref.get("url", ""))
        if arxiv_id not in DEMO_REF_IDS:
            continue
        included_sources.append(
            {
                "citation_id": f"arxiv_{arxiv_id.replace('.', '_')}",
                "title": ref["title"],
                "authors": ref.get("authors", []),
                "url": ref["url"],
                "abstract": ref.get("abstract", "")[:1200],
                "used_for": ref.get("used_for", []),
                "relevance_score": 0.9,
            }
        )
    # Preserve the demo ordering (Qwen-AgentWorld first).
    order = ["2606.24597", "2408.06292", "2501.04227", "2410.07095"]
    included_sources.sort(key=lambda s: order.index(_extract_arxiv_id(s["url"])[0]) if _extract_arxiv_id(s["url"])[0] in order else 99)
    study_inference = {"queries": ["feature scaling logistic regression breast_cancer"], "included_sources": included_sources}

    state_id = f"s_{abs(hash((DATASET, effect_size, seed_noise))) % 0x10000000:07X}"

    # --- write artifacts in real shapes -------------------------------------
    slug = f"breast_cancer_{SCENARIO}_offline_real"
    ws = ROOT / "reports" / "lab" / slug
    (ws / "literature").mkdir(parents=True, exist_ok=True)
    (ws / "analyses").mkdir(parents=True, exist_ok=True)

    def _w(rel: str, obj) -> None:
        (ws / rel).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    _w("research_program.json", {"mode": "ml_research_validity", "dataset": DATASET, "study_id": state_id})
    _w("dataset_audit.json", audit.model_dump())
    _w("literature/study_inference.json", study_inference)
    _w("literature/literature_brief.json", {"study_id": state_id, "study_inference": study_inference})
    _w("analyses/analysis_000.json", analysis.model_dump())
    _w("claim_ledger.json", [claim.model_dump()])
    _w("top_model_summary.json", top_model_summary)
    _w("study_result.json", {"state_id": state_id, "summary": analysis.summary})
    (ws / "notebook.jsonl").write_text(
        json.dumps({"step": 0, "state_id": state_id, "selected_action": {"kind": "run_protocol"}}) + "\n",
        encoding="utf-8",
    )
    _w(
        "_provenance.json",
        {
            "generator": "scripts/make_demo_workspace.py",
            "mode": "offline_real_metrics_no_llm",
            "seeds": SEEDS,
            "note": "sklearn metrics are really computed; LLM scientist + Qwen world model are bypassed.",
        },
    )

    print(f"Wrote real-metrics workspace: {ws.relative_to(ROOT)}")
    print(f"  models (mean acc): " + ", ".join(f"{m['model_key']}={m['metric']}±{m['metric_std']}" for m in top_models))
    print(f"  ablation: {condition_means}")
    print(f"  effect={effect_size} seed_noise={seed_noise} -> verdict={verdict_word.upper()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=["scaling", "close_models"],
        default="scaling",
        help="scaling -> real CONFIRMED (scaling clears noise); close_models -> real BLOCKED (top two within noise)",
    )
    SCENARIO = parser.parse_args().scenario
    main()
