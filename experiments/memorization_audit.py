#!/usr/bin/env python3
"""Audit mémorisation du world-model — le warning cut-off du mentor, quantifié.

Question : la calibration du world-model sur breast_cancer/wine/digits est-elle de la VRAIE
prédiction (raisonnement sur la structure de l'expérience) ou du RECALL (datasets célèbres,
benchmarks pré-cutoff mémorisés) ?

Design propre (contrôle du confound) — on prédit l'accuracy test sous 3 conditions :
  - canonical_named : dataset célèbre AVEC son nom ("breast_cancer, 569 samples...")
  - canonical_anon  : MÊME dataset, nom CACHÉ (seulement les stats : "569 samples, 30 features, 2 classes")
  - synthetic_anon  : dataset synthétique (make_classification, seed random) — non mémorisable

Lecture :
  - Si erreur(canonical_named) << erreur(canonical_anon) ≈ erreur(synthetic_anon)
    → l'avantage du world-model vient du NOM (recall), pas du raisonnement. Mémorisation confirmée.
  - Si erreur similaire dans les 3 → vraie capacité de prédiction structurelle. Pas de contamination.

Endpoint : vLLM local sur le box (Qwen/Qwen-AgentWorld-35B-A3B).
    python3 experiments/memorization_audit.py
"""
from __future__ import annotations
import json, os, time, urllib.request, warnings
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen-AgentWorld-35B-A3B")
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
ROOT = Path(__file__).resolve().parents[1]
MODELS = ["logistic_regression", "random_forest"]

CANONICAL = {  # nom célèbre -> loader
    "breast_cancer": load_breast_cancer, "wine": load_wine,
    "digits": load_digits, "iris": load_iris,
}


def synth(seed):
    X, y = make_classification(n_samples=800, n_features=20, n_informative=10, n_redundant=0,
                               n_classes=2, class_sep=1.0, random_state=seed)
    return X, y


def build(model):
    return (LogisticRegression(max_iter=1000, random_state=0) if model == "logistic_regression"
            else RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))


def run_real(X, y, model):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = build(model)
    if model == "logistic_regression":
        clf = make_pipeline(StandardScaler(), clf)
    clf.fit(Xtr, ytr)
    return round(accuracy_score(yte, clf.predict(Xte)), 4)


def llm_predict(desc: str) -> dict | None:
    sysmsg = ("You are a world-model that predicts ML experiment outcomes BEFORE they run. "
              "Reason from the experiment structure. Be honest about uncertainty. Answer ONLY JSON.")
    user = (f"Predict the TEST accuracy (0-1) for this experiment:\n{desc}\n"
            "25% test split, default-ish hyperparameters.\n"
            'JSON: {"point": <float 0-1>, "lo": <float>, "hi": <float>}\n'
            "lo/hi = your 90% interval.")
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
            "temperature": 0.2, "max_tokens": 800,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {API_KEY}",
                                          "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            msg = json.loads(r.read())["choices"][0]["message"]
        txt = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        i, j = txt.find("{"), txt.rfind("}")
        return json.loads(txt[i:j + 1])
    except Exception as e:
        print(f"  LLM err: {e}", flush=True); return None


def named_desc(name, n, d, k, model):
    return f"- dataset: the well-known scikit-learn '{name}' dataset ({n} samples, {d} features, {k} classes)\n- model: {model}"


def anon_desc(n, d, k, model):
    return f"- dataset: an unnamed tabular dataset with {n} samples, {d} features, {k} classes\n- model: {model}"


def evaluate(group, rows):
    errs = [r["abs_err"] for r in rows if r["group"] == group]
    covs = [r["covered"] for r in rows if r["group"] == group]
    if not errs:
        return None
    return {"n": len(errs), "MAE": round(float(np.mean(errs)), 4),
            "interval_coverage": round(float(np.mean(covs)), 3)}


def main():
    rows = []
    # canonical : named + anon (même data réelle)
    for name, loader in CANONICAL.items():
        X, y = loader(return_X_y=True)
        n, d, k = X.shape[0], X.shape[1], len(set(y))
        for model in MODELS:
            actual = run_real(X, y, model)
            for group, desc in (("canonical_named", named_desc(name, n, d, k, model)),
                                ("canonical_anon", anon_desc(n, d, k, model))):
                pred = llm_predict(desc)
                if not pred:
                    continue
                lo, hi = float(pred.get("lo", 0)), float(pred.get("hi", 1))
                pt = float(pred.get("point", (lo + hi) / 2))
                rows.append({"group": group, "name": name, "model": model, "pred": round(pt, 4),
                             "lo": round(lo, 4), "hi": round(hi, 4), "actual": actual,
                             "abs_err": round(abs(pt - actual), 4), "covered": bool(lo <= actual <= hi)})
                print(f"{group:16s} {name:13s} {model:18s} | pred {pt:.3f} [{lo:.2f},{hi:.2f}] | actual {actual:.3f} | err {abs(pt-actual):.3f}", flush=True)
    # synthetic : anon only (non mémorisable)
    for seed in range(4):
        X, y = synth(seed)
        n, d, k = X.shape[0], X.shape[1], len(set(y))
        for model in MODELS:
            actual = run_real(X, y, model)
            pred = llm_predict(anon_desc(n, d, k, model))
            if not pred:
                continue
            lo, hi = float(pred.get("lo", 0)), float(pred.get("hi", 1))
            pt = float(pred.get("point", (lo + hi) / 2))
            rows.append({"group": "synthetic_anon", "name": f"synth_{seed}", "model": model, "pred": round(pt, 4),
                         "lo": round(lo, 4), "hi": round(hi, 4), "actual": actual,
                         "abs_err": round(abs(pt - actual), 4), "covered": bool(lo <= actual <= hi)})
            print(f"{'synthetic_anon':16s} synth_{seed}       {model:18s} | pred {pt:.3f} [{lo:.2f},{hi:.2f}] | actual {actual:.3f} | err {abs(pt-actual):.3f}", flush=True)

    groups = ["canonical_named", "canonical_anon", "synthetic_anon"]
    summary = {g: evaluate(g, rows) for g in groups}
    nmae = summary.get("canonical_named", {}).get("MAE")
    amae = summary.get("canonical_anon", {}).get("MAE")
    smae = summary.get("synthetic_anon", {}).get("MAE")
    recall_gap = round(amae - nmae, 4) if (nmae is not None and amae is not None) else None
    out = {"model": MODEL, "n_predictions": len(rows), "by_group": summary,
           "recall_gap_anon_minus_named": recall_gap, "rows": rows}
    rep = ROOT / "reports"; rep.mkdir(exist_ok=True)
    (rep / "memorization_audit.json").write_text(json.dumps(out, indent=2))

    print("\n═══ AUDIT MÉMORISATION ═══", flush=True)
    for g in groups:
        s = summary[g]
        if s: print(f"{g:16s} | MAE {s['MAE']:.3f} | couverture {s['interval_coverage']*100:.0f}% | n={s['n']}", flush=True)
    if recall_gap is not None:
        print(f"\nRecall gap (MAE_anon − MAE_named) = {recall_gap:+.3f}", flush=True)
        if recall_gap > 0.02:
            print("→ Le world-model prédit MIEUX avec le nom qu'avec les stats seules.", flush=True)
            print("  Une partie de sa calibration sur datasets célèbres = RECALL, pas raisonnement.", flush=True)
            print("  Sur tâches NOVEL (synthétiques), c'est la vraie capacité de prédiction qui compte.", flush=True)
        else:
            print("→ Pas d'écart notable nom vs stats : la prédiction vient du raisonnement structurel, pas du recall.", flush=True)
    print("\nreports/memorization_audit.json", flush=True)
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
