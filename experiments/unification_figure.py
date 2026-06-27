#!/usr/bin/env python3
"""Figure d'unification — la guidance ne vaut que ce que vaut la calibration.

Pour chaque dataset, on mesure DEUX choses et on les corrèle :
  X = qualité de calibration du world-model = corrélation de rang (Spearman) entre
      les métriques PRÉDITES et les métriques RÉELLES des candidats (range-t-il bien ?).
  Y = compute économisé par la guidance vs ordre aléatoire (l'ablation).

Thèse visuelle : calibration haute → grosse économie ; calibration basse/négative → la guidance coûte.
→ unifie les 2 trust layers (predict-before-compute + verify-before-claim).

    python unification_figure.py   # -> reports/unification.png + .json
"""
from __future__ import annotations
import json, time, statistics as st, warnings
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"breast_cancer": load_breast_cancer, "wine": load_wine, "digits": load_digits}
WM_PRIOR = {("logistic_regression", True): 0.965, ("logistic_regression", False): 0.94,
            ("random_forest", False): 0.96, ("gradient_boosting", False): 0.96,
            ("hist_gb", False): 0.96, ("svc", True): 0.96, ("svc", False): 0.90}
CANDIDATES = list(WM_PRIOR.keys())


def build(model, seed):
    return {"logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(random_state=seed),
            "hist_gb": HistGradientBoostingClassifier(random_state=seed),
            "svc": SVC(C=1.0, random_state=seed)}[model]


def run_candidate(dataset, model, scale, seed=0):
    X, y = DATASETS[dataset](return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = build(model, seed)
    if scale: clf = make_pipeline(StandardScaler(), clf)
    t0 = time.perf_counter(); clf.fit(Xtr, ytr); rt = time.perf_counter() - t0
    return round(accuracy_score(yte, clf.predict(Xte)), 4), rt


def spearman(a, b):
    def ranks(x):
        order = np.argsort(x); r = np.empty(len(x)); r[order] = np.arange(len(x)); return r
    ra, rb = ranks(np.array(a, float)), ranks(np.array(b, float))
    return float(np.corrcoef(ra, rb)[0, 1])


def cost_to_best(order, accs, runtimes, best):
    spent = 0.0
    for c in order:
        spent += runtimes[c]
        if accs[c] >= best - 1e-9: return spent
    return spent


def main():
    rng = np.random.default_rng(0)
    pts = {}
    for d in DATASETS:
        accs, rts = {}, {}
        for c in CANDIDATES:
            a, rt = run_candidate(d, c[0], c[1]); accs[c], rts[c] = a, rt
        preds = [WM_PRIOR[c] for c in CANDIDATES]
        reals = [accs[c] for c in CANDIDATES]
        calib = spearman(preds, reals)                      # X : le WM range-t-il bien ?
        best = max(reals)
        guided = sorted(CANDIDATES, key=lambda c: WM_PRIOR[c], reverse=True)
        gc = cost_to_best(guided, accs, rts, best)
        rc = st.mean(cost_to_best((lambda o: (rng.shuffle(o) or o))(list(CANDIDATES)), accs, rts, best) for _ in range(200))
        saving = 100 * (rc - gc) / rc if rc else 0
        pts[d] = {"calibration_spearman": round(calib, 3), "compute_saved_pct": round(saving, 1)}
        print(f"{d:13s} | calibration(Spearman)={calib:+.2f} | compute économisé={saving:+.1f}%")

    rep = ROOT / "reports"; rep.mkdir(exist_ok=True)
    (rep / "unification.json").write_text(json.dumps(pts, indent=2))
    figure(pts, rep / "unification.png")
    print("\n═══ THÈSE VISUELLE ═══")
    print("Plus le world-model est calibré (range bien les modèles), plus la guidance économise du compute.")
    print("Calibration faible → la guidance coûte. → On ne fait confiance à la guidance qu'à hauteur de la calibration.")
    return 0


def figure(pts, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [v["calibration_spearman"] for v in pts.values()]
    ys = [v["compute_saved_pct"] for v in pts.values()]
    plt.figure(figsize=(8, 6))
    plt.axhline(0, color="#888", lw=1); plt.axvline(0, color="#888", lw=1)
    for d, v in pts.items():
        c = "#46c08a" if v["compute_saved_pct"] > 0 else "#d96a6a"
        plt.scatter(v["calibration_spearman"], v["compute_saved_pct"], s=180, color=c, zorder=3, edgecolors="k")
        plt.annotate(d, (v["calibration_spearman"], v["compute_saved_pct"]),
                     textcoords="offset points", xytext=(10, 8), fontsize=11)
    # tendance
    if len(xs) >= 2:
        z = np.polyfit(xs, ys, 1); xx = np.linspace(min(xs) - .1, max(xs) + .1, 50)
        plt.plot(xx, z[0] * xx + z[1], "--", color="#5aa0e0", alpha=.7, label="tendance")
    plt.xlabel("Calibration du world-model  (Spearman prédit vs réel, range-t-il bien les modèles ?)")
    plt.ylabel("Compute économisé par la guidance (%)")
    plt.title("La guidance ne vaut que ce que vaut la calibration\n(calibré → économise ; mal calibré → coûte)")
    plt.legend(); plt.grid(alpha=.3)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 figure: {out}")


if __name__ == "__main__":
    import sys; sys.exit(main())
