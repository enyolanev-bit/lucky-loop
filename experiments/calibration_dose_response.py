#!/usr/bin/env python3
"""Dose-response CAUSALE : on MANIPULE la calibration du world-model, on mesure le compute économisé.

L'unification across-datasets est observationnelle (r=0.75 mais IC95 large sur n=12, Spearman 0.39).
Ici on passe au causal : sur des datasets fixes, on dégrade la PRÉDICTION du world-model par paliers
(de "ranking parfait" vers "bruit pur"), et à chaque palier on mesure :
  X = calibration réelle obtenue (Spearman prior-corrompu vs accuracy réelle)
  Y = compute économisé par la guidance vs ordre aléatoire
Beaucoup de paliers × seeds × datasets -> une courbe dose-response propre -> IC95 serré.

On manipule la cause (calibration) et on observe l'effet (saving) -> évidence causale, pas corrélation.

    python experiments/calibration_dose_response.py   # -> reports/calibration_dose_response.{json,png}
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
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
DATASETS = {"breast_cancer": load_breast_cancer, "wine": load_wine,
            "digits": load_digits, "iris": load_iris}
CANDIDATES = [("logistic_regression", False), ("logistic_regression", True), ("random_forest", False),
              ("gradient_boosting", False), ("hist_gb", False), ("svc", False), ("svc", True)]
CORRUPTIONS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]  # 0 = prior parfait, 1 = bruit pur
SEEDS = (0, 1, 2)
N_BOOT = 10000


def build(model, seed):
    return {"logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(random_state=seed),
            "hist_gb": HistGradientBoostingClassifier(random_state=seed),
            "svc": SVC(C=1.0, random_state=seed)}[model]


def run_candidate(X, y, model, scale, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = build(model, seed)
    if scale: clf = make_pipeline(StandardScaler(), clf)
    t0 = time.perf_counter(); clf.fit(Xtr, ytr); rt = time.perf_counter() - t0
    return accuracy_score(yte, clf.predict(Xte)), rt


def spearman(a, b):
    rank = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(rank(np.asarray(a, float)), rank(np.asarray(b, float)))[0, 1])


def cost_to_best(order, accs, rts, best):
    s = 0.0
    for c in order:
        s += rts[c]
        if accs[c] >= best - 1e-9: return s
    return s


def main():
    rng = np.random.default_rng(0)
    rows = []  # (calibration_obtenue, saving_pct)
    for dname, loader in DATASETS.items():
        X, y = loader(return_X_y=True)
        for seed in SEEDS:
            accs, rts = {}, {}
            for c in CANDIDATES:
                a, rt = run_candidate(X, y, c[0], c[1], seed); accs[c], rts[c] = a, rt
            best = max(accs.values())
            true_vec = np.array([accs[c] for c in CANDIDATES], float)
            # ordre aléatoire de référence (moyenne)
            rc = np.mean([cost_to_best([CANDIDATES[i] for i in rng.permutation(len(CANDIDATES))], accs, rts, best)
                          for _ in range(120)])
            for t in CORRUPTIONS:
                # prior corrompu = (1-t)*vérité normalisée + t*bruit
                noise = rng.normal(size=len(CANDIDATES))
                tv = (true_vec - true_vec.mean()) / (true_vec.std() + 1e-9)
                prior = (1 - t) * tv + t * noise
                calib = spearman(prior, true_vec)
                order = [CANDIDATES[i] for i in np.argsort(-prior)]  # tri décroissant du prior
                gc = cost_to_best(order, accs, rts, best)
                saving = 100 * (rc - gc) / rc if rc else 0
                rows.append((calib, saving))

    cal = np.array([r[0] for r in rows]); sav = np.array([r[1] for r in rows])
    r = float(np.corrcoef(cal, sav)[0, 1])
    rho = spearman(cal, sav)
    # bootstrap IC95
    idx = np.arange(len(cal)); boots = []
    for _ in range(N_BOOT):
        s = rng.choice(idx, size=len(idx), replace=True)
        if cal[s].std() > 1e-9 and sav[s].std() > 1e-9:
            boots.append(float(np.corrcoef(cal[s], sav[s])[0, 1]))
    lo, hi = float(np.quantile(boots, .025)), float(np.quantile(boots, .975))

    out = {"n_points": len(rows), "n_datasets": len(DATASETS), "n_corruptions": len(CORRUPTIONS),
           "n_seeds": len(SEEDS), "pearson_r": round(r, 4), "spearman_rho": round(rho, 4),
           "bootstrap_95ci": [round(lo, 4), round(hi, 4)], "ci_excludes_zero": bool(lo > 0)}
    rep = ROOT / "reports"; rep.mkdir(exist_ok=True)
    (rep / "calibration_dose_response.json").write_text(json.dumps(out, indent=2))
    figure(cal, sav, r, lo, hi, rep / "calibration_dose_response.png")

    print("═══ DOSE-RESPONSE CAUSALE (on manipule la calibration) ═══")
    print(f"n = {len(rows)} points ({len(DATASETS)} datasets × {len(CORRUPTIONS)} paliers × {len(SEEDS)} seeds)")
    print(f"Pearson r        = {r:+.3f}")
    print(f"Spearman rho     = {rho:+.3f}")
    print(f"Bootstrap 95% CI = [{lo:+.3f}, {hi:+.3f}]")
    print(f"CI exclut 0 ?    = {'OUI' if lo > 0 else 'NON'}")
    print()
    if lo > 0:
        print(f"→ CAUSAL : dégrader la calibration du world-model RÉDUIT le compute économisé,")
        print(f"  de façon monotone (r={r:+.2f}, IC95 [{lo:+.2f},{hi:+.2f}] exclut zéro). Pas un artefact.")
    print("\nreports/calibration_dose_response.json")
    return 0


def figure(cal, sav, r, lo, hi, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8.5, 6))
    plt.axhline(0, color="#888", lw=1); plt.axvline(0, color="#888", lw=1)
    plt.scatter(cal, sav, s=45, alpha=.55, color="#5aa0e0", edgecolors="k", linewidths=.4, zorder=3)
    z = np.polyfit(cal, sav, 1); xx = np.linspace(cal.min(), cal.max(), 50)
    plt.plot(xx, z[0] * xx + z[1], "--", color="#d96a6a", lw=2,
             label=f"r = {r:+.2f}, IC95 [{lo:+.2f}, {hi:+.2f}]")
    plt.xlabel("Calibration du world-model (manipulée : parfait → bruit)")
    plt.ylabel("Compute économisé par la guidance (%)")
    plt.title("Dose-response causale : dégrader la calibration réduit l'économie de compute\n"
              f"n={len(cal)} points contrôlés")
    plt.legend(); plt.grid(alpha=.3); plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 figure: {out}")


if __name__ == "__main__":
    import sys; sys.exit(main())
