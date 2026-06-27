#!/usr/bin/env python3
"""Noise sweep — le showcase du Verifier (la figure clé du paper).

Compare plusieurs méthodes sklearn à des niveaux de bruit croissants sur les labels.
À chaque niveau, le Verifier dit si le "meilleur" est fiable ou noyé dans le bruit.
Montre : un agent naïf nomme toujours un gagnant ; le nôtre s'abstient sous le bruit.

Usage :
    python noise_sweep.py --dataset breast_cancer --noise 0 0.1 0.2 0.4 --seeds 0 1 2 3
    # -> reports/noise_sweep.json + reports/noise_sweep.png
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from luckyworld.verifier import verify  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
DATASETS = {"breast_cancer": load_breast_cancer, "wine": load_wine, "digits": load_digits}
METHODS = {
    "logreg_scaled": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=s)),
    "random_forest": lambda s: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=-1),
    "svc_rbf":       lambda s: make_pipeline(StandardScaler(), SVC(C=1.0, random_state=s)),
    "hist_gb":       lambda s: HistGradientBoostingClassifier(random_state=s),
}


def flip_labels(y, frac, rng):
    """Flippe une fraction des labels (bruit contrôlé)."""
    y = y.copy()
    n = int(len(y) * frac)
    if n == 0:
        return y
    idx = rng.choice(len(y), n, replace=False)
    classes = np.unique(y)
    for i in idx:
        y[i] = rng.choice(classes[classes != y[i]])
    return y


def run(dataset, noises, seeds):
    load = DATASETS[dataset]
    X, y = load(return_X_y=True)
    levels = []
    for noise in noises:
        per_method = {m: [] for m in METHODS}
        for s in seeds:
            rng = np.random.default_rng(s)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=s, stratify=y)
            ytr_noisy = flip_labels(ytr, noise, rng)
            for m, build in METHODS.items():
                clf = build(s); clf.fit(Xtr, ytr_noisy)
                per_method[m].append(round(accuracy_score(yte, clf.predict(Xte)), 4))
        verdict = verify(per_method)
        levels.append({"noise": noise, "per_method": per_method, "verdict": verdict.to_dict()})
        print(f"noise={noise:>4}: {verdict.statement}")
    return levels


def figure(levels, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    noises = [l["noise"] for l in levels]
    methods = list(levels[0]["per_method"].keys())
    plt.figure(figsize=(8, 5))
    for m in methods:
        means = [float(np.mean(l["per_method"][m])) for l in levels]
        errs = [float(np.max(l["per_method"][m]) - np.min(l["per_method"][m])) / 2 for l in levels]
        plt.errorbar(noises, means, yerr=errs, marker="o", capsize=3, label=m)
    # marque le 1er niveau où le Verifier s'abstient
    for l in levels:
        if not l["verdict"]["trustworthy"]:
            plt.axvline(l["noise"], ls="--", color="red", alpha=.6)
            plt.text(l["noise"], plt.ylim()[0], "  Verifier s'abstient\n  (écart < bruit)",
                     color="red", va="bottom", fontsize=9)
            break
    plt.xlabel("Niveau de bruit sur les labels (fraction flippée)")
    plt.ylabel("Accuracy test (moy ± demi-étendue inter-seed)")
    plt.title("Calibrated Autoresearch : le Verifier s'abstient quand l'écart passe sous le bruit")
    plt.legend(); plt.grid(alpha=.3)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 figure: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_cancer", choices=list(DATASETS))
    ap.add_argument("--noise", nargs="+", type=float, default=[0, 0.1, 0.2, 0.4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    a = ap.parse_args()
    levels = run(a.dataset, a.noise, a.seeds)
    # bilan naïf vs Verifier
    naive_winners = len(levels)  # un agent naïf nomme un gagnant à chaque niveau
    verifier_winners = sum(1 for l in levels if l["verdict"]["trustworthy"])
    avoided = naive_winners - verifier_winners
    summary = {"dataset": a.dataset, "noise_levels": a.noise, "seeds": a.seeds, "levels": levels,
               "naive_claims": naive_winners, "verifier_claims": verifier_winners,
               "false_winners_avoided": avoided}
    rep = ROOT / "reports"; rep.mkdir(exist_ok=True)
    (rep / "noise_sweep.json").write_text(json.dumps(summary, indent=2))
    figure(levels, rep / "noise_sweep.png")
    print(f"\n✅ Naïf: {naive_winners} gagnants annoncés | Verifier: {verifier_winners} | "
          f"faux gagnants évités: {avoided}")
    print("→ reports/noise_sweep.json + .png")


if __name__ == "__main__":
    main()
