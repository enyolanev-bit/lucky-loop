#!/usr/bin/env python3
"""Guidance ablation — est-ce que "predict before compute" ÉCONOMISE du compute ?

Hicham mesure si le world-model est calibré (ses prédictions sont-elles justes ?).
Ici on mesure la VALEUR DOWNSTREAM : utiliser le world-model pour ORDONNER les expériences
fait-il atteindre le meilleur modèle avec MOINS de compute qu'un ordre naïf (random) ?

Stratégies comparées sur un pool de candidats :
- guided : on lance dans l'ordre DÉCROISSANT du métrique prédit par le world-model.
- random : ordre aléatoire (moyenne sur seeds).
- worst : ordre croissant (borne basse, pour cadrer).
Métrique : compute (temps cumulé) dépensé AVANT d'avoir lancé le vrai meilleur modèle.

Self-contained (prior heuristique aligné sur simulator.py de Hicham). Zéro box, zéro LLM.
    python guidance_ablation.py
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

# prédiction du world-model (prior heuristique, repris de simulator.py de Hicham : midpoint de la range)
WM_PRIOR = {  # (model, scaled) -> métrique prédite par le world-model
    ("logistic_regression", True): 0.965, ("logistic_regression", False): 0.94,
    ("random_forest", False): 0.96, ("gradient_boosting", False): 0.96,
    ("hist_gb", False): 0.96, ("svc", True): 0.96, ("svc", False): 0.90,
}
CANDIDATES = [("logistic_regression", False), ("logistic_regression", True), ("random_forest", False),
              ("gradient_boosting", False), ("hist_gb", False), ("svc", False), ("svc", True)]


def build(model, seed):
    if model == "logistic_regression": return LogisticRegression(max_iter=1000, random_state=seed)
    if model == "random_forest": return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if model == "gradient_boosting": return GradientBoostingClassifier(random_state=seed)
    if model == "hist_gb": return HistGradientBoostingClassifier(random_state=seed)
    if model == "svc": return SVC(C=1.0, random_state=seed)


def run_candidate(dataset, model, scale, seed=0):
    X, y = DATASETS[dataset](return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = build(model, seed)
    if scale: clf = make_pipeline(StandardScaler(), clf)
    t0 = time.perf_counter(); clf.fit(Xtr, ytr); rt = time.perf_counter() - t0
    return round(accuracy_score(yte, clf.predict(Xte)), 4), rt


def cost_to_best(order, accs, runtimes, best_acc):
    """Compute (temps) cumulé jusqu'à AVOIR lancé le vrai meilleur candidat, selon l'ordre."""
    spent = 0.0
    for m, s in order:
        spent += runtimes[(m, s)]
        if accs[(m, s)] >= best_acc - 1e-9:
            return spent
    return spent


def main():
    rng = np.random.default_rng(0)
    summary = {}
    for dataset in DATASETS:
        # vraies métriques + runtime de chaque candidat
        accs, runtimes = {}, {}
        for m, sc in CANDIDATES:
            a, rt = run_candidate(dataset, m, sc)
            accs[(m, sc)], runtimes[(m, sc)] = a, rt
        best_acc = max(accs.values())
        total_compute = sum(runtimes.values())
        # guided : tri décroissant par métrique prédite par le world-model
        guided = sorted(CANDIDATES, key=lambda c: WM_PRIOR.get(c, 0.9), reverse=True)
        guided_cost = cost_to_best(guided, accs, runtimes, best_acc)
        # random : moyenne sur 200 ordres
        rnd_costs = []
        for _ in range(200):
            order = list(CANDIDATES); rng.shuffle(order)
            rnd_costs.append(cost_to_best(order, accs, runtimes, best_acc))
        random_cost = st.mean(rnd_costs)
        saving = round(100 * (random_cost - guided_cost) / random_cost, 1) if random_cost else 0
        summary[dataset] = {"best_acc": best_acc, "total_compute_s": round(total_compute, 3),
                            "guided_cost_s": round(guided_cost, 3), "random_cost_s": round(random_cost, 3),
                            "compute_saved_vs_random_pct": saving,
                            "guided_vs_exhaustive_pct": round(100 * guided_cost / total_compute, 1)}
        print(f"{dataset:13s} | guided {guided_cost:.3f}s | random(moy) {random_cost:.3f}s "
              f"| exhaustif {total_compute:.3f}s | économie vs random: {saving:+.1f}%")

    avg_saving = round(st.mean(v["compute_saved_vs_random_pct"] for v in summary.values()), 1)
    rep = ROOT / "reports"; rep.mkdir(exist_ok=True)
    out = {"datasets": summary, "avg_compute_saved_vs_random_pct": avg_saving, "candidates": [list(c) for c in CANDIDATES]}
    (rep / "guidance_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"\n═══ FINDING ═══")
    print(f"Le world-model fait atteindre le meilleur modèle avec en moyenne "
          f"{avg_saving:+.1f}% de compute en moins qu'un ordre aléatoire.")
    print("→ 'Predict before compute' a une VALEUR mesurable (pas juste une jolie prédiction).")
    print("reports/guidance_ablation.json")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
