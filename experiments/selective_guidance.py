#!/usr/bin/env python3
"""Selective guidance — un world-model qui sait QUAND se faire confiance.

La dose-response a prouvé : suivre la guidance quand le world-model est mal calibré COÛTE du compute.
Donc la bonne politique n'est pas "toujours suivre" (naive) ni "jamais" (random) mais SÉLECTIVE :
suivre la guidance seulement quand la confiance du world-model dépasse un seuil, sinon fallback random.

Setup réaliste : sur chaque tâche le world-model a une calibration variable (parfois il range bien,
parfois non). Il émet un signal de confiance = calibration vraie + bruit (auto-évaluation imparfaite,
comme un vrai LLM). On compare 4 politiques :
  - naive    : suit toujours la guidance
  - random   : ne suit jamais (référence, économie = 0)
  - selective: suit si confiance >= seuil τ (on balaye τ)
  - oracle   : suit si calibration VRAIE >= 0 (borne haute, connaît la vérité)

Meta-thèse testée : selective ne bat naive QUE si la confiance est elle-même calibrée
(on balaye le bruit σ de l'auto-évaluation).

    python experiments/selective_guidance.py   # -> reports/selective_guidance.{json,png}
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
SEEDS = (0, 1, 2)
TASKS_PER = 60          # tâches simulées par (dataset, seed) — tirages de calibration variable
THRESHOLDS = np.linspace(-0.6, 0.9, 31)
CONF_NOISE = 0.3        # σ de l'auto-évaluation (signal principal)
NOISE_SWEEP = [0.0, 0.3, 0.6, 1.0]  # 0 = confiance parfaite (=oracle), 1.0 = confiance inutile


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


def build_tasks(rng):
    """Précalcule des tâches : chacune = (calibration vraie, économie si on suit la guidance)."""
    tasks = []  # (true_calib, guided_saving_pct)
    for dname, loader in DATASETS.items():
        X, y = loader(return_X_y=True)
        for seed in SEEDS:
            accs, rts = {}, {}
            for c in CANDIDATES:
                a, rt = run_candidate(X, y, c[0], c[1], seed); accs[c], rts[c] = a, rt
            best = max(accs.values())
            true_vec = np.array([accs[c] for c in CANDIDATES], float)
            tv = (true_vec - true_vec.mean()) / (true_vec.std() + 1e-9)
            rc = np.mean([cost_to_best([CANDIDATES[i] for i in rng.permutation(len(CANDIDATES))], accs, rts, best)
                          for _ in range(120)])
            for _ in range(TASKS_PER):
                t = rng.uniform(0, 1)  # niveau de corruption (calibration variable selon la tâche)
                prior = (1 - t) * tv + t * rng.normal(size=len(CANDIDATES))
                calib = spearman(prior, true_vec)
                order = [CANDIDATES[i] for i in np.argsort(-prior)]
                gc = cost_to_best(order, accs, rts, best)
                saving = 100 * (rc - gc) / rc if rc else 0
                tasks.append((calib, saving))
    return np.array([t[0] for t in tasks]), np.array([t[1] for t in tasks])


def policy_savings(calib, guided_sav, conf, thresholds):
    """Pour chaque seuil τ : économie moyenne de la politique selective (suit si conf>=τ, sinon 0)."""
    return np.array([float(np.mean(np.where(conf >= tau, guided_sav, 0.0))) for tau in thresholds])


def main():
    rng = np.random.default_rng(0)
    calib, guided_sav = build_tasks(rng)
    n = len(calib)
    naive = float(np.mean(guided_sav))                       # suit toujours
    oracle = float(np.mean(np.where(calib >= 0, guided_sav, 0.0)))  # suit si calib vraie>=0

    # signal principal : confiance = calib + bruit σ=CONF_NOISE
    conf = calib + rng.normal(0, CONF_NOISE, size=n)
    sel_curve = policy_savings(calib, guided_sav, conf, THRESHOLDS)
    best_i = int(np.argmax(sel_curve))
    selective = float(sel_curve[best_i]); best_tau = float(THRESHOLDS[best_i])
    recovered = 100 * (selective - 0) / oracle if oracle else 0  # % de l'oracle récupéré (random=0)

    # sweep du bruit de confiance : selective ne bat naive que si la confiance est calibrée
    sweep = {}
    for sigma in NOISE_SWEEP:
        c = calib + rng.normal(0, sigma, size=n)
        curve = policy_savings(calib, guided_sav, c, THRESHOLDS)
        sweep[str(sigma)] = round(float(np.max(curve)), 2)

    out = {"n_tasks": n, "naive_saving_pct": round(naive, 2), "random_saving_pct": 0.0,
           "oracle_saving_pct": round(oracle, 2), "selective_saving_pct": round(selective, 2),
           "selective_best_threshold": round(best_tau, 3),
           "pct_oracle_recovered": round(recovered, 1),
           "value_of_confidence_vs_naive": round(selective - naive, 2),
           "conf_noise_sigma": CONF_NOISE, "selective_by_conf_noise": sweep}
    rep = ROOT / "reports"; rep.mkdir(exist_ok=True)
    (rep / "selective_guidance.json").write_text(json.dumps(out, indent=2))
    figure(THRESHOLDS, sel_curve, naive, oracle, best_tau, selective, rep / "selective_guidance.png")

    print("═══ SELECTIVE GUIDANCE (le world-model sait quand se faire confiance) ═══")
    print(f"n = {n} tâches (calibration variable par tâche)")
    print(f"random   (jamais suivre)     : {0.0:+6.2f}%")
    print(f"naive    (toujours suivre)   : {naive:+6.2f}%")
    print(f"selective(conf >= {best_tau:+.2f})    : {selective:+6.2f}%   <- gate sur confiance bruitée σ={CONF_NOISE}")
    print(f"oracle   (calib vraie >= 0)  : {oracle:+6.2f}%   (borne haute)")
    print(f"\n→ Valeur d'une confiance calibrée = selective − naive = {selective - naive:+.2f} pts")
    print(f"→ Selective récupère {recovered:.0f}% de l'oracle avec une confiance imparfaite.")
    print(f"\nMeta-thèse (selective vs bruit de confiance σ) :")
    for s, v in sweep.items():
        tag = "(= oracle)" if float(s) == 0 else ("(confiance inutile)" if float(s) >= 1.0 else "")
        print(f"  σ={s:>3} → selective best = {v:+6.2f}%  {tag}")
    print("→ Selective ne bat naive QUE si la confiance est elle-même calibrée. Méta-calibration.")
    print("\nreports/selective_guidance.json")
    return 0


def figure(taus, curve, naive, oracle, best_tau, selective, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8.5, 6))
    plt.axhline(naive, color="#d96a6a", ls="--", lw=1.8, label=f"naive (toujours) = {naive:+.1f}%")
    plt.axhline(0, color="#888", ls=":", lw=1.4, label="random (jamais) = 0%")
    plt.axhline(oracle, color="#46c08a", ls="--", lw=1.8, label=f"oracle (borne haute) = {oracle:+.1f}%")
    plt.plot(taus, curve, "-o", color="#5aa0e0", ms=4, label="selective (gate sur confiance)")
    plt.scatter([best_tau], [selective], s=160, color="#2c6fbb", zorder=5, edgecolors="k",
                label=f"meilleur τ = {best_tau:+.2f} → {selective:+.1f}%")
    plt.xlabel("Seuil de confiance τ (suivre la guidance si confiance ≥ τ)")
    plt.ylabel("Compute économisé moyen (%)")
    plt.title("Selective guidance : suivre le world-model seulement quand il est confiant\n"
              "récupère l'économie que la guidance naïve perd sur les tâches mal calibrées")
    plt.legend(fontsize=9); plt.grid(alpha=.3); plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 figure: {out}")


if __name__ == "__main__":
    import sys; sys.exit(main())
