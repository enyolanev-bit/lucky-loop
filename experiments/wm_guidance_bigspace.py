#!/usr/bin/env python3
"""World-model guidance on a LARGE candidate space — does ordering by Qwen's predictions save compute?

The controlled lab ablation was NULL because tasks had ~4 candidates (the right choice was obvious).
Here we give a BIG candidate space (varied cost + accuracy) so guidance can matter:
  - ON  : order candidates by the REAL Qwen-AgentWorld predicted accuracy (desc) -> cost-to-reach-best.
  - OFF : random order (mean over shuffles) -> cost-to-reach-best (no world model).
  - saving = (off_cost - on_cost) / off_cost. Positive = world model saved compute.
Per dataset we also report Qwen calibration (Spearman predicted vs actual) to relate it to the saving.

Real Qwen via the box endpoint; real sklearn timing. No fabrication.
    LLM_BASE_URL=http://YOUR_SIMULATOR_HOST:8000/v1 python experiments/wm_guidance_bigspace.py
"""
from __future__ import annotations
import json, os, time, statistics as st, urllib.request, warnings
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
BASE = os.environ.get("LLM_BASE_URL", "http://YOUR_SIMULATOR_HOST:8000/v1")
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen-AgentWorld-35B-A3B")
KEY = os.environ.get("OPENAI_API_KEY", "dummy")
ROOT = Path(__file__).resolve().parents[1]

DATASETS = {"breast_cancer": load_breast_cancer, "wine": load_wine, "digits": load_digits, "iris": load_iris}
def synth(seed, sep): return lambda: make_classification(n_samples=900, n_features=20, n_informative=10,
                                                         n_redundant=0, n_classes=2, class_sep=sep, random_state=seed)
DATASETS_EXTRA = {"synth_easy": synth(1, 2.0), "synth_hard": synth(2, 0.6)}

# ~16 candidats : coût ET accuracy variés
CANDS = []
for C in [0.01, 0.1, 1.0, 10.0]:
    for sc in [True, False]:
        CANDS.append(("logistic_regression", {"C": C, "scale": sc}))
for n in [50, 150, 400]:
    CANDS.append(("random_forest", {"n_estimators": n}))
for n in [50, 150] :
    CANDS.append(("gradient_boosting", {"n_estimators": n}))
CANDS.append(("hist_gb", {}))
for C in [1.0, 10.0]:
    CANDS.append(("svc", {"C": C, "scale": True}))


def build(model, p):
    if model == "logistic_regression":
        clf = LogisticRegression(C=p.get("C", 1.0), max_iter=1000, random_state=0)
        return make_pipeline(StandardScaler(), clf) if p.get("scale") else clf
    if model == "random_forest": return RandomForestClassifier(n_estimators=p.get("n_estimators", 300), random_state=0, n_jobs=-1)
    if model == "gradient_boosting": return GradientBoostingClassifier(n_estimators=p.get("n_estimators", 150), random_state=0)
    if model == "hist_gb": return HistGradientBoostingClassifier(random_state=0)
    if model == "svc":
        return make_pipeline(StandardScaler(), SVC(C=p.get("C", 1.0), random_state=0))


def run_cand(X, y, model, p):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = build(model, p); t0 = time.perf_counter(); clf.fit(Xtr, ytr); rt = time.perf_counter() - t0
    return round(accuracy_score(yte, clf.predict(Xte)), 4), rt


def qwen_pred(desc, model, p):
    sysmsg = "You are Qwen-AgentWorld, predicting ML experiment test accuracy before compute. Answer ONLY JSON."
    user = (f"Predict TEST accuracy (0-1) for: {desc}, model={model}, params={p}, 25% split.\n"
            'JSON: {"point": <float 0-1>}')
    body = {"model": MODEL, "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
            "temperature": 0.2, "max_tokens": 600, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            m = json.loads(r.read())["choices"][0]["message"]
        txt = (m.get("content") or m.get("reasoning_content") or "").strip()
        i, j = txt.find("{"), txt.rfind("}")
        return float(json.loads(txt[i:j + 1]).get("point"))
    except Exception as e:
        print("  qwen err:", str(e)[:60], flush=True); return None


def spearman(a, b):
    rk = lambda v: np.argsort(np.argsort(v)); return float(np.corrcoef(rk(np.array(a, float)), rk(np.array(b, float)))[0, 1])


def cost_to_best(order, accs, rts, best):
    s = 0.0
    for c in order:
        s += rts[c]
        if accs[c] >= best - 1e-9: return s
    return s


def main():
    rng = np.random.default_rng(0)
    allds = {**{k: (lambda L=v: L(return_X_y=True)) for k, v in DATASETS.items()}, **DATASETS_EXTRA}
    rows = {}
    for name, loader in allds.items():
        X, y = loader()
        accs, rts, preds = {}, {}, {}
        for i, (m, p) in enumerate(CANDS):
            a, rt = run_cand(X, y, m, p); accs[i], rts[i] = a, rt
            q = qwen_pred(f"the '{name}' dataset ({X.shape[0]}x{X.shape[1]}, {len(set(y))} classes)", m, p)
            preds[i] = q if q is not None else 0.9
        best = max(accs.values())
        calib = spearman([preds[i] for i in range(len(CANDS))], [accs[i] for i in range(len(CANDS))])
        # ON variants
        guided = sorted(range(len(CANDS)), key=lambda i: preds[i], reverse=True)            # accuracy-only
        guided_cost = sorted(range(len(CANDS)), key=lambda i: preds[i] / max(rts[i], 1e-4), reverse=True)  # acc-per-cost
        on_cost = cost_to_best(guided, accs, rts, best)
        on_cost_costaware = cost_to_best(guided_cost, accs, rts, best)
        # OFF baselines
        off_random = st.mean(cost_to_best(list(rng.permutation(len(CANDS))), accs, rts, best) for _ in range(300))
        off_cheapest = cost_to_best(sorted(range(len(CANDS)), key=lambda i: rts[i]), accs, rts, best)  # cheapest-first heuristic
        sv = lambda base, on: round(100 * (base - on) / base, 1) if base else 0.0
        rows[name] = {"calibration": round(calib, 3),
                      "on_cost_s": round(on_cost, 4), "on_cost_costaware_s": round(on_cost_costaware, 4),
                      "off_random_s": round(off_random, 4), "off_cheapest_s": round(off_cheapest, 4),
                      "saved_vs_random_pct": sv(off_random, on_cost),
                      "saved_vs_cheapest_pct": sv(off_cheapest, on_cost),
                      "costaware_saved_vs_cheapest_pct": sv(off_cheapest, on_cost_costaware),
                      "n_candidates": len(CANDS)}
        print(f"{name:14s} | calib {calib:+.2f} | vs_random {sv(off_random, on_cost):+6.1f}% | "
              f"vs_cheapest {sv(off_cheapest, on_cost):+6.1f}% | costaware_vs_cheapest {sv(off_cheapest, on_cost_costaware):+6.1f}%", flush=True)
    real = [k for k in rows if not k.startswith("synth")]
    avg = lambda key, keys: round(st.mean(rows[k][key] for k in keys), 1)
    out = {"model": MODEL, "n_candidates": len(CANDS), "by_dataset": rows,
           "avg_saved_vs_random_all": avg("saved_vs_random_pct", rows),
           "avg_saved_vs_cheapest_all": avg("saved_vs_cheapest_pct", rows),
           "avg_saved_vs_cheapest_real_only": avg("saved_vs_cheapest_pct", real),
           "avg_costaware_saved_vs_cheapest_all": avg("costaware_saved_vs_cheapest_pct", rows)}
    rep = ROOT / "reports"; rep.mkdir(exist_ok=True)
    (rep / "wm_guidance_bigspace.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== BIG-SPACE GUIDANCE ({len(CANDS)} candidats) ===")
    print(f"accuracy-only ranking: vs random {out['avg_saved_vs_random_all']:+.1f}% | "
          f"vs cheapest-first {out['avg_saved_vs_cheapest_all']:+.1f}% (real-only {out['avg_saved_vs_cheapest_real_only']:+.1f}%)")
    print(f"cost-aware ranking: vs cheapest-first {out['avg_costaware_saved_vs_cheapest_all']:+.1f}%")
    print("reports/wm_guidance_bigspace.json")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
