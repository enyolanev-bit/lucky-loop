"""Sandbox d'expériences — le terrain de jeu de l'agent (la 'ruée vers l'hyperparamètre').

Entraîne un petit MLP sur un dataset jouet (sklearn digits) en variant UN hyperparamètre.
Rapide (CPU ou GPU MI300X), résultats RÉELS, reproductible (seed). C'est ce que
l'Experimenter agent appelle pour produire de vrais chiffres, et que le Verifier re-checke.

Usage direct (test) :
    python sandbox.py --hp weight_decay --values 0 1e-4 1e-3 1e-2 --seeds 0 1
"""
from __future__ import annotations
import argparse, json, time
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

DEV = "cuda" if torch.cuda.is_available() else "cpu"   # 'cuda' = ROCm sur MI300X


def _data(seed):
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    f = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    g = lambda a: torch.tensor(a, dtype=torch.long, device=DEV)
    return f(Xtr) / 16.0, g(ytr), f(Xte) / 16.0, g(yte)


def run_one(hp: str, value: float, seed: int = 0, epochs: int = 80, hidden: int = 64) -> float:
    """Entraîne 1 MLP avec hp=value, renvoie l'accuracy test (réelle)."""
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = _data(seed)
    dropout = value if hp == "dropout" else 0.0
    wd = value if hp == "weight_decay" else 0.0
    lr = value if hp == "lr" else 1e-2
    hidden = int(value) if hp == "hidden" else hidden
    net = nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(hidden, 10)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        lossf(net(Xtr), ytr).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(Xte).argmax(1) == yte).float().mean().item()
    return round(acc, 4)


def sweep(hp: str, values: list[float], seeds: list[int]) -> dict:
    """Sweep complet -> résultats réels (moyenne sur seeds). Renvoie un dict JSON-able."""
    t0 = time.time()
    rows = []
    for v in values:
        accs = [run_one(hp, v, s) for s in seeds]
        rows.append({"value": v, "acc_mean": round(sum(accs) / len(accs), 4),
                     "acc_per_seed": accs})
    best = max(rows, key=lambda r: r["acc_mean"])
    return {"hp": hp, "device": DEV, "seeds": seeds, "results": rows,
            "best": best, "elapsed_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hp", default="weight_decay", choices=["weight_decay", "dropout", "lr", "hidden"])
    p.add_argument("--values", nargs="+", type=float, default=[0, 1e-4, 1e-3, 1e-2])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    a = p.parse_args()
    print(json.dumps(sweep(a.hp, a.values, a.seeds), indent=2))
