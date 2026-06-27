"""Verifier — couche de confiance #2 : gating effet-vs-bruit (déterministe, un-foolable).

Complète le world-model de Hicham (couche #1 : prédiction-vs-réalité).
Principe : un finding ("méthode X est la meilleure") n'est digne de confiance que si
l'écart best vs 2e dépasse le bruit inter-seed. Sinon → "inconclusive (within noise)".
La vérité vient des chiffres, pas du LLM.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import statistics as st


@dataclass
class Verdict:
    best_method: str
    best_acc: float
    second_method: str | None
    effect_size: float      # best_mean - second_mean
    seed_noise: float       # bruit inter-seed sur le best (max-min)
    trustworthy: bool       # effect_size > seed_noise ?
    statement: str

    def to_dict(self):
        return asdict(self)


def verify(per_method_accs: dict[str, list[float]]) -> Verdict:
    """per_method_accs: {method_name: [acc_seed0, acc_seed1, ...]} -> Verdict.

    Distingue un vrai gagnant d'un écart noyé dans le bruit.
    """
    means = {m: st.mean(a) for m, a in per_method_accs.items() if a}
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    best_m, best_mean = ranked[0]
    second_m, second_mean = (ranked[1] if len(ranked) > 1 else (None, best_mean))
    effect = round(best_mean - second_mean, 4)
    best_accs = per_method_accs[best_m]
    noise = round((max(best_accs) - min(best_accs)) if len(best_accs) > 1 else 0.0, 4)
    trust = effect > noise
    if trust:
        stmt = f"'{best_m}' est le meilleur (écart {effect:.4f} > bruit {noise:.4f}) — finding fiable."
    else:
        stmt = (f"AUCUN gagnant fiable : écart best-2e {effect:.4f} ≤ bruit inter-seed {noise:.4f} "
                f"→ inconclusive (within noise).")
    return Verdict(best_m, round(best_mean, 4), second_m, effect, noise, trust, stmt)
