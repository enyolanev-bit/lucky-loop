"""Verifier — couche de confiance #2 : gating statistique déterministe.

Complète le world-model de Hicham (couche #1 : prédiction-vs-réalité).
Principe : un finding ("méthode X est la meilleure") n'est digne de confiance que si
la borne basse de l'IC95 apparié best-vs-2e dépasse 0. Sinon → "inconclusive".
La vérité vient des chiffres, pas du LLM.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import statistics as st


@dataclass
class Verdict:
    best_method: str
    best_acc: float
    second_method: str | None
    effect_size: float      # best_mean - second_mean
    seed_noise: float       # marge IC95 apparié sur l'écart best-vs-2e
    trustworthy: bool       # borne basse IC95(best-2e) > 0 ?
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
    paired = list(zip(per_method_accs[best_m], per_method_accs.get(second_m, []))) if second_m else []
    if len(paired) < 2:
        noise = 0.0
        trust = False
        ci_low = effect
    else:
        diffs = [a - b for a, b in paired]
        stderr = st.stdev(diffs) / math.sqrt(len(diffs))
        t95 = [
            12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
            2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
            2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
        ][min(len(diffs) - 2, 29)]
        margin = t95 * stderr
        noise = round(margin, 4)
        ci_low = st.mean(diffs) - margin
        trust = ci_low > 0
    if trust:
        stmt = (f"'{best_m}' est le meilleur : écart {effect:.4f}, IC95 apparié "
                f"[{ci_low:.4f}, {st.mean(diffs) + margin:.4f}] — finding fiable.")
    else:
        stmt = (f"AUCUN gagnant fiable : écart best-2e {effect:.4f}, marge IC95 apparié {noise:.4f} "
                f"(borne basse {ci_low:.4f}) "
                f"→ inconclusive (within noise).")
    return Verdict(best_m, round(best_mean, 4), second_m, effect, noise, trust, stmt)
