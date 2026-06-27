from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .schemas import Verification


def verify_sweep(sweep: dict) -> Verification:
    """Verify whether a sweep supports a claim beyond inter-seed noise.

    The verifier is intentionally deterministic: the truth comes from measured
    metrics, not from an LLM-written explanation.
    """
    runs = sweep.get("runs") or []
    metric = sweep.get("metric", "accuracy")
    if not runs:
        return Verification(
            status="missing_data",
            metric=metric,
            effect_size=None,
            seed_noise=None,
            trustworthy=False,
            supported_claims=[],
            inconclusive_findings=["No sweep runs were available for verification."],
            rationale="Verifier could not compute effect-vs-noise without repeated runs.",
        )

    by_config: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in runs:
        key = str(row.get("config_key", row.get("value", row.get("model", "unknown"))))
        labels[key] = str(row.get("label", key))
        value = row.get(metric)
        if value is None:
            continue
        by_config[key].append(float(value))

    if len(by_config) < 2:
        return Verification(
            status="missing_data",
            metric=metric,
            effect_size=None,
            seed_noise=None,
            trustworthy=False,
            supported_claims=[],
            inconclusive_findings=["Verifier needs at least two configurations to estimate an effect size."],
            rationale="Only one configuration had metric data.",
        )

    means = {k: mean(v) for k, v in by_config.items() if v}
    if len(means) < 2:
        return Verification(
            status="missing_data",
            metric=metric,
            effect_size=None,
            seed_noise=None,
            trustworthy=False,
            supported_claims=[],
            inconclusive_findings=["Not enough numeric metric values were available."],
            rationale="Metric extraction failed for most sweep runs.",
        )

    best_key = max(means, key=means.get)
    worst_key = min(means, key=means.get)
    effect_size = round(means[best_key] - means[worst_key], 6)
    best_values = by_config[best_key]
    seed_noise = round(max(best_values) - min(best_values), 6) if len(best_values) > 1 else None

    supported: list[str] = []
    inconclusive: list[str] = []
    trustworthy = bool(seed_noise is not None and effect_size > seed_noise)
    status = "supported" if trustworthy else "inconclusive"
    best_label = labels.get(best_key, best_key)
    worst_label = labels.get(worst_key, worst_key)

    if trustworthy:
        supported.append(
            f"Best config '{best_label}' beats worst config '{worst_label}' by {effect_size:.4f} {metric}, above seed noise {seed_noise:.4f}."
        )
        rationale = "Measured effect is larger than inter-seed noise for the best config. The finding is supported, but still bounded by the small benchmark."
    else:
        if seed_noise is None:
            inconclusive.append("Only one seed was run for the best config, so seed noise is unknown.")
            rationale = "A claim cannot be verified without repeated seeds."
        else:
            inconclusive.append(
                f"Best config '{best_label}' improves over '{worst_label}' by {effect_size:.4f} {metric}, but best-config seed noise is {seed_noise:.4f}."
            )
            rationale = "Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery."

    return Verification(
        status=status,
        metric=metric,
        effect_size=effect_size,
        seed_noise=seed_noise,
        trustworthy=trustworthy,
        best_config=best_label,
        supported_claims=supported,
        inconclusive_findings=inconclusive,
        rationale=rationale,
    )
