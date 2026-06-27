from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .schemas import Verification


def ladder_status(effect_size: float | None, seed_noise: float | None) -> tuple[str, float | None]:
    if effect_size is None or seed_noise is None or seed_noise <= 0:
        return "missing_data", None
    ratio = round(effect_size / seed_noise, 6)
    if ratio < 1.0:
        return "inconclusive", ratio
    if ratio < 2.0:
        return "weakly_supported", ratio
    if ratio < 3.0:
        return "supported", ratio
    return "strongly_supported", ratio


def verify_sweep(sweep: dict) -> Verification:
    """Verify whether a sweep supports a claim beyond inter-seed noise.

    The verifier is intentionally deterministic: the truth comes from measured
    metrics, not from an LLM-written explanation.
    """
    runs = sweep.get("runs") or []
    metric = sweep.get("metric", "accuracy")
    protocol_warning = sweep.get("protocol_warning")
    verification_type = sweep.get("type")
    if not runs:
        return Verification(
            status="missing_data",
            metric=metric,
            effect_size=None,
            seed_noise=None,
            effect_to_noise_ratio=None,
            min_seed_count=0,
            low_n_warning=True,
            trustworthy=False,
            blocked_claim="No sweep-level claim can be made without repeated runs.",
            allowed_claim="No sweep runs were available for verification.",
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
            effect_to_noise_ratio=None,
            min_seed_count=min((len(v) for v in by_config.values()), default=0),
            low_n_warning=True,
            trustworthy=False,
            blocked_claim="A configuration winner cannot be claimed with fewer than two configurations.",
            allowed_claim="Verifier needs at least two configurations to estimate an effect size.",
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
            effect_to_noise_ratio=None,
            min_seed_count=min((len(v) for v in by_config.values()), default=0),
            low_n_warning=True,
            trustworthy=False,
            blocked_claim="A sweep claim cannot be made without numeric metric values.",
            allowed_claim="Not enough numeric metric values were available.",
            supported_claims=[],
            inconclusive_findings=["Not enough numeric metric values were available."],
            rationale="Metric extraction failed for most sweep runs.",
        )

    best_key = max(means, key=means.get)
    if verification_type == "top_model_verification" and sweep.get("runner_up", {}).get("value") in means:
        worst_key = str(sweep["runner_up"]["value"])
    else:
        worst_key = min(means, key=means.get)
    effect_size = round(means[best_key] - means[worst_key], 6)
    best_values = by_config[best_key]
    seed_noise = round(max(best_values) - min(best_values), 6) if len(best_values) > 1 else None
    min_seed_count = min(len(v) for v in by_config.values())
    low_n_warning = min_seed_count < 3
    status, ratio = ladder_status(effect_size, seed_noise)

    supported: list[str] = []
    inconclusive: list[str] = []
    trustworthy = status in {"weakly_supported", "supported", "strongly_supported"}
    best_label = labels.get(best_key, best_key)
    worst_label = labels.get(worst_key, worst_key)
    if verification_type == "top_model_verification":
        blocked_claim = f"{best_label} is robustly better than {worst_label}."
    else:
        blocked_claim = f"{best_label} is robustly better than {worst_label}."
    allowed_claim = (
        f"{best_label} had the best mean {metric}, but the effect was smaller than seed noise."
    )

    if protocol_warning:
        return Verification(
            status="inconclusive",
            metric=metric,
            effect_size=effect_size,
            seed_noise=seed_noise,
            effect_to_noise_ratio=ratio,
            min_seed_count=min_seed_count,
            low_n_warning=low_n_warning,
            trustworthy=False,
            best_config=best_label,
            worst_config=worst_label,
            blocked_claim=f"{best_label} is a valid scientific winner.",
            allowed_claim=f"{best_label} produced the best observed {metric}, but the protocol warning blocks a strong claim: {protocol_warning}.",
            supported_claims=[],
            inconclusive_findings=[protocol_warning],
            rationale=f"Protocol warning blocks the claim even though the metric effect may look large: {protocol_warning}",
        )

    if status == "missing_data":
        inconclusive.append("Seed noise could not be estimated for the best config.")
        rationale = "A claim cannot be verified without repeated seeds and non-zero seed-noise information."
    elif trustworthy:
        strength = status.replace("_", " ")
        allowed_claim = (
            f"{best_label} beat {worst_label} by {effect_size:.4f} {metric}; "
            f"effect/noise ratio={ratio:.2f}, status={strength}."
        )
        supported.append(
            f"Best config '{best_label}' beats worst config '{worst_label}' by {effect_size:.4f} {metric}, above seed noise {seed_noise:.4f}; status={strength}."
        )
        rationale = (
            "Measured effect is larger than inter-seed noise for the best config. "
            "This is not a full statistical significance engine; it is a conservative claim gate."
        )
    else:
        if seed_noise is None:
            inconclusive.append("Only one seed was run for the best config, so seed noise is unknown.")
            rationale = "A claim cannot be verified without repeated seeds."
            allowed_claim = "Only one seed was run for the best config, so robustness remains unverified."
        else:
            inconclusive.append(
                f"Best config '{best_label}' improves over '{worst_label}' by {effect_size:.4f} {metric}, but best-config seed noise is {seed_noise:.4f}."
            )
            if verification_type == "top_model_verification":
                allowed_claim = (
                    f"{best_label} had the best multi-seed mean {metric}, but the effect was smaller than seed noise; "
                    "no robust best-model claim is allowed."
                )
                rationale = "Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model."
            else:
                rationale = "Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery."

    return Verification(
        status=status,
        metric=metric,
        effect_size=effect_size,
        seed_noise=seed_noise,
        effect_to_noise_ratio=ratio,
        min_seed_count=min_seed_count,
        low_n_warning=low_n_warning,
        trustworthy=trustworthy,
        best_config=best_label,
        worst_config=worst_label,
        blocked_claim=blocked_claim if not trustworthy else None,
        allowed_claim=allowed_claim,
        supported_claims=supported,
        inconclusive_findings=inconclusive,
        rationale=rationale,
    )
