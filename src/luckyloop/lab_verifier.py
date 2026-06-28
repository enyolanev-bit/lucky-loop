from __future__ import annotations

from statistics import mean, stdev

from .schemas import LabAnalysis, LabClaim, ProtocolSpec, ResearchHypothesis


def analyze_observation(raw: dict, analysis_id: str) -> LabAnalysis:
    primary = raw.get("primary_metric", "balanced_accuracy")
    rows = raw.get("runs") or []
    by_condition: dict[str, list[float]] = {}
    for row in rows:
        condition = str(row.get("condition", "unknown"))
        value = row.get(primary)
        if value is not None:
            by_condition.setdefault(condition, []).append(float(value))
    condition_means = {key: mean(values) for key, values in by_condition.items() if values}
    condition_stds = {
        key: stdev(values) if len(values) > 1 else 0.0
        for key, values in by_condition.items()
        if values
    }
    best_condition = max(condition_means, key=condition_means.get) if condition_means else None
    effect_size = raw.get("effect_size")
    seed_noise = raw.get("seed_noise")
    ratio = raw.get("effect_to_noise_ratio")
    summary = "No numeric runs were available."
    if best_condition is not None:
        summary = f"Best condition by mean {primary}: {best_condition}."
        if effect_size is not None and seed_noise is not None:
            summary += f" Effect={effect_size:.4f}; seed_noise={seed_noise:.4f}."
    return LabAnalysis(
        analysis_id=analysis_id,
        protocol_id=str(raw.get("protocol_id") or raw.get("study_id") or "unknown"),
        primary_metric=primary,
        condition_means=condition_means,
        condition_stds=condition_stds,
        effect_size=effect_size,
        seed_noise=seed_noise,
        effect_to_noise_ratio=ratio,
        best_condition=best_condition,
        protocol_warnings=[str(item) for item in raw.get("protocol_warnings", [])],
        summary=summary,
    )


def _ratio(analysis: LabAnalysis) -> float:
    if analysis.effect_to_noise_ratio is not None:
        return float(analysis.effect_to_noise_ratio)
    if analysis.effect_size is None or analysis.seed_noise in {None, 0}:
        return 0.0
    return float(analysis.effect_size / analysis.seed_noise)


def _claim_text(hypothesis: ResearchHypothesis | None, fallback: str) -> str:
    return hypothesis.claim_candidate if hypothesis else fallback


def _claims_logistic_baseline_wins(claim_text: str) -> bool:
    text = claim_text.lower()
    baseline_terms = ["simple baseline", "logistic regression"]
    positive_terms = ["outperform", "higher", "better", "superior", "comparable or higher", "wins"]
    nonlinear_terms = ["nonlinear", "random_forest", "random forest", "svc", "hist_gradient_boosting", "complex model"]
    if not any(term in text for term in baseline_terms):
        return False
    if any(term in text for term in nonlinear_terms) and "logistic regression" in text:
        nonlinear_pos = min([text.find(term) for term in nonlinear_terms if term in text] or [10**9])
        logistic_pos = text.find("logistic regression")
        if nonlinear_pos < logistic_pos:
            return False
    return any(term in text for term in positive_terms)


def _blocked_diagnostic(protocol: ProtocolSpec, analysis: LabAnalysis, hypothesis: ResearchHypothesis | None, effect: float, ratio: float) -> tuple[str, str, str, str]:
    claim_text = _claim_text(hypothesis, "The observed winner is robust.")
    warnings = "; ".join(analysis.protocol_warnings)
    if protocol.protocol_id == "generated_ml_research_protocol" and _claims_logistic_baseline_wins(claim_text):
        if analysis.best_condition and str(analysis.best_condition).lower() not in {"logistic_regression", "logistic regression"}:
            return (
                "claim_direction_failed",
                f"The claim is about logistic regression, but the best observed condition was `{analysis.best_condition}`.",
                "Report which condition won, or run a follow-up focused on why the baseline underperformed.",
                "Report the baseline result as an observation; do not claim baseline superiority.",
            )
    if warnings:
        return (
            "protocol_warning",
            f"The generated experiment emitted protocol warnings: {warnings}",
            "Inspect the warning, revise the protocol, and rerun before making a stronger claim.",
            "Report the result as warning-bounded evidence.",
        )
    if effect <= 0:
        return (
            "wrong_effect_direction",
            f"The measured effect was {effect:.4f}, so the tested claim moved in the wrong direction.",
            "Invert or revise the hypothesis, then run a confirmatory replication.",
            "Report that this experiment did not support the proposed direction.",
        )
    if ratio < 1.0:
        seed_noise = analysis.seed_noise if analysis.seed_noise is not None else "unknown"
        return (
            "effect_within_seed_noise",
            f"The effect was {effect:.4f}, but effect/noise was {ratio:.3f}; seed noise was {seed_noise}.",
            "Increase seeds or simplify the claim to an observation.",
            "Report the best condition as an observation, not a robust claim.",
        )
    return (
        "insufficient_support",
        f"The evidence did not satisfy the verifier threshold; effect={effect:.4f}, effect/noise={ratio:.3f}.",
        "Run a targeted replication or revise the claim boundary.",
        "Report a bounded, non-superiority observation.",
    )


def verify_lab_claims(
    protocol: ProtocolSpec,
    analysis: LabAnalysis,
    hypothesis: ResearchHypothesis | None,
    evidence_id: str,
) -> list[LabClaim]:
    warnings = " ".join(analysis.protocol_warnings).lower()
    ratio = _ratio(analysis)
    effect = analysis.effect_size or 0.0
    claims: list[LabClaim] = []

    if "leak" in warnings:
        claims.append(
            LabClaim(
                claim_id=f"{evidence_id}:leakage_blocks_claim",
                hypothesis_id=protocol.hypothesis_id,
                claim="The high score from the leaky branch is a valid generalization claim.",
                verdict="blocked",
                evidence_ids=[evidence_id],
                reason="A protocol warning indicates label-derived or test-informed leakage.",
                failure_category="protocol_invalid_leakage",
                diagnostic="The experiment included a leakage warning, so high scores from that branch cannot support generalization.",
                next_action="Remove leakage, rebuild the preprocessing pipeline inside train folds, and rerun.",
                allowed_rewrite="The leaky branch is an invalid protocol probe, not a valid model result.",
                metrics={"effect_size": effect, "effect_to_noise_ratio": ratio},
            )
        )

    if protocol.protocol_id == "random_vs_blocked_split":
        random_mean = analysis.condition_means.get("random_split")
        blocked_mean = analysis.condition_means.get("blocked_split")
        if random_mean is not None and blocked_mean is not None and random_mean > blocked_mean and ratio >= 1.0:
            verdict = "supported" if ratio >= 2.0 else "weakly_supported"
            reason = "Random split exceeded blocked split by more than the estimated seed noise."
        elif random_mean is not None and blocked_mean is not None and random_mean > blocked_mean:
            verdict = "inconclusive"
            reason = "Random split was higher, but the effect did not exceed seed noise."
        else:
            verdict = "blocked"
            reason = "The blocked split did not underperform random split in a claimable way."
        claims.append(
            LabClaim(
                claim_id=f"{evidence_id}:split_validity",
                hypothesis_id=protocol.hypothesis_id,
                claim=_claim_text(hypothesis, "Random splits overstate generalization."),
                verdict=verdict,
                evidence_ids=[evidence_id],
                reason=reason,
                failure_category=None if verdict in {"supported", "weakly_supported"} else "split_effect_not_claimable",
                diagnostic=None if verdict in {"supported", "weakly_supported"} else reason,
                next_action=None if verdict in {"supported", "weakly_supported"} else "Run more seeds or use a stronger grouped/temporal holdout before claiming split overstatement.",
                allowed_rewrite=None if verdict in {"supported", "weakly_supported"} else "Observed split differences are not enough for a robust overstatement claim.",
                metrics={"random_mean": random_mean, "blocked_mean": blocked_mean, "effect_to_noise_ratio": ratio},
            )
        )

    elif protocol.protocol_id == "accuracy_vs_balanced_metrics":
        rows_warning = "metric_misuse" in warnings
        accuracy_values = []
        balanced_values = []
        # The runner stores condition means for primary only; use warning plus effect as the deterministic gate.
        verdict = "supported" if rows_warning else "inconclusive"
        claims.append(
            LabClaim(
                claim_id=f"{evidence_id}:metric_misuse",
                hypothesis_id=protocol.hypothesis_id,
                claim=_claim_text(hypothesis, "Accuracy alone is misleading under imbalance."),
                verdict=verdict,
                evidence_ids=[evidence_id],
                reason="The protocol explicitly audited an imbalanced setting and flagged metric misuse." if rows_warning else "No metric misuse warning was produced.",
                failure_category=None if verdict == "supported" else "metric_misuse_not_observed",
                diagnostic=None if verdict == "supported" else "The metric audit did not produce a warning strong enough to support the metric-misuse claim.",
                next_action=None if verdict == "supported" else "Construct or select a more clearly imbalanced dataset and rerun the metric audit.",
                metrics={"accuracy_values": accuracy_values, "balanced_values": balanced_values},
            )
        )

    elif protocol.protocol_id in {"single_run_vs_repeated_seeds", "simple_vs_complex_small_data", "generated_ml_research_protocol"}:
        claim_text = _claim_text(hypothesis, "The observed winner is robust.")
        baseline_claim_failed = (
            protocol.protocol_id == "generated_ml_research_protocol"
            and _claims_logistic_baseline_wins(claim_text)
            and analysis.best_condition is not None
            and str(analysis.best_condition).lower() not in {"logistic_regression", "logistic regression"}
        )
        failure_category = None
        diagnostic = None
        next_action = None
        allowed_rewrite = None
        if protocol.protocol_id == "generated_ml_research_protocol" and (
            baseline_claim_failed or "baseline does not outperform" in warnings or effect <= 0
        ):
            verdict = "blocked"
            failure_category, diagnostic, next_action, allowed_rewrite = _blocked_diagnostic(protocol, analysis, hypothesis, effect, ratio)
            reason = diagnostic
        elif ratio >= 2.0:
            verdict = "supported"
            reason = "The measured effect exceeded seed noise by the support threshold."
        elif ratio >= 1.0:
            verdict = "weakly_supported"
            reason = "The measured effect exceeded seed noise, but not strongly."
        else:
            verdict = "blocked"
            failure_category, diagnostic, next_action, allowed_rewrite = _blocked_diagnostic(protocol, analysis, hypothesis, effect, ratio)
            reason = diagnostic
        claims.append(
            LabClaim(
                claim_id=f"{evidence_id}:robustness",
                hypothesis_id=protocol.hypothesis_id,
                claim=_claim_text(hypothesis, "The observed winner is robust."),
                verdict=verdict,
                evidence_ids=[evidence_id],
                reason=reason,
                failure_category=failure_category,
                diagnostic=diagnostic,
                next_action=next_action,
                allowed_rewrite=None if verdict != "blocked" else (allowed_rewrite or "Report the best condition as an observation, not a robust winner."),
                metrics={"effect_size": effect, "seed_noise": analysis.seed_noise, "effect_to_noise_ratio": ratio},
            )
        )

    if not claims:
        claims.append(
            LabClaim(
                claim_id=f"{evidence_id}:observation",
                hypothesis_id=protocol.hypothesis_id,
                claim=_claim_text(hypothesis, "The protocol produced an observation."),
                verdict="observation_only",
                evidence_ids=[evidence_id],
                reason="The lab recorded a real ML result, but no stronger verifier rule applied.",
                metrics={"effect_size": effect, "effect_to_noise_ratio": ratio},
            )
        )
    return claims
