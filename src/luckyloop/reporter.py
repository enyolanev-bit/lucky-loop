from __future__ import annotations
from pathlib import Path
from .schemas import ExperimentTrace


def generate_report(goal: str, traces: list[ExperimentTrace], path: Path) -> None:
    best = max((t for t in traces if t.actual_result.accuracy is not None), key=lambda t: t.actual_result.accuracy or -1, default=None)
    lines = [
        "# Lucky Loop Research Report",
        "",
        f"Goal: {goal}",
        "",
        "## Thesis",
        "",
        "Predict before you compute, then verify before you claim: each experiment is simulated before real execution, compared against actual metrics, and any sweep claim is gated by a deterministic effect-vs-noise verifier.",
        "",
        "## Experiment timeline",
        "",
        "| Run | Hypothesis | Model | Prediction | Actual accuracy | Match | Verifier | Decision |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for t in traces:
        acc = "" if t.actual_result.accuracy is None else f"{t.actual_result.accuracy:.4f}"
        if acc == "" and t.actual_result.raw.get("best"):
            best_raw = t.actual_result.raw["best"]
            if best_raw.get("mean_accuracy") is not None:
                acc = f"best mean {best_raw['mean_accuracy']:.4f}"
        match = "yes" if t.comparison.metric_match and t.comparison.runtime_match else "partial/no"
        verifier = ""
        if t.verification:
            verifier = f"{t.verification.status}; effect={t.verification.effect_size}; noise={t.verification.seed_noise}"
        lines.append(f"| {t.run_id} | {t.hypothesis} | {t.proposed_action.model} | {t.world_model_prediction.expected_metric} | {acc} | {match} | {verifier} | {t.next_decision} |")

    lines += ["", "## Best result", ""]
    if best:
        lines.append(f"Best single run: {best.run_id}, model={best.proposed_action.model}, accuracy={best.actual_result.accuracy:.4f}, f1={best.actual_result.f1:.4f}.")
    else:
        lines.append("No successful accuracy-bearing single run yet.")

    lines += ["", "## Supported claims", ""]
    supported = [claim for t in traces if t.verification for claim in t.verification.supported_claims]
    if supported:
        lines += [f"- {claim}" for claim in supported]
    else:
        lines.append("- No sweep-level claim cleared the effect-vs-noise verifier yet.")

    lines += ["", "## Weak / inconclusive findings", ""]
    inconclusive = [finding for t in traces if t.verification for finding in t.verification.inconclusive_findings]
    if inconclusive:
        lines += [f"- {finding}" for finding in inconclusive]
    else:
        lines.append("- No verifier-level inconclusive finding was recorded.")

    lines += ["", "## Prediction misses", ""]
    misses = []
    for t in traces:
        if t.comparison.unexpected_events:
            misses.append(f"- {t.run_id}: {'; '.join(t.comparison.unexpected_events)}")
    lines += misses or ["- No unexpected prediction miss was recorded."]

    lines += ["", "## Evidence notes", ""]
    for t in traces:
        lines.append(f"### {t.run_id}")
        lines.append(f"- Prediction rationale: {t.world_model_prediction.rationale}")
        lines.append(f"- Risks: {', '.join(t.world_model_prediction.risks) or 'none'}")
        lines.append(f"- Actual status: {t.actual_result.status}, runtime: {t.actual_result.runtime_seconds}s")
        if t.comparison.unexpected_events:
            lines.append(f"- Unexpected: {'; '.join(t.comparison.unexpected_events)}")
        lines.append(f"- Lesson: {t.comparison.lesson}")
        if t.verification:
            lines.append(f"- Verifier verdict: {t.verification.status}; trustworthy={t.verification.trustworthy}; effect_size={t.verification.effect_size}; seed_noise={t.verification.seed_noise}")
            lines.append(f"- Verifier rationale: {t.verification.rationale}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
