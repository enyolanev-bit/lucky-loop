from __future__ import annotations
from pathlib import Path
from .schemas import ExperimentTrace


def generate_report(goal: str, traces: list[ExperimentTrace], path: Path, verdict=None) -> None:
    best = max((t for t in traces if t.actual_result.accuracy is not None), key=lambda t: t.actual_result.accuracy, default=None)
    lines = ["# LuckyWorld Research Report", "", f"Goal: {goal}", "", "## Thesis", "", "Predict before you compute: each experiment was simulated before real execution, then compared against actual logs and metrics.", "", "## Experiment timeline", "", "| Run | Hypothesis | Model | Prediction | Actual accuracy | Match | Decision |", "|---|---|---|---|---:|---|---|"]
    for t in traces:
        acc = "" if t.actual_result.accuracy is None else f"{t.actual_result.accuracy:.4f}"
        match = "yes" if t.comparison.metric_match and t.comparison.runtime_match else "partial/no"
        lines.append(f"| {t.run_id} | {t.hypothesis} | {t.proposed_action.model} | {t.world_model_prediction.expected_metric} | {acc} | {match} | {t.next_decision} |")
    lines += ["", "## Best result", ""]
    if best:
        lines.append(f"Best run: {best.run_id}, model={best.proposed_action.model}, accuracy={best.actual_result.accuracy:.4f}, f1={best.actual_result.f1:.4f}.")
    else:
        lines.append("No successful accuracy-bearing run yet.")
    lines += ["", "## Trust verification (Verifier — effect vs noise)", ""]
    if verdict is not None:
        lines.append("Couche de confiance #2 : le 'best' ci-dessus est-il fiable, ou dans le bruit inter-seed ?")
        lines.append("")
        lines.append(f"- Méthode best (multi-seed): **{verdict.best_method}** (acc {verdict.best_acc:.4f})")
        lines.append(f"- Écart best vs 2e: {verdict.effect_size:.4f} | bruit inter-seed: {verdict.seed_noise:.4f}")
        flag = "✅ FINDING FIABLE" if verdict.trustworthy else "❌ INCONCLUSIVE (within noise)"
        lines.append(f"- **Verdict: {flag}**")
        lines.append(f"- {verdict.statement}")
    else:
        lines.append("_(pas assez de méthodes comparables pour vérifier — ≥2 requises.)_")
    lines += ["", "## Evidence notes", ""]
    for t in traces:
        lines.append(f"### {t.run_id}")
        lines.append(f"- Prediction rationale: {t.world_model_prediction.rationale}")
        lines.append(f"- Risks: {', '.join(t.world_model_prediction.risks) or 'none'}")
        lines.append(f"- Actual status: {t.actual_result.status}, runtime: {t.actual_result.runtime_seconds}s")
        if t.comparison.unexpected_events:
            lines.append(f"- Unexpected: {'; '.join(t.comparison.unexpected_events)}")
        lines.append(f"- Lesson: {t.comparison.lesson}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
