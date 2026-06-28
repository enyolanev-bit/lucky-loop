from __future__ import annotations
from pathlib import Path
from .calibration import compute_world_model_calibration
from .claim_ledger import build_claim_ledger
from .schemas import ExperimentTrace


def _write_markdown(path: Path, lines: list[str]) -> None:
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _actual_metric_text(t: ExperimentTrace) -> str:
    if t.actual_result.accuracy is not None:
        return f"accuracy {t.actual_result.accuracy:.4f}"
    if t.actual_result.raw.get("best"):
        best_raw = t.actual_result.raw["best"]
        metric = t.actual_result.raw.get("metric", "accuracy")
        if best_raw.get(f"mean_{metric}") is not None:
            return f"best mean {metric} {best_raw[f'mean_{metric}']:.4f}"
    return t.actual_result.status


def _world_model_summary(t: ExperimentTrace) -> str:
    prediction = t.world_model_prediction
    signal = prediction.action_specific_signal or prediction.why_this_action_changes_claims or prediction.rationale
    risks = "; ".join(t.world_model_prediction.risks[:2])
    impact = f"claim_impact={prediction.claim_impact}, compute_value={prediction.compute_value}, rec={prediction.recommendation}"
    base = signal or risks or prediction.expected_metric
    return f"{base} ({impact})"


def _agent_hypothesis(t: ExperimentTrace) -> str:
    if t.agent_decision:
        return t.agent_decision.working_hypothesis
    return t.research_hypothesis or t.hypothesis


def _claim_verdict(t: ExperimentTrace) -> str:
    if t.verification:
        if t.verification.trustworthy:
            return t.verification.status
        return f"blocked: {t.verification.allowed_claim}"
    if t.comparison.unexpected_events:
        return "prediction miss logged; no robust claim"
    return "observation only; no robust claim"


def _agent_action_text(t: ExperimentTrace) -> str:
    if t.decision_trace:
        signal = t.decision_trace.causal_signal_type
    else:
        signal = "unknown"
    if t.proposed_action.model == "verification_sweep":
        base = t.proposed_action.params.get("base_model", "model")
        param = t.proposed_action.params.get("sweep_param", "param")
        return f"ran multi-seed {base} {param} sweep; signal={signal}"
    if t.proposed_action.model == "top_model_verification":
        models = t.actual_result.raw.get("verified_models") or [
            item.get("model_key", item.get("model", "model"))
            for item in t.proposed_action.params.get("models", [])
        ]
        return f"verified top models: {', '.join(models)}; signal={signal}"
    return f"ran {t.proposed_action.model}; signal={signal}"


def write_demo_summary(goal: str, traces: list[ExperimentTrace], path: Path) -> None:
    lines = [
        "# Lucky Loop Demo Summary",
        "",
        f"Goal: {goal}",
        "",
        "All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: agent proposes, world model predicts, Lucky Loop runs, verifier gates claims.",
        "",
        "| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |",
        "|---|---|---|---|---|---|",
    ]
    for t in traces:
        lines.append(
            f"| {t.run_id} | {_agent_hypothesis(t)} | {_world_model_summary(t)} | {_agent_action_text(t)} | "
            f"{_actual_metric_text(t)} | {_claim_verdict(t)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(path, lines)


def generate_report(goal: str, traces: list[ExperimentTrace], path: Path) -> None:
    best = max((t for t in traces if t.actual_result.accuracy is not None), key=lambda t: t.actual_result.accuracy or -1, default=None)
    calibration = compute_world_model_calibration(traces)
    ledger = build_claim_ledger(traces)
    lines = [
        "# Lucky Loop Research Report",
        "",
        f"Goal: {goal}",
        "",
        "## Thesis",
        "",
        "An API-backed autoresearch planner proposes hypotheses and safe catalog actions. Qwen-AgentWorld predicts experimental consequences before compute. Lucky Loop then runs real code, compares prediction with reality, and gates scientific claims through a deterministic verifier.",
        "",
        "## Experiment timeline",
        "",
        "| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |",
        "|---|---|---|---|---|---|",
    ]
    for t in traces:
        lines.append(
            f"| {t.run_id} | {_agent_hypothesis(t)} | {_world_model_summary(t)} | {_agent_action_text(t)} | "
            f"{_actual_metric_text(t)} | {_claim_verdict(t)} |"
        )

    lines += ["", "## Autoresearch agent decisions", ""]
    for t in traces:
        if t.agent_decision:
            override = ""
            if t.safety_validation and t.safety_validation.selection_overrode_agent:
                override = f"; safety override={t.safety_validation.override_reason}"
            lines.append(
                f"- {t.run_id}: preferred={t.agent_decision.preferred_action_id}; "
                f"backend={t.agent_backend}; evidence_needed={t.agent_decision.expected_evidence_needed}{override}"
            )
        elif t.decision_trace:
            lines.append(f"- {t.run_id}: selector-only mode; {t.decision_trace.causal_reason}")

    lines += ["", "## Best result", ""]
    if best:
        lines.append(f"Best single run: {best.run_id}, model={best.proposed_action.model}, accuracy={best.actual_result.accuracy:.4f}, f1={best.actual_result.f1:.4f}.")
    else:
        lines.append("No successful accuracy-bearing single run yet.")

    lines += ["", "## Top model robustness", ""]
    top_verifications = [t for t in traces if t.proposed_action.model == "top_model_verification"]
    if top_verifications:
        for t in top_verifications:
            raw = t.actual_result.raw
            models = ", ".join(raw.get("verified_models") or [])
            verdict = t.verification.status if t.verification else "missing_data"
            ratio = t.verification.effect_to_noise_ratio if t.verification else None
            allowed = t.verification.allowed_claim if t.verification else "No verifier claim was produced."
            lines.append(f"- {t.run_id}: verified {models}; verdict={verdict}; effect/noise={ratio}; {allowed}")
    else:
        lines.append("- No top-model multi-seed verification was run.")

    lines += [
        "",
        "## World model calibration",
        "",
        f"- Metric interval coverage: {'n/a' if calibration.metric_interval_coverage is None else f'{calibration.metric_interval_coverage:.2%}'}",
        f"- Runtime interval coverage: {'n/a' if calibration.runtime_interval_coverage is None else f'{calibration.runtime_interval_coverage:.2%}'}",
        f"- Prediction miss count: {calibration.prediction_miss_count}",
        f"- Useful decision signals: {calibration.useful_decision_count}/{len(traces)}",
        f"- High claim-impact verification/stop decisions: {calibration.high_claim_impact_verification_count}",
        f"- Skip/stop recommendations: {calibration.skip_or_stop_recommendation_count}",
        f"- Memory-augmented predictions: {calibration.memory_augmented_prediction_count}/{len(traces)}",
        f"- Few-shot-augmented predictions: {calibration.few_shot_augmented_prediction_count}/{len(traces)}",
        "- Full calibration table: `reports/world_model_calibration.md`",
    ]

    lines += ["", "## Supported claims", ""]
    supported = [entry for entry in ledger if entry.status in {"supported", "strongly_supported"}]
    if supported:
        lines += [f"- {entry.claim}" for entry in supported]
    else:
        lines.append("- No claim reached supported or strongly_supported yet.")

    lines += ["", "## Weakly supported claims", ""]
    weak = [entry for entry in ledger if entry.status == "weakly_supported"]
    if weak:
        lines += [f"- {entry.claim}" for entry in weak]
    else:
        lines.append("- No weakly supported claim was recorded.")

    lines += ["", "## Blocked / inconclusive claims", ""]
    blocked = [entry for entry in ledger if entry.status in {"blocked", "inconclusive"}]
    if blocked:
        for entry in blocked:
            lines.append(f"- Blocked: {entry.claim}")
            if entry.allowed_rewrite:
                lines.append(f"  Allowed rewrite: {entry.allowed_rewrite}")
    else:
        lines.append("- No verifier-level blocked claim was recorded.")

    lines += ["", "## Claim ledger", "", "- Full ledger: `reports/claim_ledger.json`"]

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
        lines.append(
            f"- Prediction schema: prompt={t.world_model_prediction.prompt_version or t.prompt_version or 'legacy'}; "
            f"schema={t.world_model_prediction.world_model_schema_version or t.world_model_schema_version or 'legacy'}; "
            f"claim_impact={t.world_model_prediction.claim_impact}; compute_value={t.world_model_prediction.compute_value}; "
            f"recommendation={t.world_model_prediction.recommendation}"
        )
        if t.world_model_prediction.why_this_action_changes_claims:
            lines.append(f"- Claim impact rationale: {t.world_model_prediction.why_this_action_changes_claims}")
        if t.world_model_prediction.why_this_action_may_be_wasteful:
            lines.append(f"- Wastefulness rationale: {t.world_model_prediction.why_this_action_may_be_wasteful}")
        if t.world_model_prediction.memory_example_ids:
            lines.append(f"- Retrieved memory examples: {', '.join(t.world_model_prediction.memory_example_ids)}")
        if t.world_model_prediction.few_shot_example_ids:
            lines.append(f"- Few-shot examples: {', '.join(t.world_model_prediction.few_shot_example_ids)}")
        lines.append(f"- Risks: {', '.join(t.world_model_prediction.risks) or 'none'}")
        if t.state_before:
            lines.append(f"- State before: {t.state_before.state_id}; budget_remaining={t.state_before.budget_remaining}; known_results={len(t.state_before.known_results)}")
        if t.candidate_actions:
            candidates = ", ".join(f"{c.action_id or 'candidate'}:{c.model}" for c in t.candidate_actions)
            lines.append(f"- Candidates considered: {candidates}")
        if t.decision_trace:
            lines.append(f"- Planner decision: {t.decision_trace.causal_reason}")
            if t.agent_decision:
                lines.append(f"- Agent rationale: {t.agent_decision.rationale}")
                lines.append(f"- Evidence needed: {t.agent_decision.expected_evidence_needed}")
                lines.append(f"- Claim risk: {t.agent_decision.claim_risk}")
            if t.safety_validation:
                lines.append(
                    f"- Safety validation: selected={t.safety_validation.selected_action_id}; "
                    f"override={t.safety_validation.selection_overrode_agent}; "
                    f"reason={t.safety_validation.override_reason or 'none'}"
                )
            if t.decision_trace.rejected_candidates:
                rejected = "; ".join(f"{r.action.model}: {r.reason}" for r in t.decision_trace.rejected_candidates)
                lines.append(f"- Rejected / deferred: {rejected}")
        lines.append(f"- Actual status: {t.actual_result.status}, runtime: {t.actual_result.runtime_seconds}s")
        if t.comparison.unexpected_events:
            lines.append(f"- Unexpected: {'; '.join(t.comparison.unexpected_events)}")
        lines.append(f"- Lesson: {t.comparison.lesson}")
        if t.verification:
            lines.append(
                f"- Verifier verdict: {t.verification.status}; trustworthy={t.verification.trustworthy}; "
                f"effect_size={t.verification.effect_size}; seed_noise={t.verification.seed_noise}; "
                f"effect_to_noise_ratio={t.verification.effect_to_noise_ratio}"
            )
            if t.verification.blocked_claim:
                lines.append(f"- Blocked claim: {t.verification.blocked_claim}")
            if t.verification.allowed_claim:
                lines.append(f"- Allowed claim: {t.verification.allowed_claim}")
            lines.append(f"- Verifier rationale: {t.verification.rationale}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(path, lines)
    write_demo_summary(goal, traces, path.parent / "demo_summary.md")
