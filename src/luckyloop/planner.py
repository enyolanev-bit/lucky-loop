from __future__ import annotations

import json
from itertools import product

from .schemas import (
    AgentDecision,
    CandidatePrediction,
    DecisionTrace,
    ExperimentTrace,
    ProposedAction,
    RejectedCandidate,
    ResearchState,
    SafetyValidation,
    TaskSpec,
)
from .simulator import predict
from .top_models import build_top_model_verification_action


def action_key(action: ProposedAction) -> str:
    return f"{action.model}:{json.dumps(action.params, sort_keys=True)}"


def with_action_id(action: ProposedAction, action_id: str) -> ProposedAction:
    return ProposedAction(action_id=action_id, command=action.command, model=action.model, params=action.params)


def initial_action(task: TaskSpec) -> ProposedAction:
    return ProposedAction(
        action_id="action_001",
        command=f"python experiments/train_sklearn.py --dataset {task.dataset} --model logistic_regression",
        model="logistic_regression",
        params={"dataset": task.dataset, "scale": False},
    )


def initial_hypothesis() -> str:
    return "Establish a simple linear baseline before spending search budget."


def _candidate(action_id: str, command: str, model: str, params: dict) -> ProposedAction:
    return ProposedAction(action_id=action_id, command=command, model=model, params=params)


def _model_command(task: TaskSpec, model: str, overrides: dict | None = None) -> tuple[str, dict]:
    params = {"dataset": task.dataset}
    overrides = overrides or {}
    cmd = f"python experiments/train_sklearn.py --dataset {task.dataset} --model {model}"
    if model == "logistic_regression":
        params.update({"scale": bool(overrides.get("scale", True)), "C": overrides.get("C", 1.0)})
        if params["scale"]:
            cmd += " --scale"
        if params["C"] != 1.0:
            cmd += f" --C {params['C']}"
    elif model == "svc":
        params.update({"scale": bool(overrides.get("scale", True)), "C": overrides.get("C", 2.0), "kernel": overrides.get("kernel", "rbf")})
        if params["scale"]:
            cmd += " --scale"
        cmd += f" --C {params['C']} --kernel {params['kernel']}"
    elif model == "random_forest":
        params.update({"n_estimators": overrides.get("n_estimators", 300), "max_depth": overrides.get("max_depth")})
        cmd += f" --n-estimators {params['n_estimators']}"
        if params["max_depth"] is not None:
            cmd += f" --max-depth {params['max_depth']}"
    elif model == "gradient_boosting":
        params.update({"n_estimators": overrides.get("n_estimators", 150), "learning_rate": overrides.get("learning_rate", 0.05), "max_depth": overrides.get("max_depth")})
        cmd += f" --n-estimators {params['n_estimators']} --learning-rate {params['learning_rate']}"
        if params["max_depth"] is not None:
            cmd += f" --max-depth {params['max_depth']}"
    elif model == "hist_gradient_boosting":
        params.update({"n_estimators": overrides.get("n_estimators", 150), "learning_rate": overrides.get("learning_rate", 0.08), "max_depth": overrides.get("max_depth")})
        cmd += f" --n-estimators {params['n_estimators']} --learning-rate {params['learning_rate']}"
        if params["max_depth"] is not None:
            cmd += f" --max-depth {params['max_depth']}"
    return cmd, params


def _catalog_candidates(task: TaskSpec) -> list[ProposedAction]:
    if not task.candidate_space:
        candidates = []
        for model in task.models:
            cmd, params = _model_command(task, model)
            candidates.append(_candidate(f"action_{model}", cmd, model, params))
        return candidates

    candidates: list[ProposedAction] = []
    for model, grid in task.candidate_space.items():
        keys = list(grid)
        values = [grid[key] for key in keys]
        for combo in product(*values):
            overrides = dict(zip(keys, combo))
            cmd, params = _model_command(task, model, overrides)
            suffix = "_".join(f"{k}={v}" for k, v in sorted(overrides.items()) if v is not None)
            action_id = f"action_{model}_{suffix}".replace(".", "p").replace("=", "-")
            candidates.append(_candidate(action_id, cmd, model, params))
    return candidates


def _sweep_candidate(task: TaskSpec, index: int) -> ProposedAction:
    sweep = task.sweeps[index]
    values = " ".join(str(v) for v in sweep.values)
    seeds = " ".join(str(s) for s in sweep.seeds)
    cmd = (
        f"python experiments/sweep_sklearn.py --dataset {task.dataset} --model {sweep.model} "
        f"--sweep-param {sweep.param} --values {values} --seeds {seeds}"
    )
    if sweep.scale:
        cmd += " --scale"
    if sweep.label_noise:
        cmd += f" --label-noise {sweep.label_noise}"
    return _candidate(
        f"action_sweep_{index + 1}_{sweep.model}_{sweep.param}",
        cmd,
        "verification_sweep",
        {
            "dataset": task.dataset,
            "base_model": sweep.model,
            "scale": sweep.scale,
            "sweep_param": sweep.param,
            "values": sweep.values,
            "seeds": sweep.seeds,
            "label_noise": sweep.label_noise,
        },
    )


def _single_model_search_complete(task: TaskSpec, seen: set[str]) -> bool:
    for model in task.models:
        if not any(key.startswith(f"{model}:") for key in seen):
            return False
    return True


def generate_candidates(task: TaskSpec, state: ResearchState, traces: list[ExperimentTrace], seen: set[str]) -> list[ProposedAction]:
    catalog = _catalog_candidates(task)
    candidates = []
    tested_models = {trace.proposed_action.model for trace in traces if trace.actual_result.accuracy is not None}

    best_model = None
    successful = [trace for trace in traces if trace.actual_result.accuracy is not None]
    if successful:
        best_model = max(successful, key=lambda t: t.actual_result.accuracy or -1).proposed_action.model
    if traces and traces[0].proposed_action.model == "logistic_regression" and not traces[0].proposed_action.params.get("scale"):
        candidates.extend(
            [
                candidate
                for candidate in catalog
                if candidate.model == "logistic_regression" and candidate.params.get("scale")
            ]
        )
    if best_model:
        candidates.extend([candidate for candidate in catalog if candidate.model == best_model])

    for candidate in catalog:
        if candidate.model not in tested_models:
            candidates.append(candidate)

    if any(trace.comparison.unexpected_events for trace in traces):
        candidates.extend(
            [
                candidate
                for candidate in catalog
                if candidate.model == "svc" or "boost" in candidate.model
            ]
        )

    model_search_complete = _single_model_search_complete(task, seen)
    if state.top_model_summary and state.top_model_summary.needs_robustness_verification:
        top_model_action = build_top_model_verification_action(task, state.top_model_summary)
        if top_model_action is not None:
            candidates.insert(0, top_model_action)

    if model_search_complete or (state.budget_remaining is not None and state.budget_remaining <= 2):
        for index, _sweep in enumerate(task.sweeps):
            candidates.append(_sweep_candidate(task, index))

    if not traces:
        candidates.insert(0, initial_action(task))

    deduped = []
    added = set()
    for candidate in candidates:
        key = action_key(candidate)
        if key in seen or key in added:
            continue
        deduped.append(candidate)
        added.add(key)
    return deduped[:12]


def prediction_source(prediction, simulator_configured: bool) -> str:
    if not simulator_configured:
        return "heuristic_fallback"
    text = " ".join([prediction.rationale, *prediction.risks]).lower()
    return "heuristic_fallback" if "fallback" in text or "heuristic" in text else "qwen_agentworld"


def predict_candidates(task: TaskSpec, state: ResearchState, candidates: list[ProposedAction], simulator_configured: bool) -> list[CandidatePrediction]:
    state_text = json.dumps({"task": task.model_dump(), "state": state.model_dump()}, indent=2)
    predictions = []
    for candidate in candidates:
        prediction = predict(candidate, state_text)
        predictions.append(
            CandidatePrediction(
                action=candidate,
                prediction=prediction,
                source=prediction_source(prediction, simulator_configured),
            )
        )
    return predictions


def _best_accuracy(traces: list[ExperimentTrace]) -> float:
    values = [t.actual_result.accuracy for t in traces if t.actual_result.accuracy is not None]
    return max(values, default=0.0)


def _score_candidate(
    candidate_prediction: CandidatePrediction,
    traces: list[ExperimentTrace],
    agent_decision: AgentDecision | None = None,
) -> tuple[float, list[str], list[str], list[str], dict[str, float]]:
    action = candidate_prediction.action
    prediction = candidate_prediction.prediction
    risks = " ".join(prediction.risks).lower()
    rationale = prediction.rationale.lower()
    score = 0.0
    reasons = []
    world_reasons = []
    selector_reasons = []
    breakdown = {
        "expected_gain": 0.0,
        "uncertainty": 0.0,
        "novelty": 0.0,
        "verification_value": 0.0,
        "qwen_signal": 0.0,
        "redundancy_penalty": 0.0,
        "compute_cost_penalty": 0.0,
        "unsupported_claim_risk": 0.0,
        "agent_preference": 0.0,
    }
    tested_models = {t.proposed_action.model for t in traces if t.actual_result.accuracy is not None}

    if agent_decision and action.action_id == agent_decision.preferred_action_id:
        score += 42
        breakdown["agent_preference"] += 42
        reasons.append("autoresearch agent preferred this catalog action")
        selector_reasons.append("agent preferred action accepted into safety scoring")
    elif agent_decision and action.action_id in set(agent_decision.candidate_action_ids):
        score += 8
        breakdown["agent_preference"] += 8
        reasons.append("autoresearch agent included this action in its candidate shortlist")

    if prediction.recommendation == "run":
        score += 20
        breakdown["qwen_signal"] += 20
        reasons.append("world model recommended run")
    elif prediction.recommendation == "modify":
        score += 12
        breakdown["qwen_signal"] += 12
        reasons.append("world model recommended modification, so the action remains informative but lower priority")
        world_reasons.append("world model recommended modification")
    else:
        score -= 25
        breakdown["qwen_signal"] -= 25
        reasons.append("world model recommended skip")
        world_reasons.append("world model recommended skip")

    signal = f"{prediction.action_specific_signal} {prediction.claim_risk} {risks} {rationale}".lower()
    if any(word in signal for word in ["scal", "seed", "variance", "noise", "robust", "leak", "protocol", "metric", "overfit"]):
        score += 10
        breakdown["qwen_signal"] += 10
        world_reasons.append("prediction contained an action-specific research signal")

    if action.model not in tested_models and action.model not in {"verification_sweep", "top_model_verification"}:
        score += 18
        breakdown["novelty"] += 18
        reasons.append("new model family increases search coverage")
        selector_reasons.append("new model family increases search coverage")
    elif action.model in tested_models and action.model not in {"verification_sweep", "top_model_verification"}:
        score -= 12
        breakdown["redundancy_penalty"] -= 12
        reasons.append("candidate is a variant of an already tested family")

    if action.model == "logistic_regression" and not action.params.get("scale") and not traces:
        score += 80
        breakdown["novelty"] += 80
        reasons.append("first run should establish the unscaled baseline before interventions")
        selector_reasons.append("first run should establish the unscaled baseline before interventions")

    if action.model == "logistic_regression" and action.params.get("scale") and not traces:
        score -= 40
        breakdown["redundancy_penalty"] -= 40
        reasons.append("scaled baseline is deferred until after an unscaled control")
        selector_reasons.append("scaled baseline is deferred until after an unscaled control")

    if action.model == "logistic_regression" and action.params.get("scale") and traces:
        first = traces[0]
        if first.proposed_action.model == "logistic_regression" and not first.proposed_action.params.get("scale"):
            score += 45
            breakdown["expected_gain"] += 45
            reasons.append("after an unscaled logistic baseline, scaling is the direct world-model intervention to test")
            selector_reasons.append("after an unscaled logistic baseline, scaling is the direct intervention to test")

    if action.model == "verification_sweep" and _best_accuracy(traces) >= 0.98:
        score += 35
        breakdown["verification_value"] += 35
        reasons.append("high single-run score needs robustness verification before a strong claim")
        selector_reasons.append("high single-run score needs robustness verification before a strong claim")
    if action.model == "verification_sweep" and ("seed variance" in risks or "non-robust" in risks or "noise" in risks):
        score += 25
        breakdown["qwen_signal"] += 25
        reasons.append("world model predicted robustness or seed-variance risk")
        world_reasons.append("world model predicted robustness or seed-variance risk")
    if action.model == "verification_sweep" and len(traces) < 4:
        score -= 30
        breakdown["compute_cost_penalty"] -= 30
        reasons.append("defer verifier sweep until several single-model baselines exist")
        selector_reasons.append("defer verifier sweep until several single-model baselines exist")

    if action.model == "top_model_verification":
        score += 45
        breakdown["verification_value"] += 45
        reasons.append("top observed models need multi-seed verification before a robust best-model claim")
        selector_reasons.append("top observed models need multi-seed verification before a robust best-model claim")
        if any(word in signal for word in ["seed", "variance", "robust", "claim", "single split", "tied", "top"]):
            score += 35
            breakdown["qwen_signal"] += 35
            reasons.append("world model predicted top-model robustness or claim risk")
            world_reasons.append("world model predicted top-model robustness or claim risk")

    if action.model == "svc" and any(t.proposed_action.model == "random_forest" and not t.comparison.metric_match for t in traces):
        score += 30
        breakdown["uncertainty"] += 30
        reasons.append("tree prediction missed, so a scaled margin model tests a different hypothesis")
        selector_reasons.append("tree prediction missed, so a scaled margin model tests a different hypothesis")

    if action.model == "random_forest" and _best_accuracy(traces) >= 0.98:
        score += 35
        breakdown["novelty"] += 35
        reasons.append("strong linear baseline makes a different inductive bias useful")
        selector_reasons.append("strong linear baseline makes a different inductive bias useful")

    if action.model == "gradient_boosting":
        score += 5
        breakdown["novelty"] += 5
        reasons.append("ensemble baseline is useful as a late comparison")
        selector_reasons.append("ensemble baseline is useful as a late comparison")

    if "overfit" in risks or "overfit" in rationale:
        score -= 4
        breakdown["compute_cost_penalty"] -= 4
        reasons.append("world model predicted overfitting risk")
        world_reasons.append("world model predicted overfitting risk")

    if _best_accuracy(traces) >= 0.98 and action.model not in {"verification_sweep", "top_model_verification"}:
        score -= 8
        breakdown["unsupported_claim_risk"] -= 8
        reasons.append("high best score increases claim risk for additional single-run score chasing")

    return score, reasons, world_reasons, selector_reasons, breakdown


def select_candidate(
    state: ResearchState,
    predictions: list[CandidatePrediction],
    traces: list[ExperimentTrace],
    agent_decision: AgentDecision | None = None,
) -> tuple[CandidatePrediction, DecisionTrace, SafetyValidation]:
    if not predictions:
        raise ValueError("select_candidate requires at least one candidate prediction")

    candidate_ids = {cp.action.action_id or "" for cp in predictions}
    valid_agent_action = bool(
        agent_decision
        and agent_decision.preferred_action_id in candidate_ids
        and agent_decision.stop_or_continue == "continue"
    )
    scored = []
    for cp in predictions:
        score, reasons, world_reasons, selector_reasons, breakdown = _score_candidate(cp, traces, agent_decision)
        scored.append((score, cp, reasons, world_reasons, selector_reasons, breakdown))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected_score, selected, selected_reasons, selected_world_reasons, selected_selector_reasons, selected_breakdown = scored[0]
    selection_overrode_agent = bool(
        agent_decision
        and agent_decision.preferred_action_id
        and selected.action.action_id != agent_decision.preferred_action_id
    )
    override_reason = None
    validation_notes = []
    if agent_decision:
        validation_notes.append(f"agent preferred {agent_decision.preferred_action_id}")
    if not valid_agent_action and agent_decision:
        override_reason = "agent preferred action was not present in the candidate catalog or requested stop"
        validation_notes.append(override_reason)
    elif selection_overrode_agent:
        override_reason = "safety selector chose a higher-scoring action after world-model and evidence-risk scoring"
        validation_notes.append(override_reason)

    rejected = []
    for score, candidate_prediction, reasons, _world_reasons, _selector_reasons, breakdown in scored[1:]:
        rejected.append(
            RejectedCandidate(
                action=candidate_prediction.action,
                reason=f"score={score}; " + "; ".join(reasons),
                score_breakdown=breakdown,
            )
        )

    risk_text = ", ".join(selected.prediction.risks) or "no specific risk"
    agent_used = bool(agent_decision and selected.action.action_id == agent_decision.preferred_action_id)
    world_used = bool(selected_world_reasons)
    selector_used = bool(selected_selector_reasons)
    if agent_used and (world_used or selector_used):
        causal_signal_type = "mixed"
    elif agent_used:
        causal_signal_type = "agent_decision"
    elif world_used and selector_used:
        causal_signal_type = "mixed"
    elif world_used:
        causal_signal_type = "world_model_prediction"
    elif selector_used:
        causal_signal_type = "selector_policy"
    else:
        causal_signal_type = "unknown"
    causal_reason = (
        f"Selected {selected.action.model} because score={selected_score}; "
        f"{'; '.join(selected_reasons)}. "
        f"Agent hypothesis: {agent_decision.working_hypothesis if agent_decision else 'selector-only mode'}. "
        f"World model predicted {selected.prediction.expected_metric}, runtime {selected.prediction.expected_runtime_seconds}, "
        f"recommendation={selected.prediction.recommendation}, risks={risk_text}. "
        f"Causal signal type: {causal_signal_type}."
    )

    safety_validation = SafetyValidation(
        valid_agent_action=valid_agent_action if agent_decision else True,
        selected_action_id=selected.action.action_id or "",
        selection_overrode_agent=selection_overrode_agent,
        override_reason=override_reason,
        validation_notes=validation_notes,
    )

    return selected, DecisionTrace(
        selected_action=selected.action,
        agent_signal_used=agent_used,
        world_model_signal_used=world_used,
        selector_policy_signal_used=selector_used,
        causal_signal_type=causal_signal_type,
        observed_state_signal=state.top_model_summary.reason if state.top_model_summary else "",
        world_model_signal="; ".join(selected_world_reasons),
        selector_policy_signal="; ".join(selected_selector_reasons),
        selected_score=selected_score,
        score_breakdown=selected_breakdown,
        qwen_suggested_action=selected.action.action_id,
        catalog_validation="accepted",
        agent_rationale=agent_decision.rationale if agent_decision else "",
        preferred_action_id=agent_decision.preferred_action_id if agent_decision else None,
        selection_overrode_agent=selection_overrode_agent,
        override_reason=override_reason,
        causal_reason=causal_reason,
        rejected_candidates=rejected,
    ), safety_validation
