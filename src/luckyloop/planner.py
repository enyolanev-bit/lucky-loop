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
from .world_model_memory import retrieve_similar_memories


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
        params.update({
            "scale": bool(overrides.get("scale", True)),
            "C": overrides.get("C", 1.0),
            "max_iter": overrides.get("max_iter"),
            "solver": overrides.get("solver"),
        })
        if params["scale"]:
            cmd += " --scale"
        if params["C"] != 1.0:
            cmd += f" --C {params['C']}"
        if params["max_iter"] is not None:
            cmd += f" --max-iter {params['max_iter']}"
        if params["solver"] is not None:
            cmd += f" --solver {params['solver']}"
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


def _stop_candidate(task: TaskSpec) -> ProposedAction:
    return _candidate(
        "action_stop_and_report",
        "python -c \"print('stop_and_report: no further compute requested')\"",
        "stop_and_report",
        {"dataset": task.dataset, "reason": "report from current verified evidence"},
    )


def _protocol_probe_candidate(task: TaskSpec) -> ProposedAction | None:
    if not task.sweeps:
        return None
    sweep = task.sweeps[0]
    values = " ".join(str(v) for v in sweep.values[:3])
    seeds = " ".join(str(s) for s in (sweep.seeds[:3] or [0, 1, 2]))
    warning = (
        "protocol_probe: this result is intentionally treated as protocol-fragile; "
        "a high metric must be rewritten as an observation, not a robust scientific claim"
    )
    cmd = (
        f"python experiments/sweep_sklearn.py --dataset {task.dataset} --model {sweep.model} "
        f"--sweep-param {sweep.param} --values {values} --seeds {seeds} "
        f"--protocol-warning {json.dumps(warning)}"
    )
    if sweep.scale:
        cmd += " --scale"
    if sweep.label_noise:
        cmd += f" --label-noise {sweep.label_noise}"
    return _candidate(
        f"action_protocol_probe_{sweep.model}_{sweep.param}",
        cmd,
        "protocol_probe",
        {
            "dataset": task.dataset,
            "base_model": sweep.model,
            "scale": sweep.scale,
            "sweep_param": sweep.param,
            "values": sweep.values[:3],
            "seeds": sweep.seeds[:3] or [0, 1, 2],
            "label_noise": sweep.label_noise,
            "protocol_warning": warning,
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

    if (
        state.top_model_summary
        and state.top_model_summary.needs_robustness_verification
        and not any(trace.proposed_action.model == "protocol_probe" for trace in traces)
    ):
        protocol_probe = _protocol_probe_candidate(task)
        if protocol_probe is not None:
            candidates.append(protocol_probe)

    if any(trace.verification and not trace.verification.trustworthy for trace in traces):
        candidates.insert(0, _stop_candidate(task))
    elif state.budget_remaining is not None and state.budget_remaining <= 1 and traces:
        candidates.append(_stop_candidate(task))

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


def predict_candidates(
    task: TaskSpec,
    state: ResearchState,
    candidates: list[ProposedAction],
    simulator_configured: bool,
    traces: list[ExperimentTrace] | None = None,
) -> list[CandidatePrediction]:
    state_text = json.dumps({"task": task.model_dump(), "state": state.model_dump()}, indent=2)
    traces = traces or []
    predictions = []
    for candidate in candidates:
        memory_examples = retrieve_similar_memories(task, candidate, traces)
        prediction = predict(candidate, state_text, memory_examples=memory_examples)
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


def _level_value(value: str | None) -> float:
    return {"low": 0.2, "medium": 0.55, "high": 0.9}.get(str(value or "medium").lower(), 0.55)


def _runtime_cost_value(candidate_prediction: CandidatePrediction) -> float:
    prediction = candidate_prediction.prediction
    if prediction.predicted_next_state.expected_compute_cost_seconds is not None:
        seconds = prediction.predicted_next_state.expected_compute_cost_seconds
    elif prediction.expected_runtime_range_seconds:
        seconds = max(prediction.expected_runtime_range_seconds)
    else:
        seconds = 5.0
    if seconds <= 1:
        return 0.05
    if seconds <= 5:
        return 0.15
    if seconds <= 20:
        return 0.35
    if seconds <= 60:
        return 0.65
    return 0.9


def _expected_metric_gain_value(candidate_prediction: CandidatePrediction, traces: list[ExperimentTrace]) -> float:
    best = _best_accuracy(traces)
    rng = candidate_prediction.prediction.expected_metric_range
    if not rng or best <= 0:
        return 0.3 if not traces else 0.15
    midpoint = sum(rng[:2]) / 2
    delta = midpoint - best
    if delta >= 0.03:
        return 0.9
    if delta >= 0.01:
        return 0.65
    if delta >= -0.005:
        return 0.35
    return 0.1


def _cost_aware_utility(
    candidate_prediction: CandidatePrediction,
    traces: list[ExperimentTrace],
    redundant_variant: bool,
) -> dict[str, float]:
    prediction = candidate_prediction.prediction
    next_state = prediction.predicted_next_state
    expected_information_gain = prediction.expected_value_of_information
    expected_claim_resolution = prediction.expected_claim_resolution
    expected_metric_gain = _expected_metric_gain_value(candidate_prediction, traces)
    expected_uncertainty_reduction = _level_value(next_state.uncertainty_reduction)
    expected_research_value = _level_value(next_state.expected_research_value)
    runtime_cost = _runtime_cost_value(candidate_prediction)
    redundancy_cost = 0.75 if redundant_variant else 0.0
    low_claim_impact_cost = 0.6 if prediction.claim_impact == "low" else 0.0
    utility = (
        1.5 * expected_information_gain
        + 2.0 * expected_claim_resolution
        + 1.2 * expected_uncertainty_reduction
        + 1.0 * expected_research_value
        + 0.8 * expected_metric_gain
        - 0.9 * runtime_cost
        - 1.1 * redundancy_cost
        - 0.8 * low_claim_impact_cost
    )
    return {
        "expected_information_gain": round(expected_information_gain, 4),
        "expected_claim_resolution": round(expected_claim_resolution, 4),
        "expected_uncertainty_reduction": round(expected_uncertainty_reduction, 4),
        "expected_research_value": round(expected_research_value, 4),
        "expected_metric_gain": round(expected_metric_gain, 4),
        "expected_runtime_cost": round(runtime_cost, 4),
        "redundancy_cost": round(redundancy_cost, 4),
        "low_claim_impact_cost": round(low_claim_impact_cost, 4),
        "research_value_per_compute": round(utility, 4),
    }


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
        "expected_information_gain": 0.0,
        "expected_claim_resolution": 0.0,
        "expected_uncertainty_reduction": 0.0,
        "expected_research_value": 0.0,
        "expected_runtime_cost": 0.0,
        "research_value_per_compute": 0.0,
    }
    tested_models = {t.proposed_action.model for t in traces if t.actual_result.accuracy is not None}
    non_model_actions = {"verification_sweep", "top_model_verification", "protocol_probe", "stop_and_report"}
    redundant_variant = action.model in tested_models and action.model not in non_model_actions
    verifier_blocked_claim = any(trace.verification and not trace.verification.trustworthy for trace in traces)
    cost_aware = _cost_aware_utility(candidate_prediction, traces, redundant_variant)
    utility_bonus = 28 * cost_aware["research_value_per_compute"]
    score += utility_bonus
    breakdown.update(cost_aware)
    claim_delta_value = {
        "none": -0.6,
        "adds_observation": 0.1,
        "reduces_uncertainty": 0.55,
        "enables_claim": 0.85,
        "blocks_or_rewrites_claim": 1.0,
        "report_ready": 0.9,
    }.get(prediction.expected_claim_delta, 0.1)
    protocol_risk_value = min(len(prediction.protocol_risks), 4) / 4
    waste_penalty = prediction.compute_waste_risk
    world_controller_bonus = 34 * claim_delta_value + 16 * protocol_risk_value - 20 * waste_penalty
    score += world_controller_bonus
    breakdown["expected_claim_delta"] = round(claim_delta_value, 4)
    breakdown["protocol_risk_reduction"] = round(protocol_risk_value, 4)
    breakdown["compute_waste_risk"] = round(waste_penalty, 4)
    breakdown["world_controller_bonus"] = round(world_controller_bonus, 4)
    reasons.append(f"cost-aware utility={cost_aware['research_value_per_compute']}")
    reasons.append(
        f"world-controller claim_delta={prediction.expected_claim_delta}, "
        f"compute_waste_risk={prediction.compute_waste_risk:.2f}"
    )
    if prediction.expected_claim_delta in {"reduces_uncertainty", "enables_claim", "blocks_or_rewrites_claim", "report_ready"}:
        world_reasons.append(f"world model predicted claim_delta={prediction.expected_claim_delta}")
    if prediction.why_not_classic_autoresearch:
        world_reasons.append("world model contrasted this action with classic autoresearch")
    if cost_aware["research_value_per_compute"] >= 2.5:
        world_reasons.append("world model predicted high research value per compute")
    elif cost_aware["research_value_per_compute"] <= 0.7:
        world_reasons.append("world model predicted low research value per compute")

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
    elif prediction.recommendation == "verify":
        score += 34
        breakdown["qwen_signal"] += 34
        breakdown["verification_value"] += 18
        reasons.append("world model recommended verification")
        world_reasons.append("world model recommended verification")
    elif prediction.recommendation == "stop_and_report":
        score += 10
        breakdown["qwen_signal"] += 10
        reasons.append("world model recommended stop_and_report")
        world_reasons.append("world model recommended stop_and_report")
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

    if prediction.claim_impact == "high":
        score += 20
        breakdown["unsupported_claim_risk"] += 10
        reasons.append("world model predicted high claim impact")
        world_reasons.append("world model predicted high claim impact")
    elif prediction.claim_impact == "low":
        score -= 14
        breakdown["unsupported_claim_risk"] -= 14
        reasons.append("world model predicted low claim impact")
        world_reasons.append("world model predicted low claim impact")

    if prediction.compute_value == "high":
        score += 12
        breakdown["qwen_signal"] += 12
        reasons.append("world model predicted high compute value")
        world_reasons.append("world model predicted high compute value")
    elif prediction.compute_value == "low":
        score -= 16
        breakdown["compute_cost_penalty"] -= 16
        reasons.append("world model predicted low compute value")
        world_reasons.append("world model predicted low compute value")

    signal = f"{prediction.action_specific_signal} {prediction.claim_risk} {risks} {rationale}".lower()
    if any(word in signal for word in ["scal", "seed", "variance", "noise", "robust", "leak", "protocol", "metric", "overfit"]):
        score += 10
        breakdown["qwen_signal"] += 10
        world_reasons.append("prediction contained an action-specific research signal")

    if action.model not in tested_models and action.model not in non_model_actions:
        score += 18
        breakdown["novelty"] += 18
        reasons.append("new model family increases search coverage")
        selector_reasons.append("new model family increases search coverage")
    elif action.model in tested_models and action.model not in non_model_actions:
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

    if action.model == "protocol_probe":
        score += 55
        breakdown["verification_value"] += 35
        breakdown["qwen_signal"] += 20
        reasons.append("protocol probe tests whether a tempting result should be blocked or rewritten")
        world_reasons.append("world model predicted protocol or metric-risk value")

    if action.model == "stop_and_report":
        has_verification = any(trace.verification for trace in traces)
        if verifier_blocked_claim:
            score += 260
            breakdown["verification_value"] += 160
            breakdown["compute_cost_penalty"] += 60
            reasons.append("verifier already blocked the robust claim; stop instead of score-chasing")
            selector_reasons.append("verifier already blocked the robust claim; stop instead of score-chasing")
        elif prediction.recommendation == "stop_and_report" and has_verification:
            # Verify before claim: the world model may only motivate stopping AFTER a verifier
            # has actually run. Without a verification on record, stopping is always premature.
            score += 35
            breakdown["qwen_signal"] += 35
            reasons.append("world model recommended stopping and a verifier has already run")
            world_reasons.append("world model recommended stopping and a verifier has already run")
        else:
            score -= 35
            breakdown["compute_cost_penalty"] -= 35
            reason = (
                "stop is premature: no verifier has run yet (verify before claim)"
                if not has_verification
                else "stop is premature before claim risk has been checked"
            )
            reasons.append(reason)
            selector_reasons.append(reason)

    if verifier_blocked_claim and action.model not in non_model_actions:
        score -= 140
        breakdown["compute_cost_penalty"] -= 70
        breakdown["unsupported_claim_risk"] -= 70
        reasons.append("single-run score chasing is penalized after the verifier blocked the robust claim")
        selector_reasons.append("single-run score chasing is penalized after the verifier blocked the robust claim")

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

    if _best_accuracy(traces) >= 0.98 and action.model not in non_model_actions:
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
        f"next_state_claim_status={selected.prediction.predicted_next_state.claim_status_after_action}, "
        f"uncertainty_reduction={selected.prediction.predicted_next_state.uncertainty_reduction}, "
        f"research_value_per_compute={selected_breakdown.get('research_value_per_compute')}, "
        f"recommendation={selected.prediction.recommendation}, risks={risk_text}. "
        f"Claim delta={selected.prediction.expected_claim_delta}; "
        f"why_not_classic={selected.prediction.why_not_classic_autoresearch}. "
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
        world_model_decision_basis=(
            f"expected_claim_delta={selected.prediction.expected_claim_delta}; "
            f"expected_claim_resolution={selected.prediction.expected_claim_resolution:.2f}; "
            f"expected_value_of_information={selected.prediction.expected_value_of_information:.2f}; "
            f"compute_waste_risk={selected.prediction.compute_waste_risk:.2f}; "
            f"protocol_risks={', '.join(selected.prediction.protocol_risks)}"
        ),
        classic_counterfactual_action_id=next(
            (
                cp.action.action_id
                for _score, cp, _reasons, _world, _selector, _breakdown in scored
                if cp.action.model not in {"top_model_verification", "verification_sweep", "protocol_probe", "stop_and_report"}
            ),
            None,
        ),
        why_world_model_mattered=selected.prediction.why_not_classic_autoresearch,
        causal_reason=causal_reason,
        rejected_candidates=rejected,
    ), safety_validation
