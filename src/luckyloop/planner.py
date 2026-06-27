from __future__ import annotations

import json

from .schemas import CandidatePrediction, DecisionTrace, ExperimentTrace, ProposedAction, RejectedCandidate, ResearchState, TaskSpec
from .simulator import predict


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


def _model_command(task: TaskSpec, model: str) -> tuple[str, dict]:
    params = {"dataset": task.dataset}
    cmd = f"python experiments/train_sklearn.py --dataset {task.dataset} --model {model}"
    if model == "logistic_regression":
        params.update({"scale": True})
        cmd += " --scale"
    elif model == "svc":
        params.update({"scale": True, "C": 2.0, "kernel": "rbf"})
        cmd += " --scale --C 2.0"
    elif model == "random_forest":
        params.update({"n_estimators": 300})
        cmd += " --n-estimators 300"
    elif model == "gradient_boosting":
        params.update({"n_estimators": 150, "learning_rate": 0.05})
        cmd += " --n-estimators 150 --learning-rate 0.05"
    elif model == "hist_gradient_boosting":
        params.update({"n_estimators": 150, "learning_rate": 0.08})
        cmd += " --n-estimators 150 --learning-rate 0.08"
    return cmd, params


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


def generate_candidates(task: TaskSpec, state: ResearchState, traces: list[ExperimentTrace], seen: set[str]) -> list[ProposedAction]:
    candidates = []
    for model in task.models:
        cmd, params = _model_command(task, model)
        candidates.append(_candidate(f"action_{model}", cmd, model, params))

    for index, _sweep in enumerate(task.sweeps):
        candidates.append(_sweep_candidate(task, index))

    if not traces:
        candidates.insert(0, initial_action(task))

    unseen = [c for c in candidates if action_key(c) not in seen]
    return unseen[:8]


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


def _score_candidate(candidate_prediction: CandidatePrediction, traces: list[ExperimentTrace]) -> tuple[int, list[str], list[str], list[str]]:
    action = candidate_prediction.action
    prediction = candidate_prediction.prediction
    risks = " ".join(prediction.risks).lower()
    rationale = prediction.rationale.lower()
    score = 0
    reasons = []
    world_reasons = []
    selector_reasons = []

    if prediction.recommendation == "run":
        score += 30
        reasons.append("world model recommended run")
    elif prediction.recommendation == "modify":
        score += 12
        reasons.append("world model recommended modification, so the action remains informative but lower priority")
        world_reasons.append("world model recommended modification")
    else:
        score -= 20
        reasons.append("world model recommended skip")
        world_reasons.append("world model recommended skip")

    signal = f"{prediction.action_specific_signal} {prediction.claim_risk} {risks} {rationale}".lower()
    if any(word in signal for word in ["scal", "seed", "variance", "noise", "robust", "leak", "protocol", "metric", "overfit"]):
        world_reasons.append("prediction contained an action-specific research signal")

    if action.model == "logistic_regression" and not action.params.get("scale") and not traces:
        score += 80
        reasons.append("first run should establish the unscaled baseline before interventions")
        selector_reasons.append("first run should establish the unscaled baseline before interventions")

    if action.model == "logistic_regression" and action.params.get("scale") and not traces:
        score -= 40
        reasons.append("scaled baseline is deferred until after an unscaled control")
        selector_reasons.append("scaled baseline is deferred until after an unscaled control")

    if action.model == "logistic_regression" and action.params.get("scale") and traces:
        first = traces[0]
        if first.proposed_action.model == "logistic_regression" and not first.proposed_action.params.get("scale"):
            score += 45
            reasons.append("after an unscaled logistic baseline, scaling is the direct world-model intervention to test")
            selector_reasons.append("after an unscaled logistic baseline, scaling is the direct intervention to test")

    if action.model == "verification_sweep" and _best_accuracy(traces) >= 0.98:
        score += 35
        reasons.append("high single-run score needs robustness verification before a strong claim")
        selector_reasons.append("high single-run score needs robustness verification before a strong claim")
    if action.model == "verification_sweep" and ("seed variance" in risks or "non-robust" in risks or "noise" in risks):
        score += 25
        reasons.append("world model predicted robustness or seed-variance risk")
        world_reasons.append("world model predicted robustness or seed-variance risk")
    if action.model == "verification_sweep" and len(traces) < 4:
        score -= 30
        reasons.append("defer verifier sweep until several single-model baselines exist")
        selector_reasons.append("defer verifier sweep until several single-model baselines exist")

    if action.model == "svc" and any(t.proposed_action.model == "random_forest" and not t.comparison.metric_match for t in traces):
        score += 30
        reasons.append("tree prediction missed, so a scaled margin model tests a different hypothesis")
        selector_reasons.append("tree prediction missed, so a scaled margin model tests a different hypothesis")

    if action.model == "random_forest" and _best_accuracy(traces) >= 0.98:
        score += 35
        reasons.append("strong linear baseline makes a different inductive bias useful")
        selector_reasons.append("strong linear baseline makes a different inductive bias useful")

    if action.model == "gradient_boosting":
        score += 5
        reasons.append("ensemble baseline is useful as a late comparison")
        selector_reasons.append("ensemble baseline is useful as a late comparison")

    if "overfit" in risks or "overfit" in rationale:
        score -= 4
        reasons.append("world model predicted overfitting risk")
        world_reasons.append("world model predicted overfitting risk")

    return score, reasons, world_reasons, selector_reasons


def select_candidate(state: ResearchState, predictions: list[CandidatePrediction], traces: list[ExperimentTrace]) -> tuple[CandidatePrediction, DecisionTrace]:
    if not predictions:
        raise ValueError("select_candidate requires at least one candidate prediction")

    scored = []
    for cp in predictions:
        score, reasons, world_reasons, selector_reasons = _score_candidate(cp, traces)
        scored.append((score, cp, reasons, world_reasons, selector_reasons))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected_score, selected, selected_reasons, selected_world_reasons, selected_selector_reasons = scored[0]

    rejected = []
    for score, candidate_prediction, reasons, _world_reasons, _selector_reasons in scored[1:]:
        rejected.append(
            RejectedCandidate(
                action=candidate_prediction.action,
                reason=f"score={score}; " + "; ".join(reasons),
            )
        )

    risk_text = ", ".join(selected.prediction.risks) or "no specific risk"
    world_used = bool(selected_world_reasons)
    selector_used = bool(selected_selector_reasons)
    if world_used and selector_used:
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
        f"World model predicted {selected.prediction.expected_metric}, runtime {selected.prediction.expected_runtime_seconds}, "
        f"recommendation={selected.prediction.recommendation}, risks={risk_text}. "
        f"Causal signal type: {causal_signal_type}."
    )

    return selected, DecisionTrace(
        selected_action=selected.action,
        world_model_signal_used=world_used,
        selector_policy_signal_used=selector_used,
        causal_signal_type=causal_signal_type,
        causal_reason=causal_reason,
        rejected_candidates=rejected,
    )
