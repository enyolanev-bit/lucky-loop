from __future__ import annotations

import json

from .schemas import CandidatePrediction, DecisionTrace, ExperimentTrace, ProposedAction, RejectedCandidate, ResearchState
from .simulator import predict


def action_key(action: ProposedAction) -> str:
    return f"{action.model}:{json.dumps(action.params, sort_keys=True)}"


def with_action_id(action: ProposedAction, action_id: str) -> ProposedAction:
    return ProposedAction(action_id=action_id, command=action.command, model=action.model, params=action.params)


def initial_action() -> ProposedAction:
    return ProposedAction(
        action_id="action_001",
        command="python experiments/train_sklearn.py --dataset breast_cancer --model logistic_regression",
        model="logistic_regression",
        params={"scale": False},
    )


def initial_hypothesis() -> str:
    return "Establish a simple linear baseline before spending search budget."


def _candidate(action_id: str, command: str, model: str, params: dict) -> ProposedAction:
    return ProposedAction(action_id=action_id, command=command, model=model, params=params)


def generate_candidates(state: ResearchState, traces: list[ExperimentTrace], seen: set[str]) -> list[ProposedAction]:
    candidates = [
        _candidate(
            "action_scaled_logreg",
            "python experiments/train_sklearn.py --dataset breast_cancer --model logistic_regression --scale",
            "logistic_regression",
            {"scale": True},
        ),
        _candidate(
            "action_random_forest",
            "python experiments/train_sklearn.py --dataset breast_cancer --model random_forest --n-estimators 300",
            "random_forest",
            {"n_estimators": 300},
        ),
        _candidate(
            "action_svc_scaled",
            "python experiments/train_sklearn.py --dataset breast_cancer --model svc --scale --C 2.0",
            "svc",
            {"scale": True, "C": 2.0, "kernel": "rbf"},
        ),
        _candidate(
            "action_noisy_sweep",
            "python experiments/sweep_sklearn.py --dataset breast_cancer --model logistic_regression --scale --sweep-param C --values 0.1 1.0 10.0 --seeds 0 1 2 3 --label-noise 0.08",
            "verification_sweep",
            {
                "base_model": "logistic_regression",
                "scale": True,
                "sweep_param": "C",
                "values": [0.1, 1.0, 10.0],
                "seeds": [0, 1, 2, 3],
                "label_noise": 0.08,
            },
        ),
        _candidate(
            "action_gradient_boosting",
            "python experiments/train_sklearn.py --dataset breast_cancer --model gradient_boosting --n-estimators 150 --learning-rate 0.05",
            "gradient_boosting",
            {"n_estimators": 150, "learning_rate": 0.05},
        ),
    ]

    if not traces:
        candidates.insert(0, initial_action())

    unseen = [c for c in candidates if action_key(c) not in seen]
    return unseen[:5]


def prediction_source(prediction, simulator_configured: bool) -> str:
    if not simulator_configured:
        return "heuristic_fallback"
    text = " ".join([prediction.rationale, *prediction.risks]).lower()
    return "heuristic_fallback" if "fallback" in text or "heuristic" in text else "qwen_agentworld"


def predict_candidates(state: ResearchState, candidates: list[ProposedAction], simulator_configured: bool) -> list[CandidatePrediction]:
    state_text = state.model_dump_json(indent=2)
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


def _score_candidate(candidate_prediction: CandidatePrediction, traces: list[ExperimentTrace]) -> tuple[int, list[str]]:
    action = candidate_prediction.action
    prediction = candidate_prediction.prediction
    risks = " ".join(prediction.risks).lower()
    rationale = prediction.rationale.lower()
    score = 0
    reasons = []

    if prediction.recommendation == "run":
        score += 30
        reasons.append("world model recommended run")
    elif prediction.recommendation == "modify":
        score += 12
        reasons.append("world model recommended modification, so the action remains informative but lower priority")
    else:
        score -= 20
        reasons.append("world model recommended skip")

    if action.model == "logistic_regression" and not action.params.get("scale") and not traces:
        score += 80
        reasons.append("first run should establish the unscaled baseline before interventions")

    if action.model == "logistic_regression" and action.params.get("scale") and not traces:
        score -= 40
        reasons.append("scaled baseline is deferred until after an unscaled control")

    if action.model == "logistic_regression" and action.params.get("scale") and traces:
        first = traces[0]
        if first.proposed_action.model == "logistic_regression" and not first.proposed_action.params.get("scale"):
            score += 45
            reasons.append("after an unscaled logistic baseline, scaling is the direct world-model intervention to test")

    if action.model == "verification_sweep" and _best_accuracy(traces) >= 0.98:
        score += 35
        reasons.append("high single-run score needs robustness verification before a strong claim")
    if action.model == "verification_sweep" and ("seed variance" in risks or "non-robust" in risks or "noise" in risks):
        score += 25
        reasons.append("world model predicted robustness or seed-variance risk")
    if action.model == "verification_sweep" and len(traces) < 4:
        score -= 30
        reasons.append("defer verifier sweep until several single-model baselines exist")

    if action.model == "svc" and any(t.proposed_action.model == "random_forest" and not t.comparison.metric_match for t in traces):
        score += 30
        reasons.append("tree prediction missed, so a scaled margin model tests a different hypothesis")

    if action.model == "random_forest" and _best_accuracy(traces) >= 0.98:
        score += 35
        reasons.append("strong linear baseline makes a different inductive bias useful")

    if action.model == "gradient_boosting":
        score += 5
        reasons.append("ensemble baseline is useful as a late comparison")

    if "overfit" in risks or "overfit" in rationale:
        score -= 4
        reasons.append("world model predicted overfitting risk")

    return score, reasons


def select_candidate(state: ResearchState, predictions: list[CandidatePrediction], traces: list[ExperimentTrace]) -> tuple[CandidatePrediction, DecisionTrace]:
    if not predictions:
        raise ValueError("select_candidate requires at least one candidate prediction")

    scored = []
    for cp in predictions:
        score, reasons = _score_candidate(cp, traces)
        scored.append((score, cp, reasons))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected_score, selected, selected_reasons = scored[0]

    rejected = []
    for score, candidate_prediction, reasons in scored[1:]:
        rejected.append(
            RejectedCandidate(
                action=candidate_prediction.action,
                reason=f"score={score}; " + "; ".join(reasons),
            )
        )

    risk_text = ", ".join(selected.prediction.risks) or "no specific risk"
    causal_reason = (
        f"Selected {selected.action.model} because score={selected_score}; "
        f"{'; '.join(selected_reasons)}. "
        f"World model predicted {selected.prediction.expected_metric}, runtime {selected.prediction.expected_runtime_seconds}, "
        f"recommendation={selected.prediction.recommendation}, risks={risk_text}."
    )

    return selected, DecisionTrace(
        selected_action=selected.action,
        world_model_signal_used=True,
        causal_reason=causal_reason,
        rejected_candidates=rejected,
    )
