from __future__ import annotations

import shlex

from .schemas import ExperimentTrace, ProposedAction, TaskSpec, TopModelCandidate, TopModelSummary


def model_key(model: str, params: dict) -> str:
    parts = [model]
    if params.get("scale"):
        parts.append("scaled")
    if params.get("C") is not None:
        parts.append(f"C={params['C']}")
    if params.get("kernel") is not None:
        parts.append(f"kernel={params['kernel']}")
    if params.get("n_estimators") is not None:
        parts.append(f"n={params['n_estimators']}")
    if params.get("max_depth") is not None:
        parts.append(f"depth={params['max_depth']}")
    if params.get("learning_rate") is not None:
        parts.append(f"lr={params['learning_rate']}")
    return "_".join(str(p).replace(" ", "") for p in parts)


def _model_arg(candidate: TopModelCandidate) -> str:
    params = candidate.params
    pieces = [candidate.model]
    options = []
    if params.get("scale"):
        options.append("scale=true")
    if params.get("C") is not None:
        options.append(f"C={params['C']}")
    if params.get("kernel") is not None:
        options.append(f"kernel={params['kernel']}")
    if params.get("n_estimators") is not None:
        options.append(f"n_estimators={params['n_estimators']}")
    if params.get("max_depth") is not None:
        options.append(f"max_depth={params['max_depth']}")
    if params.get("learning_rate") is not None:
        options.append(f"learning_rate={params['learning_rate']}")
    if options:
        pieces.append(",".join(options))
    return ":".join(pieces)


def detect_top_models(
    traces: list[ExperimentTrace],
    metric: str = "accuracy",
    top_k: int = 3,
    margin: float = 0.01,
    min_single_runs: int = 3,
) -> TopModelSummary:
    candidates: list[TopModelCandidate] = []
    verified_keys: set[str] = set()
    for trace in traces:
        if trace.proposed_action.model == "top_model_verification":
            verified_keys.update(trace.actual_result.raw.get("verified_models") or [])
            continue
        if trace.proposed_action.model == "verification_sweep":
            continue
        value = getattr(trace.actual_result, metric, None)
        if value is None:
            continue
        params = dict(trace.proposed_action.params)
        key = model_key(trace.proposed_action.model, params)
        candidates.append(
            TopModelCandidate(
                run_id=trace.run_id,
                model=trace.proposed_action.model,
                model_key=key,
                metric=float(value),
                params=params,
            )
        )

    if len(candidates) < min_single_runs:
        return TopModelSummary(
            top_models=sorted(candidates, key=lambda c: c.metric, reverse=True),
            needs_robustness_verification=False,
            reason=f"Need at least {min_single_runs} single-run model results before top-model verification.",
        )

    ranked = sorted(candidates, key=lambda c: c.metric, reverse=True)
    best = ranked[0]
    close = [c for c in ranked if best.metric - c.metric <= margin]
    selected = ranked[:top_k]
    for candidate in close:
        if candidate.model_key not in {c.model_key for c in selected}:
            selected.append(candidate)
    selected = selected[: max(top_k, 2)]
    top_gap = round(best.metric - ranked[1].metric, 6) if len(ranked) > 1 else None
    selected_keys = {c.model_key for c in selected}
    already_verified = selected_keys.issubset(verified_keys)
    needs_verification = not already_verified and len(selected) >= 2
    if already_verified:
        reason = "Top observed models already have a multi-seed verification run."
    elif top_gap == 0:
        reason = "Top models are tied on single-split evidence; robust best-model claim requires multi-seed verification."
    elif top_gap is not None and top_gap <= margin:
        reason = f"Top-model gap {top_gap:.4f} is within margin {margin:.4f}; verify before claiming a winner."
    else:
        reason = "Best observed model is based on a single split; verify top models before making a robust claim."

    return TopModelSummary(
        best_observed_model=best.model_key,
        best_observed_metric=best.metric,
        top_models=selected,
        top_gap=top_gap,
        needs_robustness_verification=needs_verification,
        reason=reason,
        verification_action_key=":".join(sorted(selected_keys)) if selected_keys else None,
    )


def build_top_model_verification_action(task: TaskSpec, summary: TopModelSummary) -> ProposedAction | None:
    if not summary.needs_robustness_verification or len(summary.top_models) < 2:
        return None
    model_args = [_model_arg(candidate) for candidate in summary.top_models]
    seeds = [0, 1, 2, 3, 4]
    command = (
        f"python experiments/compare_models_sklearn.py --dataset {task.dataset} "
        f"--models {' '.join(shlex.quote(arg) for arg in model_args)} "
        f"--seeds {' '.join(str(seed) for seed in seeds)}"
    )
    return ProposedAction(
        action_id="action_verify_top_models",
        command=command,
        model="top_model_verification",
        params={
            "dataset": task.dataset,
            "models": [candidate.model_dump() for candidate in summary.top_models],
            "seeds": seeds,
            "reason": summary.reason,
            "verification_action_key": summary.verification_action_key,
        },
    )
