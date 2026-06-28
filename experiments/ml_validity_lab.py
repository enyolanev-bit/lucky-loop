#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_sklearn import get_dataset


def _model(name: str, seed: int):
    if name == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    if name == "forest":
        return RandomForestClassifier(n_estimators=160, max_depth=None, random_state=seed, n_jobs=-1)
    if name == "svc":
        return make_pipeline(StandardScaler(), SVC(C=2.0, kernel="rbf", gamma="scale", random_state=seed))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, random_state=seed)
    raise ValueError(f"unknown model: {name}")


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _fit_eval(X_train, X_test, y_train, y_test, model_name: str, seed: int) -> dict:
    clf = _model(model_name, seed)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return _metrics(y_test, pred)


def _ordered_block_split(X, y, seed: int, test_size: float = 0.25):
    n = len(y)
    block = max(2, int(round(n * test_size)))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, n - block + 1)))
    test_idx = np.arange(start, min(start + block, n))
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False
    return X[train_mask], X[test_idx], y[train_mask], y[test_idx]


def _random_split(X, y, seed: int, test_size: float = 0.25):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=None)


def _summarize(rows: list[dict], primary_metric: str) -> tuple[dict, float | None, float | None, float | None, str | None]:
    by_condition: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(primary_metric)
        if value is not None:
            by_condition.setdefault(row["condition"], []).append(float(value))
    summary = {}
    for condition, values in by_condition.items():
        arr = np.array(values, dtype=float)
        summary[condition] = {
            f"mean_{primary_metric}": float(arr.mean()),
            f"std_{primary_metric}": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(len(arr)),
        }
    if len(summary) < 2:
        return summary, None, None, None, next(iter(summary), None)
    ordered = sorted(summary, key=lambda key: summary[key][f"mean_{primary_metric}"], reverse=True)
    best, runner_up = ordered[0], ordered[1]
    effect = summary[best][f"mean_{primary_metric}"] - summary[runner_up][f"mean_{primary_metric}"]
    best_values = by_condition[best]
    seed_noise = max(best_values) - min(best_values) if len(best_values) > 1 else None
    ratio = effect / seed_noise if seed_noise and seed_noise > 0 else None
    return summary, float(effect), None if seed_noise is None else float(seed_noise), None if ratio is None else float(ratio), best


def _inspect_dataset(X, y, dataset: str) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    return {
        "status": "success",
        "study_id": "inspect_dataset",
        "protocol_id": "inspect_dataset",
        "dataset": dataset,
        "primary_metric": "balanced_accuracy",
        "conditions": ["dataset_profile"],
        "runs": [],
        "summary": {
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]) if len(X.shape) > 1 else 1,
            "classes": [str(item) for item in classes.tolist()],
            "class_counts": {str(cls): int(count) for cls, count in zip(classes.tolist(), counts.tolist())},
        },
        "effect_size": None,
        "seed_noise": None,
        "effect_to_noise_ratio": None,
        "best_condition": None,
        "protocol_warnings": [],
    }


def _split_validity(X, y, seeds: list[int], action: str) -> tuple[list[dict], list[str], str]:
    rows = []
    for seed in seeds:
        conditions = ["random_split", "blocked_split"]
        if action == "run_random_split_baseline":
            conditions = ["random_split"]
        elif action == "run_blocked_split_protocol":
            conditions = ["blocked_split"]
        for condition in conditions:
            if condition == "random_split":
                X_train, X_test, y_train, y_test = _random_split(X, y, seed)
            else:
                X_train, X_test, y_train, y_test = _ordered_block_split(X, y, seed)
            metrics = _fit_eval(X_train, X_test, y_train, y_test, "linear", seed)
            rows.append({"condition": condition, "seed": seed, "model": "linear", **metrics})
    return rows, ["blocked split is a proxy for temporal or grouped holdout; it is stricter than random split"], "balanced_accuracy"


def _leakage_probe(X, y, seeds: list[int], action: str) -> tuple[list[dict], list[str], str]:
    rows = []
    y_float = np.asarray(y, dtype=float).reshape(-1, 1)
    X_leaky = np.concatenate([X, y_float], axis=1)
    for seed in seeds:
        if action in {"run_baseline", "run_negative_control"}:
            X_train, X_test, y_train, y_test = _random_split(X, y, seed)
            rows.append({"condition": "proper_pipeline", "seed": seed, "model": "linear", **_fit_eval(X_train, X_test, y_train, y_test, "linear", seed)})
        else:
            X_train, X_test, y_train, y_test = _random_split(X, y, seed)
            rows.append({"condition": "proper_pipeline", "seed": seed, "model": "linear", **_fit_eval(X_train, X_test, y_train, y_test, "linear", seed)})
            XL_train, XL_test, yl_train, yl_test = _random_split(X_leaky, y, seed)
            rows.append({"condition": "leaky_preprocessing", "seed": seed, "model": "linear", **_fit_eval(XL_train, XL_test, yl_train, yl_test, "linear", seed)})
    warnings = []
    if action not in {"run_baseline", "run_negative_control"}:
        warnings = ["data_leakage: label-derived feature was intentionally included before the split as a controlled invalid protocol"]
    return rows, warnings, "balanced_accuracy"


def _imbalanced_metric_probe(X, y, seeds: list[int], action: str) -> tuple[list[dict], list[str], str]:
    rows = []
    classes, counts = np.unique(y, return_counts=True)
    majority = classes[np.argmax(counts)]
    minority = classes[np.argmin(counts)]
    majority_idx = np.flatnonzero(y == majority)
    minority_idx = np.flatnonzero(y == minority)
    rng = np.random.default_rng(123)
    keep_minority = rng.choice(minority_idx, size=max(8, len(minority_idx) // 5), replace=False)
    keep = np.concatenate([majority_idx, keep_minority])
    keep.sort()
    X_imb, y_imb = X[keep], y[keep]
    for seed in seeds:
        X_train, X_test, y_train, y_test = _random_split(X_imb, y_imb, seed)
        metrics = _fit_eval(X_train, X_test, y_train, y_test, "linear", seed)
        rows.append({"condition": "balanced_metrics_audit", "seed": seed, "model": "linear", **metrics})
    warnings = [] if action == "run_baseline" else ["metric_misuse: accuracy can remain high while balanced_accuracy and F1 reveal minority-class weakness"]
    return rows, warnings, "balanced_accuracy"


def _seed_variance(X, y, seeds: list[int], action: str) -> tuple[list[dict], list[str], str]:
    rows = []
    models = ["linear"] if action == "run_baseline" else ["linear", "forest", "svc"]
    for seed in seeds:
        X_train, X_test, y_train, y_test = _random_split(X, y, seed)
        for model_name in models:
            rows.append({"condition": model_name, "seed": seed, "model": model_name, **_fit_eval(X_train, X_test, y_train, y_test, model_name, seed)})
    return rows, ["single-run winner is observation-only until repeated-seed effect exceeds seed noise"], "balanced_accuracy"


def _small_data_complexity(X, y, seeds: list[int], action: str) -> tuple[list[dict], list[str], str]:
    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = _random_split(X, y, seed)
        rng = np.random.default_rng(seed)
        take = rng.choice(len(y_train), size=max(20, int(round(len(y_train) * 0.35))), replace=False)
        X_small, y_small = X_train[take], y_train[take]
        pairs = [("simple_baseline", "linear")] if action == "run_baseline" else [("simple_baseline", "linear"), ("complex_model", "hgb")]
        for condition, model_name in pairs:
            rows.append({"condition": condition, "seed": seed, "model": model_name, **_fit_eval(X_small, X_test, y_small, y_test, model_name, seed)})
    return rows, ["small-data protocol: complex model gains must exceed seed noise before a superiority claim"], "balanced_accuracy"


def run_study(study: str, dataset: str, seeds: list[int], action: str) -> dict:
    X, y = get_dataset(dataset)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if action == "inspect_dataset":
        return _inspect_dataset(X, y, dataset)
    if action == "run_replication":
        seeds = sorted(set(seeds + [10, 11, 12, 13]))
    if action == "run_negative_control":
        rng = np.random.default_rng(999)
        y = rng.permutation(y)
    if study == "random_vs_blocked_split":
        rows, warnings, primary = _split_validity(X, y, seeds, action)
    elif study == "proper_vs_leaky_preprocessing":
        rows, warnings, primary = _leakage_probe(X, y, seeds, action)
    elif study == "accuracy_vs_balanced_metrics":
        rows, warnings, primary = _imbalanced_metric_probe(X, y, seeds, action)
    elif study == "single_run_vs_repeated_seeds":
        rows, warnings, primary = _seed_variance(X, y, seeds, action)
    elif study == "simple_vs_complex_small_data":
        rows, warnings, primary = _small_data_complexity(X, y, seeds, action)
    else:
        raise ValueError(f"unsupported protocol study: {study}")
    summary, effect, seed_noise, ratio, best = _summarize(rows, primary)
    return {
        "status": "success",
        "study_id": study,
        "protocol_id": study,
        "lab_action": action,
        "dataset": dataset,
        "primary_metric": primary,
        "conditions": sorted({row["condition"] for row in rows}),
        "runs": rows,
        "summary": summary,
        "effect_size": effect,
        "seed_noise": seed_noise,
        "effect_to_noise_ratio": ratio,
        "best_condition": best,
        "protocol_warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", default="run_main_protocol")
    parser.add_argument("--study", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    t0 = time.perf_counter()
    payload = run_study(args.study, args.dataset, args.seeds, args.action)
    payload["runtime_seconds"] = round(time.perf_counter() - t0, 4)
    payload["artifacts"] = []
    if args.out_dir:
        out_dir = Path(args.out_dir)
        runs_dir = out_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"experiment_{args.step:03d}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        payload["artifacts"].append(str(path))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
