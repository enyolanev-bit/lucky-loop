#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_model_spec(spec: str) -> tuple[str, dict[str, str]]:
    model, _, option_text = spec.partition(":")
    options: dict[str, str] = {}
    if option_text:
        for item in option_text.split(","):
            if not item:
                continue
            key, _, value = item.partition("=")
            options[key.strip()] = value.strip()
    return model.strip(), options


def label_for(model: str, options: dict[str, str]) -> str:
    parts = [model]
    if options.get("scale", "").lower() == "true":
        parts.append("scaled")
    for key in ("C", "kernel", "n_estimators", "max_depth", "learning_rate"):
        if options.get(key) not in {None, ""}:
            short = {
                "n_estimators": "n",
                "max_depth": "depth",
                "learning_rate": "lr",
            }.get(key, key)
            parts.append(f"{short}={options[key]}")
    return "_".join(parts)


def build_command(dataset: str, spec: str, seed: int) -> tuple[str, list[str]]:
    model, options = parse_model_spec(spec)
    cmd = [
        sys.executable,
        "experiments/train_sklearn.py",
        "--dataset",
        dataset,
        "--model",
        model,
        "--seed",
        str(seed),
    ]
    if options.get("scale", "").lower() == "true":
        cmd.append("--scale")
    if options.get("C"):
        cmd += ["--C", options["C"]]
    if options.get("kernel"):
        cmd += ["--kernel", options["kernel"]]
    if options.get("n_estimators"):
        cmd += ["--n-estimators", options["n_estimators"]]
    if options.get("max_depth"):
        cmd += ["--max-depth", options["max_depth"]]
    if options.get("learning_rate"):
        cmd += ["--learning-rate", options["learning_rate"]]
    return label_for(model, options), cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["breast_cancer", "wine", "digits"])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    t0 = time.perf_counter()
    rows = []
    failures = []
    labels = []
    for spec in args.models:
        label, _ = build_command(args.dataset, spec, args.seeds[0])
        labels.append(label)
        for seed in args.seeds:
            label, cmd = build_command(args.dataset, spec, seed)
            proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, timeout=120)
            try:
                raw = json.loads(proc.stdout[proc.stdout.find("{"):])
            except Exception:
                raw = {
                    "status": "failed",
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-1000:],
                    "stderr": proc.stderr[-1000:],
                }
            if proc.returncode != 0 or raw.get("status") != "success":
                failures.append(raw)
                continue
            rows.append(
                {
                    "config_key": label,
                    "label": label,
                    "sweep_param": "model",
                    "value": label,
                    "seed": seed,
                    "accuracy": raw.get("accuracy"),
                    "f1": raw.get("f1"),
                    "runtime_seconds": raw.get("runtime_seconds"),
                }
            )

    by_value = {}
    for label in labels:
        vals = [r["accuracy"] for r in rows if r["value"] == label and r.get("accuracy") is not None]
        f1s = [r["f1"] for r in rows if r["value"] == label and r.get("f1") is not None]
        if vals:
            by_value[label] = {
                "mean_accuracy": sum(vals) / len(vals),
                "min_accuracy": min(vals),
                "max_accuracy": max(vals),
                "mean_f1": sum(f1s) / len(f1s) if f1s else None,
                "n": len(vals),
            }

    ranked = sorted(by_value, key=lambda value: by_value[value]["mean_accuracy"], reverse=True)
    best_value = ranked[0] if ranked else None
    runner_up_value = ranked[1] if len(ranked) > 1 else None
    out = {
        "status": "success" if rows and not failures else "partial" if rows else "failed",
        "type": "top_model_verification",
        "metric": "accuracy",
        "dataset": args.dataset,
        "models": args.models,
        "verified_models": labels,
        "seeds": args.seeds,
        "runs": rows,
        "summary_by_value": by_value,
        "best": {"value": best_value, **by_value.get(best_value, {})} if best_value is not None else None,
        "runner_up": {"value": runner_up_value, **by_value.get(runner_up_value, {})} if runner_up_value is not None else None,
        "failures": failures,
        "runtime_seconds": round(time.perf_counter() - t0, 4),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
