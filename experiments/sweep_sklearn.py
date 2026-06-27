#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def build_command(args, seed: int, value: str) -> list[str]:
    cmd = [
        sys.executable,
        "experiments/train_sklearn.py",
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--seed",
        str(seed),
        "--test-size",
        str(args.test_size),
    ]
    if args.scale:
        cmd.append("--scale")
    if args.sweep_param == "C":
        cmd += ["--C", value]
    elif args.sweep_param == "n_estimators":
        cmd += ["--n-estimators", value]
    elif args.sweep_param == "max_depth":
        cmd += ["--max-depth", value]
    elif args.sweep_param == "learning_rate":
        cmd += ["--learning-rate", value]
    else:
        raise ValueError(f"unsupported sweep param: {args.sweep_param}")
    if args.label_noise:
        cmd += ["--label-noise", str(args.label_noise)]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="breast_cancer", choices=["breast_cancer", "wine", "digits"])
    p.add_argument("--model", default="logistic_regression", choices=["logistic_regression", "random_forest", "gradient_boosting", "hist_gradient_boosting", "svc"])
    p.add_argument("--scale", action="store_true")
    p.add_argument("--sweep-param", default="C", choices=["C", "n_estimators", "max_depth", "learning_rate"])
    p.add_argument("--values", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--label-noise", type=float, default=0.0)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    t0 = time.perf_counter()
    rows = []
    failures = []
    for value in args.values:
        for seed in args.seeds:
            cmd = build_command(args, seed, value)
            proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, timeout=120)
            try:
                raw = json.loads(proc.stdout[proc.stdout.find("{"):])
            except Exception:
                raw = {"status": "failed", "returncode": proc.returncode, "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:]}
            if proc.returncode != 0 or raw.get("status") != "success":
                failures.append(raw)
                continue
            rows.append(
                {
                    "config_key": f"{args.sweep_param}={value}",
                    "label": f"{args.sweep_param}={value}",
                    "sweep_param": args.sweep_param,
                    "value": value,
                    "seed": seed,
                    "accuracy": raw.get("accuracy"),
                    "f1": raw.get("f1"),
                    "runtime_seconds": raw.get("runtime_seconds"),
                }
            )

    by_value = {}
    for value in args.values:
        vals = [r["accuracy"] for r in rows if r["value"] == value and r.get("accuracy") is not None]
        if vals:
            by_value[value] = {
                "mean_accuracy": sum(vals) / len(vals),
                "min_accuracy": min(vals),
                "max_accuracy": max(vals),
                "n": len(vals),
            }
    best_value = max(by_value, key=lambda v: by_value[v]["mean_accuracy"]) if by_value else None
    out = {
        "status": "success" if rows and not failures else "partial" if rows else "failed",
        "metric": "accuracy",
        "dataset": args.dataset,
        "model": args.model,
        "sweep_param": args.sweep_param,
        "values": args.values,
        "seeds": args.seeds,
        "label_noise": args.label_noise,
        "runs": rows,
        "summary_by_value": by_value,
        "best": {"value": best_value, **by_value.get(best_value, {})} if best_value is not None else None,
        "failures": failures,
        "runtime_seconds": round(time.perf_counter() - t0, 4),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
