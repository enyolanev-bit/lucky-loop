#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from luckyloop.schemas import ExperimentTrace
from luckyloop.tasks import ROOT


TASKS = ["breast_cancer_accuracy", "wine_accuracy", "digits_accuracy"]


def _failures_for_task(task: str, runs_dir: Path, require_qwen: bool) -> list[str]:
    failures: list[str] = []
    trace_paths = sorted(runs_dir.glob("run_*.json"))
    if not trace_paths:
        return [f"{runs_dir.relative_to(ROOT)} has no run_*.json traces"]

    saw_verification = False
    saw_claim_gate = False
    saw_qwen = False
    for path in trace_paths:
        trace = ExperimentTrace.model_validate_json(path.read_text(encoding="utf-8"))
        label = f"{path.relative_to(ROOT)}"
        if not trace.state_before:
            failures.append(f"{label} missing state_before")
        if not trace.candidate_actions:
            failures.append(f"{label} missing candidate_actions")
        if not trace.decision_trace:
            failures.append(f"{label} missing decision_trace")
        if not trace.actual_result:
            failures.append(f"{label} missing actual_result")
        if trace.verification:
            saw_verification = True
            if trace.claim_ledger_updates:
                saw_claim_gate = True
        if any(cp.source == "qwen_agentworld" for cp in trace.candidate_predictions):
            saw_qwen = True
    if "lucky_loop_full" in str(runs_dir) and require_qwen and not saw_qwen:
        failures.append(f"{runs_dir.relative_to(ROOT)} has no qwen_agentworld candidate predictions")
    if "classic_autoresearch" not in str(runs_dir) and not saw_verification:
        failures.append(f"{runs_dir.relative_to(ROOT)} has no verifier run")
    if "classic_autoresearch" not in str(runs_dir) and not saw_claim_gate:
        failures.append(f"{runs_dir.relative_to(ROOT)} has no claim ledger updates")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-qwen", action="store_true")
    parser.add_argument("--check-ablations", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    for task in TASKS:
        failures.extend(_failures_for_task(task, ROOT / "runs" / task, args.require_qwen))

    required_reports = [
        ROOT / "reports" / "benchmark_summary.md",
        ROOT / "reports" / "pitch_backend_summary.md",
    ]
    if args.check_ablations:
        required_reports.extend(
            [
                ROOT / "reports" / "ablations" / "world_model_ablation.md",
                ROOT / "reports" / "ablations" / "classic_vs_lucky_loop.md",
                ROOT / "reports" / "ablations" / "world_model_ablation.json",
            ]
        )
        for policy in ["classic_autoresearch", "classic_verified", "lucky_loop_full"]:
            for task in TASKS:
                failures.extend(
                    _failures_for_task(
                        task,
                        ROOT / "runs" / "ablations" / policy / task,
                        args.require_qwen and policy == "lucky_loop_full",
                    )
                )

    for report in required_reports:
        if not report.exists():
            failures.append(f"missing report {report.relative_to(ROOT)}")

    ablation_json = ROOT / "reports" / "ablations" / "world_model_ablation.json"
    if args.check_ablations and ablation_json.exists():
        payload = json.loads(ablation_json.read_text(encoding="utf-8"))
        rows = payload.get("rows") or []
        expected = 9
        if len(rows) < expected:
            failures.append(f"ablation json has {len(rows)} rows, expected at least {expected}")

    if failures:
        print("Artifact validation failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Artifact validation passed.")


if __name__ == "__main__":
    main()
