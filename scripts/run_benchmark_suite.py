#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from luckyloop.loop import run
from luckyloop.tasks import ROOT, load_task


TASKS = [
    "configs/tasks/breast_cancer_accuracy.json",
    "configs/tasks/wine_accuracy.json",
    "configs/tasks/digits_accuracy.json",
]


def _actual_metric(trace):
    if trace.actual_result.accuracy is not None:
        return trace.actual_result.accuracy
    best = trace.actual_result.raw.get("best") or {}
    metric = trace.actual_result.raw.get("metric", "accuracy")
    value = best.get(f"mean_{metric}")
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planner-mode",
        choices=["llm", "agent_handoff", "agent_command", "operator_driven", "replay", "selector"],
        default="replay",
    )
    parser.add_argument("--agent-backend", default=None)
    parser.add_argument("--agent-io-dir", default="agent_io")
    parser.add_argument("--agent-timeout-seconds", type=int, default=900)
    parser.add_argument("--agent-poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    rows = []
    for task_path in TASKS:
        task = load_task(task_path)
        traces = run(
            task=task,
            output_namespace=task.task_id,
            planner_mode=args.planner_mode,
            agent_backend=args.agent_backend,
            agent_io_dir=args.agent_io_dir,
            agent_timeout_seconds=args.agent_timeout_seconds,
            agent_poll_seconds=args.agent_poll_seconds,
        )
        metric_traces = [
            t
            for t in traces
            if _actual_metric(t) is not None
            and t.proposed_action.model not in {"verification_sweep", "top_model_verification"}
        ]
        best = max(metric_traces, key=lambda t: _actual_metric(t) or -1, default=None)
        ledger_path = ROOT / "reports" / task.task_id / "claim_ledger.json"
        blocked = supported = 0
        if ledger_path.exists():
            import json

            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            summary = payload.get("summary") or {}
            blocked = summary.get("blocked", 0)
            supported = summary.get("supported", 0) + summary.get("strongly_supported", 0)
        rows.append(
            {
                "task": task.task_id,
                "runs": len(traces),
                "best_model": best.proposed_action.model if best else "",
                "best_metric": _actual_metric(best) if best else None,
                "prediction_misses": sum(1 for t in traces if t.comparison.unexpected_events),
                "top_model_verifications": sum(1 for t in traces if t.proposed_action.model == "top_model_verification"),
                "claims_blocked": blocked,
                "supported_claims": supported,
            }
        )

    report = [
        "# Lucky Loop Benchmark Summary",
        "",
        f"These benchmark tasks use real sklearn datasets, real training commands, real multi-seed sweeps, and planner_mode={args.planner_mode}.",
        "",
        "| Task | Runs | Best model | Best metric | Top-model verifications | Prediction misses | Claims blocked | Supported claims |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best_metric = "" if row["best_metric"] is None else f"{row['best_metric']:.4f}"
        report.append(
            f"| {row['task']} | {row['runs']} | {row['best_model']} | {best_metric} | "
            f"{row['top_model_verifications']} | {row['prediction_misses']} | "
            f"{row['claims_blocked']} | {row['supported_claims']} |"
        )
    out = ROOT / "reports" / "benchmark_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote benchmark summary to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
