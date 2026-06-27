#!/usr/bin/env python3
from __future__ import annotations

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
    rows = []
    for task_path in TASKS:
        task = load_task(task_path)
        traces = run(task=task, output_namespace=task.task_id)
        metric_traces = [
            t
            for t in traces
            if _actual_metric(t) is not None and t.proposed_action.model != "verification_sweep"
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
                "claims_blocked": blocked,
                "supported_claims": supported,
            }
        )

    report = [
        "# Lucky Loop Benchmark Summary",
        "",
        "These benchmark tasks use real sklearn datasets, real training commands, and real multi-seed sweeps.",
        "",
        "| Task | Runs | Best model | Best metric | Prediction misses | Claims blocked | Supported claims |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        best_metric = "" if row["best_metric"] is None else f"{row['best_metric']:.4f}"
        report.append(
            f"| {row['task']} | {row['runs']} | {row['best_model']} | {best_metric} | "
            f"{row['prediction_misses']} | {row['claims_blocked']} | {row['supported_claims']} |"
        )
    out = ROOT / "reports" / "benchmark_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote benchmark summary to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
