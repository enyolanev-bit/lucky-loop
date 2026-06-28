#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from luckyloop.benchmark_metrics import claimable_evidence_summary
from luckyloop.schemas import ExperimentTrace
from luckyloop.tasks import ROOT

from run_ablation_suite import TASK_PATHS


POLICIES = ["classic_autoresearch", "lucky_loop_full"]


def _load_traces(policy: str, task_id: str, budget: int | None) -> list[ExperimentTrace]:
    run_dir = ROOT / "runs" / "ablations" / policy / task_id
    traces = [
        ExperimentTrace.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("run_*.json"))
    ]
    return traces[:budget] if budget is not None else traces


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _policy_row(task_id: str, policy: str, traces: list[ExperimentTrace]) -> dict:
    summary = claimable_evidence_summary(traces)
    return {
        "task": task_id,
        "policy": policy,
        "budget_runs": len(traces),
        **summary,
    }


def _paired_row(task_id: str, classic: dict, lucky: dict) -> dict:
    classic_wasted = classic["wasted_score_chasing_runs"]
    lucky_wasted = lucky["wasted_score_chasing_runs"]
    classic_wasted_runtime = classic["wasted_score_chasing_runtime_seconds"]
    lucky_wasted_runtime = lucky["wasted_score_chasing_runtime_seconds"]
    # Signed delta (NOT clamped at 0): negative means Lucky wasted more than classic, so regressions
    # stay visible instead of being hidden as "0 saved".
    saved_runs = classic_wasted - lucky_wasted
    saved_runtime = classic_wasted_runtime - lucky_wasted_runtime
    strict_stop_saved_runs = lucky["saved_remaining_runs"] if lucky["qwen_skip_or_stop_recommended"] else 0
    strict_stop_saved_runtime = (
        lucky["saved_remaining_runtime_seconds"] if lucky["qwen_skip_or_stop_recommended"] else 0.0
    )
    return {
        "task": task_id,
        "budget_runs": min(classic["budget_runs"], lucky["budget_runs"]),
        "classic_wasted_score_chasing_runs": classic_wasted,
        "lucky_wasted_score_chasing_runs": lucky_wasted,
        "saved_score_chasing_runs": saved_runs,
        "classic_wasted_score_chasing_runtime_seconds": classic_wasted_runtime,
        "lucky_wasted_score_chasing_runtime_seconds": lucky_wasted_runtime,
        "saved_score_chasing_runtime_seconds": round(saved_runtime, 6),
        "classic_compute_per_claimable_claim": classic["compute_per_claimable_claim"],
        "lucky_compute_per_claimable_claim": lucky["compute_per_claimable_claim"],
        "classic_non_claimable_runs": classic["non_claimable_runs"],
        "lucky_non_claimable_runs": lucky["non_claimable_runs"],
        "classic_non_claimable_runtime_seconds": classic["non_claimable_runtime_seconds"],
        "lucky_non_claimable_runtime_seconds": lucky["non_claimable_runtime_seconds"],
        "qwen_triggered_verification": lucky["qwen_triggered_verification"],
        "qwen_skip_or_stop_recommended": lucky["qwen_skip_or_stop_recommended"],
        "qwen_recommended_action": lucky["recommended_action"],
        "qwen_stop_reason": lucky["reason"],
        "qwen_stop_after_run": lucky["stop_after_run"],
        "strict_stop_saved_runs": strict_stop_saved_runs,
        "strict_stop_saved_runtime_seconds": strict_stop_saved_runtime,
    }


def write_reports(rows: list[dict], paired_rows: list[dict]) -> None:
    out_dir = ROOT / "reports" / "budgeted_compute"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "interpretation": (
            "This evaluation holds run budget constant and measures score-chasing compute that cannot support a robust claim. "
            "Saved counts are SIGNED deltas (classic - lucky): negative means Lucky wasted more, so regressions are visible. "
            "It does not claim lower total runtime when Lucky Loop chooses expensive multi-seed verification."
        ),
        "rows": rows,
        "paired_rows": paired_rows,
        "summary": {
            "tasks": len(paired_rows),
            "tasks_with_saved_score_chasing_runs": sum(1 for row in paired_rows if row["saved_score_chasing_runs"] > 0),
            "tasks_where_lucky_wasted_more": sum(1 for row in paired_rows if row["saved_score_chasing_runs"] < 0),
            "total_saved_score_chasing_runs": sum(row["saved_score_chasing_runs"] for row in paired_rows),
            "total_saved_score_chasing_runtime_seconds": round(
                sum(row["saved_score_chasing_runtime_seconds"] for row in paired_rows),
                6,
            ),
            "tasks_with_qwen_stop_or_skip": sum(1 for row in paired_rows if row["qwen_skip_or_stop_recommended"]),
            "total_strict_stop_saved_runs": sum(row["strict_stop_saved_runs"] for row in paired_rows),
            "total_strict_stop_saved_runtime_seconds": round(
                sum(row["strict_stop_saved_runtime_seconds"] for row in paired_rows),
                6,
            ),
        },
    }
    (out_dir / "budgeted_compute_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Budgeted Compute Evaluation",
        "",
        "This report holds the run budget constant and asks whether Lucky Loop spends less compute on non-claimable score chasing. It does not claim lower total runtime when Lucky Loop chooses multi-seed verification.",
        "",
        "## Summary",
        "",
        f"- Tasks: {payload['summary']['tasks']}",
        f"- Tasks with saved score-chasing runs: {payload['summary']['tasks_with_saved_score_chasing_runs']}",
        f"- Total saved score-chasing runs: {payload['summary']['total_saved_score_chasing_runs']}",
        f"- Total saved score-chasing runtime: {_fmt(payload['summary']['total_saved_score_chasing_runtime_seconds'])}s",
        f"- Tasks where Qwen would skip/stop after verifier: {payload['summary']['tasks_with_qwen_stop_or_skip']}",
        f"- Strict stop policy saved runs after verification: {payload['summary']['total_strict_stop_saved_runs']}",
        f"- Strict stop policy saved runtime after verification: {_fmt(payload['summary']['total_strict_stop_saved_runtime_seconds'])}s",
        "",
        "## Paired Budget Results",
        "",
        "| Task | Budget | Classic wasted runs | Lucky wasted runs | Saved score-chasing runs | Saved score-chasing runtime | Qwen verification | Qwen stop/skip | Stop saved runs | Stop saved runtime |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            f"| {row['task']} | {row['budget_runs']} | "
            f"{row['classic_wasted_score_chasing_runs']} | {row['lucky_wasted_score_chasing_runs']} | "
            f"{row['saved_score_chasing_runs']} | {_fmt(row['saved_score_chasing_runtime_seconds'])}s | "
            f"{'yes' if row['qwen_triggered_verification'] else 'no'} | "
            f"{row['qwen_recommended_action'] if row['qwen_skip_or_stop_recommended'] else 'continue'} | "
            f"{row['strict_stop_saved_runs']} | {_fmt(row['strict_stop_saved_runtime_seconds'])}s |"
        )
    lines += [
        "",
        "## Policy Rows",
        "",
        "| Task | Policy | Runs | Non-claimable runs | Non-claimable runtime | Runs after verification needed | Wasted score-chasing runs | Wasted score-chasing runtime | Compute / claimable claim |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['policy']} | {row['budget_runs']} | "
            f"{row['non_claimable_runs']} | {_fmt(row['non_claimable_runtime_seconds'])}s | "
            f"{row['runs_after_verification_needed']} | {row['wasted_score_chasing_runs']} | "
            f"{_fmt(row['wasted_score_chasing_runtime_seconds'])}s | "
            f"{_fmt(row['compute_per_claimable_claim']) or '∞'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A saved score-chasing run is a model run taken after the evidence state already required top-model verification. Lucky Loop's advantage here is compute allocation: it spends budget on claim risk instead of continuing non-claimable leaderboard search.",
        "",
        "`stop_and_report` is evaluated as a strict best-model-claim mode: after an inconclusive verifier result, the system should stop making robust winner claims and skip remaining score-chasing budget unless the research objective changes.",
    ]
    (out_dir / "budgeted_compute_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=None, help="Optional fixed run budget; defaults to all existing ablation traces.")
    args = parser.parse_args()

    rows = []
    paired_rows = []
    for task_path in TASK_PATHS:
        task_id = Path(task_path).stem
        classic_traces = _load_traces("classic_autoresearch", task_id, args.budget)
        lucky_traces = _load_traces("lucky_loop_full", task_id, args.budget)
        shared_budget = min(len(classic_traces), len(lucky_traces))
        classic = _policy_row(task_id, "classic_autoresearch", classic_traces[:shared_budget])
        lucky = _policy_row(task_id, "lucky_loop_full", lucky_traces[:shared_budget])
        rows.extend([classic, lucky])
        paired_rows.append(_paired_row(task_id, classic, lucky))

    write_reports(rows, paired_rows)
    print("Wrote reports/budgeted_compute/budgeted_compute_evaluation.md")


if __name__ == "__main__":
    main()
