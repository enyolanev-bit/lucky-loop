#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from luckyloop.defaults import CORE_TASK_PATHS
from luckyloop.operator_trace import append_operator_event, write_operator_summary
from luckyloop.tasks import ROOT, load_task

from run_ablation_suite import run_policy, summarize_traces


DEFAULT_TASKS = CORE_TASK_PATHS

DEFAULT_POLICIES = [
    "score_chaser",
    "classic_verified",
    "lucky_loop_full",
]


def _load_existing(policy: str, task_id: str):
    from luckyloop.schemas import ExperimentTrace

    run_dir = ROOT / "runs" / "ablations" / policy / task_id
    return [
        ExperimentTrace.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("run_*.json"))
    ]


def _fmt(value) -> str:
    if value is None:
        return "∞"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _cost_decision(rows: list[dict]) -> dict:
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], {})[row["policy"]] = row
    paired = []
    for task, policies in sorted(by_task.items()):
        classic = policies.get("score_chaser") or policies.get("fixed_order")
        lucky = policies.get("lucky_loop_qwen_cost_aware") or policies.get("lucky_loop_full")
        if not classic or not lucky:
            continue
        classic_runs = classic["runs_to_first_verification"] or classic["runs"]
        lucky_runs = lucky["runs_to_first_verification"] or lucky["runs"]
        classic_runtime = classic["runtime_after_verification_needed_seconds"] or classic["total_runtime_seconds"]
        lucky_runtime = lucky["runtime_after_verification_needed_seconds"] or lucky["total_runtime_seconds"]
        paired.append(
            {
                "task": task,
                "classic_policy": classic["policy"],
                "lucky_policy": lucky["policy"],
                "classic_runs_to_claim_decision": classic_runs,
                "lucky_runs_to_claim_decision": lucky_runs,
                "saved_runs_to_claim_decision": max(classic_runs - lucky_runs, 0),
                "classic_compute_to_claim_decision": classic_runtime,
                "lucky_compute_to_claim_decision": lucky_runtime,
                "saved_compute_to_claim_decision": round(max(classic_runtime - lucky_runtime, 0.0), 6),
                "classic_unsupported_claims": classic["unsupported_best_model_claims"],
                "lucky_unsupported_claims": lucky["unsupported_best_model_claims"],
                "classic_wasted_score_chasing_runs": classic["wasted_score_chasing_runs"],
                "lucky_wasted_score_chasing_runs": lucky["wasted_score_chasing_runs"],
                "qwen_triggered_verification": lucky["qwen_triggered_verification"],
                "qwen_stop_or_skip": lucky["qwen_skip_or_stop_recommended"],
            }
        )
    return {
        "paired_rows": paired,
        "summary": {
            "tasks": len(paired),
            "tasks_with_saved_runs_to_claim_decision": sum(1 for row in paired if row["saved_runs_to_claim_decision"] > 0),
            "total_saved_runs_to_claim_decision": sum(row["saved_runs_to_claim_decision"] for row in paired),
            "total_saved_compute_to_claim_decision": round(
                sum(row["saved_compute_to_claim_decision"] for row in paired),
                6,
            ),
            "tasks_with_zero_lucky_unsupported_claims": sum(
                1 for row in paired if row["lucky_unsupported_claims"] == 0
            ),
            "tasks_where_qwen_triggered_verification": sum(
                1 for row in paired if row["qwen_triggered_verification"]
            ),
        },
    }


def write_reports(rows: list[dict]) -> None:
    out_dir = ROOT / "reports" / "cost_aware"
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = _cost_decision(rows)
    payload = {"schema_version": "1.0", "rows": rows, **decision}
    (out_dir / "cost_reduction_ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Cost-Aware World-Model Ablation",
        "",
        "This report evaluates whether Lucky Loop uses world-model next-state predictions to reach claim decisions with less non-claimable compute than classic score-chasing autoresearch.",
        "",
        "## Summary",
        "",
        f"- Tasks: {payload['summary']['tasks']}",
        f"- Tasks with saved runs to claim decision: {payload['summary']['tasks_with_saved_runs_to_claim_decision']}",
        f"- Total saved runs to claim decision: {payload['summary']['total_saved_runs_to_claim_decision']}",
        f"- Total saved compute to claim decision: {_fmt(payload['summary']['total_saved_compute_to_claim_decision'])}s",
        f"- Tasks where Lucky Loop kept unsupported claims at zero: {payload['summary']['tasks_with_zero_lucky_unsupported_claims']}",
        f"- Tasks where Qwen triggered verification: {payload['summary']['tasks_where_qwen_triggered_verification']}",
        "",
        "## Classic vs Lucky Loop",
        "",
        "| Task | Classic policy | Lucky policy | Classic runs to claim decision | Lucky runs to claim decision | Saved runs | Classic unsupported claims | Lucky unsupported claims | Qwen verification |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["paired_rows"]:
        lines.append(
            f"| {row['task']} | {row['classic_policy']} | {row['lucky_policy']} | "
            f"{row['classic_runs_to_claim_decision']} | {row['lucky_runs_to_claim_decision']} | "
            f"{row['saved_runs_to_claim_decision']} | {row['classic_unsupported_claims']} | "
            f"{row['lucky_unsupported_claims']} | {'yes' if row['qwen_triggered_verification'] else 'no'} |"
        )
    lines += [
        "",
        "## Policy Rows",
        "",
        "| Task | Policy | Runs | Runtime | Runs to verification | Best single | Best claimable | Unsupported claims | Wasted score-chasing runs | Qwen predictions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['policy']} | {row['runs']} | "
            f"{_fmt(row['total_runtime_seconds'])}s exp + {_fmt(row.get('world_model_runtime_seconds', 0.0))}s wm | {_fmt(row['runs_to_first_verification'])} | "
            f"{_fmt(row['best_single_run_metric'])} | {_fmt(row['best_claimable_score'])} | "
            f"{row['unsupported_best_model_claims']} | {row['wasted_score_chasing_runs']} | "
            f"{row['world_model_prediction_count']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The target win condition is not always a higher single-run score. It is fewer non-claimable score-chasing runs, earlier claim decisions, and zero unsupported best-model claims.",
    ]
    (out_dir / "cost_reduction_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--operator-agent", default="codex_operator")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    old_base = os.environ.get("LUCKYWORLD_SIMULATOR_BASE_URL")
    old_model = os.environ.get("LUCKYWORLD_SIMULATOR_MODEL")
    task_ids = [Path(task).stem for task in args.tasks]
    append_operator_event(
        event_type="cost_reduction_ablation",
        goal="Prove whether world-model next-state predictions reduce autoresearch compute to claim decision.",
        action="run_cost_reduction_ablation",
        operator=args.operator_agent,
        status="started",
        inputs={"tasks": args.tasks, "task_ids": task_ids, "policies": args.policies},
        rationale="Compare score-chasing autoresearch with cost-aware Lucky Loop under matched task budgets.",
    )

    rows = []
    status = "completed"
    error = None
    try:
        for task_path in args.tasks:
            task = load_task(task_path)
            for policy in args.policies:
                traces = (
                    _load_existing(policy, task.task_id)
                    if args.reuse_existing
                    else run_policy(policy, task, args.max_experiments, args.operator_agent)
                )
                rows.append(summarize_traces(policy, task.task_id, traces))
        write_reports(rows)
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        if old_base is not None:
            os.environ["LUCKYWORLD_SIMULATOR_BASE_URL"] = old_base
        if old_model is not None:
            os.environ["LUCKYWORLD_SIMULATOR_MODEL"] = old_model
        append_operator_event(
            event_type="cost_reduction_ablation",
            goal="Prove whether world-model next-state predictions reduce autoresearch compute to claim decision.",
            action="run_cost_reduction_ablation",
            operator=args.operator_agent,
            status=status,
            inputs={"task_ids": task_ids, "policies": args.policies},
            outputs={"rows": len(rows), "error": error},
            rationale="Cost-aware ablation report generated for selected policies.",
        )
        write_operator_summary()
    print("Wrote reports/cost_aware/cost_reduction_ablation.md")


if __name__ == "__main__":
    main()
