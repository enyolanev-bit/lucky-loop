#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from luckyloop.benchmark_metrics import actual_metric, best_single_run, claimable_evidence_summary
from luckyloop.defaults import CORE_TASK_PATHS
from luckyloop.loop import run
from luckyloop.schemas import ExperimentTrace
from luckyloop.tasks import ROOT, load_task


DEFAULT_TASKS = CORE_TASK_PATHS

POLICIES = ["deepseek_direct", "deepseek_agentworld"]


def _set_or_unset(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _load_traces(namespace: str) -> list[ExperimentTrace]:
    run_dir = ROOT / "runs" / namespace
    return [
        ExperimentTrace.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("run_*.json"))
    ]


def _run_policy(task_path: str, policy: str, max_experiments: int | None, reuse_existing: bool) -> list[ExperimentTrace]:
    task = load_task(task_path)
    namespace = f"frontier_cost/{policy}/{task.task_id}"
    if reuse_existing and (ROOT / "runs" / namespace).exists():
        return _load_traces(namespace)

    old_sim_base = os.environ.get("LUCKYWORLD_SIMULATOR_BASE_URL")
    old_sim_model = os.environ.get("LUCKYWORLD_SIMULATOR_MODEL")
    if policy == "deepseek_direct":
        os.environ.pop("LUCKYWORLD_SIMULATOR_BASE_URL", None)
        os.environ.pop("LUCKYWORLD_SIMULATOR_MODEL", None)
    try:
        return run(
            task=task,
            output_namespace=namespace,
            max_experiments=max_experiments,
            planner_mode="llm",
            agent_backend=policy,
        )
    finally:
        _set_or_unset("LUCKYWORLD_SIMULATOR_BASE_URL", old_sim_base)
        _set_or_unset("LUCKYWORLD_SIMULATOR_MODEL", old_sim_model)


def _sum_artifact(traces: list[ExperimentTrace], key: str) -> float:
    return round(sum(float(trace.artifacts.get(key) or 0.0) for trace in traces), 6)


def _summarize(policy: str, task_id: str, traces: list[ExperimentTrace]) -> dict:
    claimable = claimable_evidence_summary(traces)
    best = best_single_run(traces)
    frontier_planner_calls = sum(1 for trace in traces if trace.agent_decision is not None)
    candidate_prediction_calls = sum(int(trace.artifacts.get("candidate_prediction_count") or 0) for trace in traces)
    if policy == "deepseek_agentworld":
        agentworld_calls = candidate_prediction_calls
        estimated_frontier_candidate_eval_calls_without_agentworld = candidate_prediction_calls
        estimated_frontier_calls_saved_by_agentworld = max(candidate_prediction_calls - frontier_planner_calls, 0)
    else:
        agentworld_calls = 0
        estimated_frontier_candidate_eval_calls_without_agentworld = 0
        estimated_frontier_calls_saved_by_agentworld = 0
    return {
        "task": task_id,
        "policy": policy,
        "runs": len(traces),
        "best_single_run_model": best.proposed_action.model if best else None,
        "best_single_run_metric": actual_metric(best) if best else None,
        "top_model_verification_runs": sum(1 for trace in traces if trace.proposed_action.model == "top_model_verification"),
        "stop_runs": sum(1 for trace in traces if trace.proposed_action.model == "stop_and_report"),
        "unsupported_best_model_claims": 1 if policy == "deepseek_direct" and best else 0,
        "frontier_planner_calls": frontier_planner_calls,
        "agentworld_prediction_calls": agentworld_calls,
        "candidate_predictions_total": candidate_prediction_calls,
        "estimated_frontier_candidate_eval_calls_without_agentworld": estimated_frontier_candidate_eval_calls_without_agentworld,
        "estimated_frontier_calls_saved_by_agentworld": estimated_frontier_calls_saved_by_agentworld,
        "agent_runtime_seconds": _sum_artifact(traces, "agent_runtime_seconds"),
        "world_model_runtime_seconds": _sum_artifact(traces, "world_model_prediction_runtime_seconds"),
        **claimable,
    }


def _fmt(value) -> str:
    if value is None:
        return "∞"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(rows: list[dict]) -> None:
    out_dir = ROOT / "reports" / "frontier_cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired = []
    for task in sorted({row["task"] for row in rows}):
        direct = next((row for row in rows if row["task"] == task and row["policy"] == "deepseek_direct"), None)
        aw = next((row for row in rows if row["task"] == task and row["policy"] == "deepseek_agentworld"), None)
        if direct and aw:
            paired.append(
                {
                    "task": task,
                    "direct_frontier_calls": direct["frontier_planner_calls"],
                    "agentworld_frontier_calls": aw["frontier_planner_calls"],
                    "agentworld_prediction_calls": aw["agentworld_prediction_calls"],
                    "estimated_frontier_calls_saved_by_agentworld": aw["estimated_frontier_calls_saved_by_agentworld"],
                    "direct_unsupported_claims": direct["unsupported_best_model_claims"],
                    "agentworld_unsupported_claims": aw["unsupported_best_model_claims"],
                    "direct_runs": direct["runs"],
                    "agentworld_runs": aw["runs"],
                    "agentworld_top_model_verification_runs": aw["top_model_verification_runs"],
                    "agentworld_stop_runs": aw["stop_runs"],
                }
            )
    payload = {
        "schema_version": "1.0",
        "interpretation": (
            "AgentWorld cost reduction is measured as reduced expensive frontier-agent candidate evaluation calls. "
            "The direct DeepSeek planner still makes one planner call per step; AgentWorld cheaply performs per-candidate future simulations."
        ),
        "rows": rows,
        "paired_rows": paired,
        "summary": {
            "tasks": len(paired),
            "estimated_frontier_calls_saved_by_agentworld": sum(
                row["estimated_frontier_calls_saved_by_agentworld"] for row in paired
            ),
            "agentworld_prediction_calls": sum(row["agentworld_prediction_calls"] for row in paired),
            "tasks_with_agentworld_verification": sum(
                1 for row in paired if row["agentworld_top_model_verification_runs"] > 0
            ),
            "tasks_with_agentworld_stop": sum(1 for row in paired if row["agentworld_stop_runs"] > 0),
        },
    }
    (out_dir / "frontier_agent_cost_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Frontier Agent Cost Benchmark",
        "",
        "This benchmark tests a weaker API planner (DeepSeek) with and without AgentWorld. The key cost metric is not sklearn runtime; it is expensive frontier-agent candidate-evaluation calls avoided by using AgentWorld as the cheap world model.",
        "",
        "## Summary",
        "",
        f"- Tasks: {payload['summary']['tasks']}",
        f"- Estimated frontier candidate-evaluation calls saved by AgentWorld: {payload['summary']['estimated_frontier_calls_saved_by_agentworld']}",
        f"- AgentWorld prediction calls used instead: {payload['summary']['agentworld_prediction_calls']}",
        f"- Tasks with AgentWorld-triggered verification: {payload['summary']['tasks_with_agentworld_verification']}",
        f"- Tasks with AgentWorld stop/report: {payload['summary']['tasks_with_agentworld_stop']}",
        "",
        "## Paired Results",
        "",
        "| Task | Direct DeepSeek calls | DeepSeek+AgentWorld planner calls | AgentWorld prediction calls | Estimated frontier calls saved | AgentWorld verification | AgentWorld stop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['task']} | {row['direct_frontier_calls']} | {row['agentworld_frontier_calls']} | "
            f"{row['agentworld_prediction_calls']} | {row['estimated_frontier_calls_saved_by_agentworld']} | "
            f"{row['agentworld_top_model_verification_runs']} | {row['agentworld_stop_runs']} |"
        )
    lines += [
        "",
        "## Policy Rows",
        "",
        "| Task | Policy | Runs | Best single | Verification runs | Stop runs | Planner calls | AgentWorld calls | Est. frontier saved | Agent runtime | World model runtime |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['policy']} | {row['runs']} | "
            f"{_fmt(row['best_single_run_metric'])} | {row['top_model_verification_runs']} | {row['stop_runs']} | "
            f"{row['frontier_planner_calls']} | {row['agentworld_prediction_calls']} | "
            f"{row['estimated_frontier_calls_saved_by_agentworld']} | "
            f"{_fmt(row['agent_runtime_seconds'])}s | {_fmt(row['world_model_runtime_seconds'])}s |"
        )
    (out_dir / "frontier_agent_cost_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--policies", nargs="*", choices=POLICIES, default=POLICIES)
    parser.add_argument("--max-experiments", type=int, default=4)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    rows = []
    for task_path in args.tasks:
        task = load_task(task_path)
        for policy in args.policies:
            traces = _run_policy(task_path, policy, args.max_experiments, args.reuse_existing)
            rows.append(_summarize(policy, task.task_id, traces))
    write_report(rows)
    print("Wrote reports/frontier_cost/frontier_agent_cost_benchmark.md")


if __name__ == "__main__":
    main()
