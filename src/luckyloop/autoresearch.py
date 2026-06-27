from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .literature import synthesize_context, write_context
from .tasks import ROOT, load_task


DEFAULT_TASKS = [
    "configs/tasks/breast_cancer_accuracy.json",
    "configs/tasks/wine_accuracy.json",
    "configs/tasks/digits_accuracy.json",
]


def slugify(text: str, max_len: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].strip("-") or "autoresearch")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_question(path: Path, question: str, agent: str) -> None:
    lines = [
        "# Research Question",
        "",
        question,
        "",
        "## Agent Mode",
        "",
        f"- Autoresearch agent: `{agent}` operating inside the repository.",
        "- Lucky Loop does not call the coding agent as a private API.",
        "- The agent runs the backend protocol, inspects evidence, and writes the final interpretation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_agent_instructions(path: Path, question: str, agent: str, tasks: list[str]) -> None:
    lines = [
        "# Agent Instructions",
        "",
        f"You are `{agent}`, the autoresearch agent operating inside this repo.",
        "",
        "## Objective",
        "",
        question,
        "",
        "Use Lucky Loop as the evidence engine. You are responsible for the research decisions and final interpretation. Qwen-AgentWorld is the world model, not the planner.",
        "",
        "## Required Loop",
        "",
        "1. Read `question.md` and `literature/related_work.md`.",
        "2. Check `experiment_plan.json`.",
        "3. Run or validate the benchmark/ablation artifacts.",
        "4. Inspect prediction-vs-reality traces.",
        "5. Use the claim ledger before making any claim.",
        "6. Write the final answer from evidence only.",
        "",
        "## Tasks",
        "",
        *[f"- `{task}`" for task in tasks],
        "",
        "## Main Commands",
        "",
        "```bash",
        "export PYTHONPATH=src",
        "export LUCKYWORLD_SIMULATOR_BASE_URL=http://134.199.205.222:8000/v1",
        "export LUCKYWORLD_SIMULATOR_MODEL=Qwen/Qwen-AgentWorld-35B-A3B",
        "export LUCKYWORLD_SIMULATOR_API_KEY=dummy",
        "export LUCKYWORLD_SIMULATOR_TIMEOUT_SECONDS=25",
        "",
        "python -m luckyloop.autoresearch --question \"$QUESTION\" --execute",
        "python scripts/validate_artifacts.py --check-ablations --require-qwen",
        "```",
        "",
        "## Claim Rules",
        "",
        "- A single-run best score is an observation, not a robust claim.",
        "- A robust best-model claim requires top-model verification or equivalent multi-seed evidence.",
        "- If `effect_size < seed_noise`, report the result as inconclusive.",
        "- Prediction misses are evidence and must remain visible.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_experiment_plan(question: str, task_paths: list[str], context) -> dict:
    tasks = []
    for task_path in task_paths:
        task = load_task(task_path)
        tasks.append(
            {
                "task_id": task.task_id,
                "dataset": task.dataset,
                "primary_metric": task.primary_metric,
                "budget_runs": task.budget_runs,
                "models": task.models,
                "task_spec": task_path,
            }
        )
    return {
        "schema_version": "1.0",
        "question": question,
        "goal": "Run an end-to-end agent-operated autoresearch loop with world-model predictions before compute and claim verification after execution.",
        "tasks": tasks,
        "policies": context.recommended_baselines,
        "metrics": context.recommended_metrics,
        "steps": [
            "literature_review",
            "experiment_plan",
            "classic_autoresearch_baseline",
            "classic_verified_baseline",
            "lucky_loop_full_world_model_run",
            "artifact_validation",
            "claim_calibrated_report",
        ],
        "success_criteria": [
            "Live or cached Qwen-AgentWorld candidate predictions exist for Lucky Loop full traces.",
            "Classic autoresearch baseline is compared against Lucky Loop full.",
            "Top observed models are detected from real sklearn results.",
            "Claims are allowed or blocked through claim ledger entries.",
            "Final report distinguishes score observations from robust claims.",
        ],
    }


def write_run_commands(path: Path, question: str, agent: str, task_paths: list[str]) -> None:
    task_args = " ".join(task_paths)
    lines = [
        "# Run Commands",
        "",
        "## Full Agent-Operated Research",
        "",
        "```bash",
        "export PYTHONPATH=src",
        "export LUCKYWORLD_SIMULATOR_BASE_URL=http://134.199.205.222:8000/v1",
        "export LUCKYWORLD_SIMULATOR_MODEL=Qwen/Qwen-AgentWorld-35B-A3B",
        "export LUCKYWORLD_SIMULATOR_API_KEY=dummy",
        "export LUCKYWORLD_SIMULATOR_TIMEOUT_SECONDS=25",
        "",
        "python -m luckyloop.autoresearch \\",
        f"  --question {json.dumps(question)} \\",
        f"  --agent {agent} \\",
        f"  --tasks {task_args} \\",
        "  --execute",
        "```",
        "",
        "## Re-run Core Ablation Only",
        "",
        "```bash",
        f"PYTHONPATH=src python scripts/run_ablation_suite.py --world-model auto --operator-agent {agent}",
        "```",
        "",
        "## Validate",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/validate_artifacts.py --check-ablations --require-qwen",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_evidence_manifest(out_dir: Path, task_paths: list[str], executed: bool) -> dict:
    task_ids = [load_task(path).task_id for path in task_paths]
    evidence = {
        "schema_version": "1.0",
        "executed_this_run": executed,
        "workspace": _rel(out_dir),
        "literature": {
            "related_work": _rel(out_dir / "literature" / "related_work.md"),
            "research_context": _rel(out_dir / "literature" / "research_context.json"),
            "sources_json": _rel(out_dir / "literature" / "sources.json"),
            "sources_bib": _rel(out_dir / "literature" / "sources.bib"),
        },
        "benchmark_reports": {
            "summary": "reports/benchmark_summary.md",
            "ablation": "reports/ablations/world_model_ablation.md",
            "classic_vs_lucky_loop": "reports/ablations/classic_vs_lucky_loop.md",
            "backend_pitch": "reports/pitch_backend_summary.md",
        },
        "task_artifacts": [],
    }
    for task_id in task_ids:
        evidence["task_artifacts"].append(
            {
                "task_id": task_id,
                "main_runs": f"runs/{task_id}/run_*.json",
                "main_report": f"reports/{task_id}/final_report.md",
                "claim_ledger": f"reports/{task_id}/claim_ledger.json",
                "calibration": f"reports/{task_id}/world_model_calibration.md",
                "ablation_runs": f"runs/ablations/*/{task_id}/run_*.json",
                "ablation_reports": f"reports/ablations/*/{task_id}/final_report.md",
            }
        )
    return evidence


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def execute_backend(task_paths: list[str], agent: str, rerun_experiments: bool) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    if rerun_experiments:
        _run([sys.executable, "scripts/run_ablation_suite.py", "--world-model", "auto", "--operator-agent", agent, "--tasks", *task_paths], env)
    _run([sys.executable, "scripts/validate_artifacts.py", "--check-ablations", "--require-qwen"], env)


def write_final_report(out_dir: Path, question: str, context, evidence: dict) -> None:
    ablation_path = ROOT / "reports" / "ablations" / "world_model_ablation.json"
    rows = []
    if ablation_path.exists():
        rows = json.loads(ablation_path.read_text(encoding="utf-8")).get("rows", [])
    lines = [
        "# Agent-Operated Autoresearch Report",
        "",
        f"Question: {question}",
        "",
        "## Method",
        "",
        "A coding agent operates inside the repository. Lucky Loop supplies the research protocol: literature context, safe experiment catalog, Qwen-AgentWorld predictions before compute, real sklearn execution, prediction-vs-reality comparison, deterministic verification, and claim ledger reporting.",
        "",
        "## Literature-Derived Gaps",
        "",
        *[
            (
                f"- {gap.claim} Sources: "
                f"{', '.join(f'[{source_id}]' for source_id in gap.source_ids) or '[no_source]'}. "
                f"Implication: {gap.implication}"
            )
            for gap in context.gap_findings
        ],
        "",
        "## Evidence Summary",
        "",
        "- Related work: `literature/related_work.md`",
        "- Sources: `literature/sources.json` and `literature/sources.bib`",
        "- Experiment plan: `experiment_plan.json`",
        "- Evidence manifest: `evidence_manifest.json`",
        "- Backend ablation: `reports/ablations/world_model_ablation.md`",
        "",
        "## Ablation Snapshot",
        "",
        "| Task | Policy | Best single-run | Unsupported best-model claims | Claims blocked | Qwen predictions |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric = row.get("best_single_run_metric")
        metric_text = "" if metric is None else f"{metric:.4f}"
        lines.append(
            f"| {row.get('task')} | {row.get('policy')} | {metric_text} | "
            f"{row.get('unsupported_best_model_claims')} | {row.get('claims_blocked')} | "
            f"{row.get('world_model_prediction_count')} |"
        )
    lines += [
        "",
        "## Claim Discipline",
        "",
        "Classic autoresearch can find good single-run scores, but those are not robust claims. Lucky Loop full adds auditable pre-compute predictions and keeps unsupported best-model claims at zero in the generated ablation artifacts.",
        "",
        "## Source Mapping",
        "",
        "| Gap | Sources | Metric | Experiment |",
        "|---|---|---|---|",
    ]
    for gap in context.gap_findings:
        sources = ", ".join(f"[{source_id}]" for source_id in gap.source_ids) or "[no_source]"
        lines.append(f"| {gap.gap_id} | {sources} | `{gap.metric}` | {gap.experiment} |")
    lines += [
        "",
        "## Manifest",
        "",
        f"Workspace: `{evidence['workspace']}`",
    ]
    (out_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_workspace(
    question: str,
    agent: str,
    task_paths: list[str],
    execute: bool,
    rerun_experiments: bool,
) -> Path:
    slug = slugify(question)
    out_dir = ROOT / "reports" / "autoresearch" / slug
    literature_dir = out_dir / "literature"
    out_dir.mkdir(parents=True, exist_ok=True)

    context = synthesize_context(question)
    write_question(out_dir / "question.md", question, agent)
    write_agent_instructions(out_dir / "agent_instructions.md", question, agent, task_paths)
    write_context(context, literature_dir)
    plan = build_experiment_plan(question, task_paths, context)
    _write_json(out_dir / "experiment_plan.json", plan)
    write_run_commands(out_dir / "run_commands.md", question, agent, task_paths)

    if execute:
        execute_backend(task_paths, agent, rerun_experiments)

    evidence = collect_evidence_manifest(out_dir, task_paths, executed=execute)
    _write_json(out_dir / "evidence_manifest.json", evidence)
    write_final_report(out_dir, question, context, evidence)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an agent-operated Lucky Loop autoresearch workspace.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--agent", default="codex_operator")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--execute", action="store_true", help="Validate artifacts and optionally rerun experiments.")
    parser.add_argument(
        "--rerun-experiments",
        action="store_true",
        help="With --execute, rerun the full ablation suite instead of using existing artifacts.",
    )
    args = parser.parse_args()
    out_dir = create_workspace(args.question, args.agent, args.tasks, args.execute, args.rerun_experiments)
    print(f"Wrote autoresearch workspace to {_rel(out_dir)}")


if __name__ == "__main__":
    main()
