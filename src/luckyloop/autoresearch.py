from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .defaults import CORE_POLICIES, CORE_TASK_PATHS
from .literature import synthesize_context, write_context
from .schemas import ResearchAction, ResearchProgram
from .tasks import ROOT, load_task


DEFAULT_TASKS = CORE_TASK_PATHS


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


def _research_actions_for_task(task_path: str) -> list[ResearchAction]:
    task = load_task(task_path)
    return [
        ResearchAction(
            action_id=f"{task.task_id}:literature_review",
            kind="literature_review",
            task_id=task.task_id,
            description=f"Map literature gaps to the {task.task_id} evidence target.",
            expected_claim_change="Defines which claims must be verified before reporting.",
            cost_class="free",
            risk_focus=["claim_calibration", "benchmark_misuse"],
            produces=["literature/related_work.md", "experiment_plan.json"],
        ),
        ResearchAction(
            action_id=f"{task.task_id}:single_experiment",
            kind="single_experiment",
            task_id=task.task_id,
            description=f"Run safe-catalog sklearn experiments on {task.dataset}.",
            command=f"PYTHONPATH=src python -m luckyloop.loop --task {task_path}",
            expected_claim_change="Adds observations but does not by itself allow a robust best-model claim.",
            cost_class="cheap",
            risk_focus=["single_split_overclaim"],
            produces=[f"runs/ablations/lucky_loop_full/{task.task_id}/run_*.json"],
        ),
        ResearchAction(
            action_id=f"{task.task_id}:multi_seed_verification",
            kind="multi_seed_verification",
            task_id=task.task_id,
            description="Verify close top models or hyperparameter winners on matched seeds.",
            expected_claim_change="Allows, weakens, blocks, or rewrites best-model claims.",
            cost_class="moderate",
            risk_focus=["seed_variance", "effect_smaller_than_noise"],
            produces=[f"reports/ablations/lucky_loop_full/{task.task_id}/claim_ledger.json"],
        ),
        ResearchAction(
            action_id=f"{task.task_id}:protocol_probe",
            kind="protocol_probe",
            task_id=task.task_id,
            description="Probe whether a high-looking metric should be blocked by protocol risk.",
            expected_claim_change="Blocks or rewrites protocol-fragile claims.",
            cost_class="moderate",
            risk_focus=["metric_misuse", "protocol_fragility"],
            produces=[f"runs/ablations/lucky_loop_full/{task.task_id}/run_*.json"],
        ),
    ]


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
        "export LUCKYWORLD_SIMULATOR_BASE_URL=http://YOUR_SIMULATOR_HOST:8000/v1",
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
    actions = []
    for task_path in task_paths:
        actions.extend(_research_actions_for_task(task_path))
    actions.extend(
        [
            ResearchAction(
                action_id="global:ablation",
                kind="ablation",
                description="Compare classic autoresearch, verifier-only, and Lucky Loop full.",
                command="PYTHONPATH=src python scripts/run_ablation_suite.py --world-model auto",
                expected_claim_change="Measures whether world-model-guided decisions change claim quality.",
                cost_class="expensive",
                risk_focus=["unsupported_claims", "score_chasing"],
                produces=["reports/ablations/world_model_ablation.json"],
            ),
            ResearchAction(
                action_id="global:counterfactual",
                kind="counterfactual",
                description="Replay states where Lucky Loop and classic autoresearch choose different actions.",
                command="PYTHONPATH=src python scripts/run_counterfactual_evaluation.py",
                expected_claim_change="Shows whether Qwen-guided choices prevented unsupported claims.",
                cost_class="moderate",
                risk_focus=["world_model_usefulness"],
                produces=["reports/counterfactuals/counterfactual_evaluation.json"],
            ),
            ResearchAction(
                action_id="global:stop_and_report",
                kind="stop_and_report",
                description="Stop spending compute once remaining actions cannot improve claim quality.",
                expected_claim_change="Makes the final report honest instead of score-chasing.",
                cost_class="free",
                risk_focus=["non_claimable_compute"],
                produces=["reports/autoresearch/<slug>/final_report.md"],
            ),
        ]
    )
    program = ResearchProgram(
        question=question,
        goal="Run an end-to-end predictive research lab loop with world-model simulation before compute and claim verification after execution.",
        selected_tasks=[task["task_id"] for task in tasks],
        literature_gaps=[asdict(gap) for gap in context.gap_findings],
        success_metrics=context.recommended_metrics,
        candidate_research_actions=actions,
        baselines=CORE_POLICIES,
        constraints={
            "safe_catalog_only": True,
            "single_run_is_observation_not_claim": True,
            "world_model_predicts_not_verifies": True,
            "verifier_is_deterministic": True,
        },
    )
    payload = program.model_dump()
    payload.update(
        {
            "tasks": tasks,
            "policies": CORE_POLICIES,
            "metrics": context.recommended_metrics,
            "steps": [
                "literature_review",
                "research_program",
                "classic_autoresearch_baseline",
                "classic_verified_baseline",
                "lucky_loop_full_world_model_run",
                "counterfactual_evaluation",
                "budgeted_compute_evaluation",
                "artifact_validation",
                "claim_calibrated_report",
            ],
            "success_criteria": [
                "Live or cached Qwen-AgentWorld candidate predictions exist for Lucky Loop full traces.",
                "Classic autoresearch baseline is compared against Lucky Loop full.",
                "At least one world-model-driven action differs from classic score chasing.",
                "Claims are allowed or blocked through claim ledger entries.",
                "Final report distinguishes score observations from robust claims.",
            ],
        }
    )
    return payload


def write_run_commands(path: Path, question: str, agent: str, task_paths: list[str]) -> None:
    task_args = " ".join(task_paths)
    lines = [
        "# Run Commands",
        "",
        "## Full Agent-Operated Research",
        "",
        "```bash",
        "export PYTHONPATH=src",
        "export LUCKYWORLD_SIMULATOR_BASE_URL=http://YOUR_SIMULATOR_HOST:8000/v1",
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
        "PYTHONPATH=src python scripts/run_budgeted_compute_evaluation.py",
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
            "counterfactuals": "reports/counterfactuals/counterfactual_evaluation.md",
            "budgeted_compute": "reports/budgeted_compute/budgeted_compute_evaluation.md",
            "backend_pitch": "reports/pitch_backend_summary.md",
        },
        "workspace_artifacts": {
            "experiment_plan": _rel(out_dir / "experiment_plan.json"),
            "research_trace": _rel(out_dir / "research_trace.json"),
            "decision_journal": _rel(out_dir / "decision_journal.jsonl"),
            "final_report": _rel(out_dir / "final_report.md"),
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
        _run([sys.executable, "scripts/run_ablation_suite.py", "--world-model", "auto", "--operator-agent", agent, "--tasks", *task_paths, "--policies", *CORE_POLICIES], env)
    _run([sys.executable, "scripts/run_counterfactual_evaluation.py", "--tasks", *task_paths], env)
    _run([sys.executable, "scripts/run_budgeted_compute_evaluation.py", "--tasks", *task_paths], env)
    _run([sys.executable, "scripts/validate_artifacts.py", "--check-ablations", "--require-qwen", "--skip-main", "--tasks", *task_paths], env)


def _load_policy_trace_paths(task_paths: list[str], policy: str = "lucky_loop_full") -> list[Path]:
    paths: list[Path] = []
    for task_path in task_paths:
        task_id = load_task(task_path).task_id
        paths.extend(sorted((ROOT / "runs" / "ablations" / policy / task_id).glob("run_*.json")))
    return paths


def write_research_trace(out_dir: Path, question: str, task_paths: list[str]) -> None:
    traces = []
    for path in _load_policy_trace_paths(task_paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        traces.append(
            {
                "trace_path": _rel(path),
                "task_id": data.get("artifacts", {}).get("task_id"),
                "run_id": data.get("run_id"),
                "selected_action": data.get("proposed_action", {}).get("model"),
                "world_model_prediction": {
                    "recommendation": data.get("world_model_prediction", {}).get("recommendation"),
                    "expected_claim_delta": data.get("world_model_prediction", {}).get("expected_claim_delta"),
                    "expected_claim_resolution": data.get("world_model_prediction", {}).get("expected_claim_resolution"),
                    "why_not_classic_autoresearch": data.get("world_model_prediction", {}).get("why_not_classic_autoresearch"),
                },
                "actual_status": data.get("actual_result", {}).get("status"),
                "verification_status": (data.get("verification") or {}).get("status"),
            }
        )
    _write_json(
        out_dir / "research_trace.json",
        {
            "schema_version": "1.0",
            "question": question,
            "trace_count": len(traces),
            "traces": traces,
        },
    )


def write_decision_journal(out_dir: Path, task_paths: list[str]) -> None:
    lines = []
    for path in _load_policy_trace_paths(task_paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        decision = data.get("decision_trace") or {}
        state = data.get("state_before") or {}
        selected = data.get("proposed_action") or {}
        entry = {
            "state_id": state.get("state_id"),
            "task_id": data.get("artifacts", {}).get("task_id"),
            "candidate_actions": [
                {
                    "action_id": action.get("action_id"),
                    "model": action.get("model"),
                    "params": action.get("params"),
                }
                for action in data.get("candidate_actions", [])
            ],
            "world_model_predictions": [
                {
                    "action_id": item.get("action", {}).get("action_id"),
                    "model": item.get("action", {}).get("model"),
                    "recommendation": item.get("prediction", {}).get("recommendation"),
                    "expected_claim_delta": item.get("prediction", {}).get("expected_claim_delta"),
                    "expected_claim_resolution": item.get("prediction", {}).get("expected_claim_resolution"),
                    "source": item.get("source"),
                }
                for item in data.get("candidate_predictions", [])
            ],
            "selected_action": selected.get("action_id"),
            "selected_model": selected.get("model"),
            "why_world_model_mattered": decision.get("why_world_model_mattered") or decision.get("world_model_decision_basis"),
            "classic_counterfactual_action": decision.get("classic_counterfactual_action_id"),
            "claim_status_before": (
                "needs_verification"
                if (state.get("top_model_summary") or {}).get("needs_robustness_verification")
                else "observation_only"
            ),
            "claim_status_after": (data.get("world_model_prediction") or {}).get("expected_claim_delta"),
        }
        lines.append(json.dumps(entry, sort_keys=True))
    (out_dir / "decision_journal.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_final_report(out_dir: Path, question: str, context, evidence: dict) -> None:
    ablation_path = ROOT / "reports" / "ablations" / "world_model_ablation.json"
    counterfactual_path = ROOT / "reports" / "counterfactuals" / "counterfactual_evaluation.json"
    budgeted_path = ROOT / "reports" / "budgeted_compute" / "budgeted_compute_evaluation.json"
    rows = []
    if ablation_path.exists():
        rows = json.loads(ablation_path.read_text(encoding="utf-8")).get("rows", [])
    counterfactual_summary = {}
    if counterfactual_path.exists():
        counterfactual_summary = json.loads(counterfactual_path.read_text(encoding="utf-8")).get("summary", {})
    budgeted_summary = {}
    if budgeted_path.exists():
        budgeted_summary = json.loads(budgeted_path.read_text(encoding="utf-8")).get("summary", {})
    lines = [
        "# Predictive Research Lab Report",
        "",
        f"Question: {question}",
        "",
        "## Method",
        "",
        "Lucky Loop is a predictive research lab OS backend. A coding agent operates inside the repository, but Qwen-AgentWorld acts as the language world model: before compute is spent, it predicts each candidate research action's outcome, protocol risk, value of information, and likely claim impact. Real sklearn experiments then test those predictions, and a deterministic verifier decides which claims can enter the report.",
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
        "- Research trace: `research_trace.json`",
        "- Decision journal: `decision_journal.jsonl`",
        "- Evidence manifest: `evidence_manifest.json`",
        "- Backend ablation: `reports/ablations/world_model_ablation.md`",
        "- Counterfactual evaluation: `reports/counterfactuals/counterfactual_evaluation.md`",
        "- Budgeted compute evaluation: `reports/budgeted_compute/budgeted_compute_evaluation.md`",
        "",
        "## Ablation Snapshot",
        "",
        "| Task | Policy | Best single-run | Best verified mean | Best claimable | Unsupported best-model claims | Claims blocked | Qwen predictions |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric = row.get("best_single_run_metric")
        metric_text = "" if metric is None else f"{metric:.4f}"
        verified = row.get("best_verified_mean_score")
        verified_text = "" if verified is None else f"{verified:.4f}"
        claimable = row.get("best_claimable_score")
        claimable_text = "" if claimable is None else f"{claimable:.4f}"
        lines.append(
            f"| {row.get('task')} | {row.get('policy')} | {metric_text} | {verified_text} | {claimable_text} | "
            f"{row.get('unsupported_best_model_claims')} | {row.get('claims_blocked')} | "
            f"{row.get('world_model_prediction_count')} |"
        )
    if counterfactual_summary:
        usefulness = counterfactual_summary.get("qwen_choice_usefulness")
        usefulness_text = "" if usefulness is None else f"{usefulness:.2%}"
        breakdown = counterfactual_summary.get("overall_verdict_breakdown") or {}
        breakdown_text = ", ".join(f"{k}: {v}" for k, v in sorted(breakdown.items())) or "n/a"
        lines += [
            "",
            "## Counterfactual Result",
            "",
            f"- Cases: {counterfactual_summary.get('cases')}",
            f"- Claim-safety wins (Lucky verified what classic skipped): {counterfactual_summary.get('claim_safety_wins')}",
            f"- Verdict breakdown: {breakdown_text}",
            f"- Qwen choice usefulness: {usefulness_text}",
            "",
            "A win requires Lucky to have actually run verification the state needed and classic to "
            "have skipped it; raw scores across different experiment kinds (multi-seed sweep vs single "
            "split) are reported as not_comparable, never as a win.",
        ]
    if budgeted_summary:
        lines += [
            "",
            "## Budgeted Compute Result",
            "",
            f"- Tasks with saved score-chasing runs: {budgeted_summary.get('tasks_with_saved_score_chasing_runs')}",
            f"- Total saved score-chasing runs: {budgeted_summary.get('total_saved_score_chasing_runs')}",
            f"- Total saved score-chasing runtime: {budgeted_summary.get('total_saved_score_chasing_runtime_seconds')}s",
            f"- Tasks where Qwen would skip/stop after verifier: {budgeted_summary.get('tasks_with_qwen_stop_or_skip')}",
            f"- Strict stop policy saved runs after verification: {budgeted_summary.get('total_strict_stop_saved_runs')}",
        ]
    lucky_rows = [r for r in rows if r.get("policy") == "lucky_loop_full"]
    unsupported_claims = sum((r.get("unsupported_best_model_claims") or 0) for r in lucky_rows)
    claim_discipline = (
        "Classic autoresearch can find good single-run scores, but those are not robust claims. "
        "Lucky Loop full adds auditable pre-compute predictions and a deterministic claim gate; "
        f"across the generated ablation artifacts it leaves {unsupported_claims} unsupported "
        "best-model claim(s)."
    )
    lines += [
        "",
        "## Claim Discipline",
        "",
        claim_discipline,
        "",
        "The final report may discuss raw score observations, verified means, and blocked claims, but it must not turn a single-run best score into a robust scientific claim.",
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

    write_research_trace(out_dir, question, task_paths)
    write_decision_journal(out_dir, task_paths)
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
