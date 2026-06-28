#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def latest_workspace() -> Path | None:
    root = ROOT / "reports" / "lab"
    if not root.exists():
        return None
    workspaces = [path for path in root.iterdir() if path.is_dir()]
    if not workspaces:
        return None
    return max(workspaces, key=lambda path: path.stat().st_mtime)


def validate(workspace: Path, require_qwen: bool, require_agent: bool) -> list[str]:
    failures: list[str] = []
    program_path = workspace / "research_program.json"
    program = json.loads(program_path.read_text(encoding="utf-8")) if program_path.exists() else {}
    open_mode = program.get("mode") in {"open_generated_ml_research", "pure_auto_research_ml_first"}
    pure_mode = program.get("mode") == "pure_auto_research_ml_first"
    required = [
        "question.md",
        "research_program.json",
        "hypotheses.json",
        "notebook.jsonl",
        "claim_ledger.json",
        "final_report.md",
        "reproducibility.md",
        "study_result.json",
    ]
    if open_mode:
        required.extend(
            [
                "literature/domain_sources.json" if pure_mode else "literature/sources.json",
                "literature/domain_related_work.md" if pure_mode else "literature/related_work.md",
                "literature/method_sources.json" if pure_mode else "literature/sources.json",
                "literature/domain_gaps.json" if pure_mode else "research_program.json",
                "agenda/research_agenda.json" if pure_mode else "research_program.json",
                "datasets/search_plan.json" if pure_mode else "datasets/search_queries.json",
                "datasets/candidates.json",
                "datasets/selected_dataset.json",
                "datasets/selection_rationale.json" if pure_mode else "datasets/selected_dataset.json",
                "datasets/selected_dataset.csv",
                "protocol/generated_protocol.json",
                "generated/experiment.py",
                "generated/static_validation.json",
            ]
        )
    else:
        required.extend(["literature/related_work.md", "literature/sources.json", "literature/gaps.json", "literature/literature_brief.json"])
    for rel in required:
        if not (workspace / rel).exists():
            failures.append(f"missing {rel}")
    if not any((workspace / "protocols").glob("*.json")):
        failures.append("missing protocols/*.json")
    if not any((workspace / "runs").glob("*.json")):
        failures.append("missing runs/*.json")
    if not any((workspace / "analyses").glob("*.json")):
        failures.append("missing analyses/*.json")

    predictions = workspace / "predictions" / "world_model_predictions.jsonl"
    if not predictions.exists():
        failures.append("missing predictions/world_model_predictions.jsonl")
    else:
        lines = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            failures.append("world_model_predictions.jsonl is empty")
        if require_qwen and not any(item.get("prediction", {}).get("source") == "qwen_agentworld" for item in lines):
            failures.append("no qwen_agentworld prediction found")

    ledger_path = workspace / "claim_ledger.json"
    if ledger_path.exists():
        claims = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not claims:
            failures.append("claim_ledger.json has no claims")
        if not any(claim.get("verdict") in {"blocked", "supported", "weakly_supported", "inconclusive", "observation_only"} for claim in claims):
            failures.append("claim_ledger.json has no verifier verdict")

    notebook_path = workspace / "notebook.jsonl"
    if notebook_path.exists():
        entries = [json.loads(line) for line in notebook_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not entries:
            failures.append("notebook.jsonl is empty")
        else:
            first = entries[0]
            for key in ["qwen_predictions", "actual_observation", "prediction_comparison", "claim_updates"]:
                if key not in first:
                    failures.append(f"notebook entry missing {key}")
            if require_agent and not open_mode and not any((entry.get("scientist_decision") or {}).get("source") == "llm" for entry in entries):
                failures.append("no llm scientist_decision found")
            if len(entries) < 3 and require_agent:
                failures.append("require-agent final demo expects at least 3 notebook steps")
    if open_mode:
        validation_path = workspace / "generated" / "static_validation.json"
        if validation_path.exists():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if validation.get("status") != "accepted":
                failures.append("generated code static validation was not accepted")
    if pure_mode and not (workspace / "next_decision.json").exists():
        failures.append("missing next_decision.json")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--require-qwen", action="store_true")
    parser.add_argument("--require-agent", action="store_true")
    args = parser.parse_args()
    workspace = Path(args.workspace) if args.workspace else latest_workspace()
    if workspace is None:
        raise SystemExit("No reports/lab workspace found.")
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    failures = validate(workspace, args.require_qwen, args.require_agent)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print(f"OK: lab artifacts valid in {workspace.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
