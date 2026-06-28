#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def latest_workspace() -> Path:
    root = ROOT / "reports" / "lab"
    workspaces = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not workspaces:
        raise SystemExit("No reports/lab workspace found.")
    return max(workspaces, key=lambda path: path.stat().st_mtime)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt_claims(claims: list[dict]) -> list[str]:
    if not claims:
        return ["- None."]
    lines = []
    for claim in claims:
        category = f" `{claim.get('failure_category')}`" if claim.get("failure_category") else ""
        lines.append(f"- **{claim.get('verdict')}{category}**: {claim.get('claim')}")
        if claim.get("diagnostic") or claim.get("reason"):
            lines.append(f"  - Diagnostic: {claim.get('diagnostic') or claim.get('reason')}")
        if claim.get("next_action"):
            lines.append(f"  - Next action: {claim.get('next_action')}")
        if claim.get("allowed_rewrite"):
            lines.append(f"  - Allowed rewrite: {claim.get('allowed_rewrite')}")
    return lines


def build_demo(workspace: Path) -> str:
    question = (workspace / "question.md").read_text(encoding="utf-8") if (workspace / "question.md").exists() else ""
    program = _read_json(workspace / "research_program.json")
    open_mode = program.get("mode") in {"open_generated_ml_research", "pure_auto_research_ml_first"}
    pure_mode = program.get("mode") == "pure_auto_research_ml_first"
    brief_path = workspace / "literature" / "literature_brief.json"
    brief = _read_json(brief_path) if brief_path.exists() else {}
    study_inference_path = workspace / "literature" / "study_inference.json"
    study_inference = _read_json(study_inference_path) if study_inference_path.exists() else {}
    hypotheses = _read_json(workspace / "hypotheses.json")
    claims = _read_json(workspace / "claim_ledger.json")
    notebook = _read_jsonl(workspace / "notebook.jsonl")
    question_text = (brief.get("question") or "").strip()
    if not question_text:
        question_lines = [line.strip() for line in question.splitlines() if line.strip() and not line.startswith("#") and not line.startswith("-")]
        question_text = question_lines[0] if question_lines else "Your research question here"
    lines = [
        "# Complete Lucky Loop Demo",
        "",
        "## 1. Setup",
        "",
        f"- Workspace: `{workspace.relative_to(ROOT)}`",
        "- Scientist planner: DeepSeek through `LUCKYLOOP_AGENT_*` by default.",
        "- World model: Qwen-AgentWorld through `LUCKYWORLD_SIMULATOR_*` by default.",
        "- Experimental truth: generated Python + sklearn execution artifacts.",
        "- Claim authority: deterministic verifier and `claim_ledger.json`.",
        "",
        "## 2. Research Question",
        "",
        question.strip() or "- Missing question artifact.",
        "",
        "## 3. Research Setup",
        "",
    ]
    if open_mode:
        dataset = program.get("dataset_selection", {})
        candidate = dataset.get("candidate", {})
        audit = dataset.get("audit", {})
        generated_protocol = program.get("generated_protocol", {})
        agenda = program.get("research_agenda", {})
        selected_hypothesis = agenda.get("selected_hypothesis_id", "n/a")
        lines += [
            f"- Mode: `{program.get('mode')}`",
            f"- Domain literature sources: {program.get('domain_literature_sources', program.get('literature_sources'))}",
            f"- Method literature sources: {program.get('method_literature_sources', 'n/a')}",
            f"- Selected hypothesis: `{selected_hypothesis}`",
            f"- Selected dataset: `{candidate.get('dataset_id')}` from `{candidate.get('source')}`",
            f"- Dataset rows/features: {audit.get('n_rows')} rows, {audit.get('n_features')} features",
            f"- Target column: `{audit.get('target_column')}`",
            f"- Generated hypothesis: {generated_protocol.get('hypothesis')}",
            "- Dataset search: derived from research agenda, then Hugging Face/OpenML candidates are audited.",
            "",
        ]
        if pure_mode and (workspace / "next_decision.json").exists():
            next_decision = _read_json(workspace / "next_decision.json")
            lines += [
                "## 3b. Next Research Decision",
                "",
                f"- Decision: `{next_decision.get('decision')}`",
                f"- Rationale: {next_decision.get('rationale')}",
                f"- Next action goal: {next_decision.get('next_action_goal')}",
                "",
            ]
    else:
        lines += [
            f"- Study id: `{brief.get('study_id')}`",
            f"- Study inference source: `{study_inference.get('source', 'n/a')}`",
            f"- Study inference rationale: {study_inference.get('rationale', 'n/a')}",
            f"- Methodological risk: {brief.get('methodological_risks')}",
            f"- Suggested protocol families: {', '.join(brief.get('suggested_protocol_families', []))}",
            f"- Literature limit: {brief.get('literature_limit')}",
            "",
        ]
    lines += ["## 4. Hypotheses", ""]
    for hypothesis in hypotheses:
        lines += [
            f"- `{hypothesis.get('hypothesis_id')}`: {hypothesis.get('claim_candidate')}",
            f"  - Falsification: {hypothesis.get('falsification_condition')}",
            f"  - Evidence needed: {hypothesis.get('minimum_evidence_needed')}",
        ]
    lines += ["", "## 5. Lab Timeline", ""]
    for entry in notebook:
        decision = entry.get("scientist_decision") or {}
        selected = entry.get("selected_action") or {}
        observation = entry.get("actual_observation") or {}
        comparison = entry.get("prediction_comparison") or {}
        qwen_predictions = entry.get("qwen_predictions") or []
        qwen_sources = sorted({prediction.get("source", "unknown") for prediction in qwen_predictions})
        decision_source = decision.get("source", "n/a")
        lines += [
            f"### Step {entry.get('step')}",
            "",
            f"- Candidate actions: {len(entry.get('candidate_actions') or [])}",
            f"- Scientist decision source: `{decision_source}`",
            f"- Scientist choice: `{decision.get('preferred_action_id', 'n/a')}`",
            f"- Scientist rationale: {decision.get('rationale', 'n/a')}",
            f"- Qwen predictions: {len(qwen_predictions)} ({', '.join(qwen_sources) or 'none'})",
            f"- Selected action: `{selected.get('action_id')}` / `{selected.get('kind')}`",
            f"- Real observation status: `{observation.get('status')}`",
            f"- Prediction comparison: {json.dumps(comparison, sort_keys=True)}",
            f"- Claim updates: {len(entry.get('claim_updates') or [])}",
            "",
        ]
    lines += [
        "## 6. Claim Ledger",
        "",
        *_fmt_claims(claims),
        "",
        "## 7. Why The World Model Mattered",
        "",
        "Qwen-AgentWorld was queried before lab actions to predict the likely computer-lab observation, artifacts, failure modes, protocol risks, and claim impact. It did not provide evidence; it helped decide what compute was worth spending before Python executed the real experiment.",
        "",
        "## 8. Repro Commands",
        "",
        "```bash",
        "set -a; source .env; set +a",
        "PYTHONPATH=src .venv/bin/python -m luckyloop.lab \\",
        f"  --question {json.dumps(question_text)} \\",
        "  --budget 6",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--out", default="reports/demo_complete_lab.md")
    args = parser.parse_args()
    workspace = Path(args.workspace) if args.workspace else latest_workspace()
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_demo(workspace), encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
