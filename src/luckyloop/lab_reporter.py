from __future__ import annotations

import json
from pathlib import Path

from .schemas import LabClaim, LabQuestion, ProtocolSpec, ResearchHypothesis


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_reproducibility(workspace: Path, commands: list[str]) -> None:
    lines = [
        "# Reproducibility",
        "",
        "Run commands used by the lab:",
        "",
        "```bash",
        *commands,
        "```",
        "",
        "Qwen-AgentWorld predictions are pre-execution forecasts. Scientific evidence comes from the Python experiment artifacts.",
    ]
    (workspace / "reproducibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    workspace: Path,
    lab_question: LabQuestion,
    hypotheses: list[ResearchHypothesis],
    protocols: list[ProtocolSpec],
    claims: list[LabClaim],
) -> Path:
    supported = [claim for claim in claims if claim.verdict in {"supported", "weakly_supported"}]
    blocked = [claim for claim in claims if claim.verdict == "blocked"]
    observation = [claim for claim in claims if claim.verdict in {"inconclusive", "observation_only"}]
    has_split_literature = (workspace / "literature" / "domain_related_work.md").exists()
    lines = [
        "# Auto-Research Report",
        "",
        "## Abstract",
        "",
        "This report summarizes an autonomous ML research loop: literature review, dataset selection, generated protocol, real execution, claim verification, and follow-up decision.",
        "",
        "## Research Question",
        "",
        lab_question.question,
        "",
    ]
    if has_split_literature:
        lines += [
            "## Domain Related Work",
            "",
            f"- Domain related work: `{_rel(workspace / 'literature' / 'domain_related_work.md', workspace)}`",
            f"- Domain sources: `{_rel(workspace / 'literature' / 'domain_sources.json', workspace)}`",
            f"- Domain gaps: `{_rel(workspace / 'literature' / 'domain_gaps.json', workspace)}`",
            "",
            "## Method Safeguards",
            "",
            f"- Method related work: `{_rel(workspace / 'literature' / 'method_related_work.md', workspace)}`",
            f"- Method sources: `{_rel(workspace / 'literature' / 'method_sources.json', workspace)}`",
        ]
    else:
        lines += [
            "## Literature Review Summary",
            "",
            f"- Related work: `{_rel(workspace / 'literature' / 'related_work.md', workspace)}`",
            f"- Sources: `{_rel(workspace / 'literature' / 'sources.json', workspace)}`",
        ]
    agenda_path = workspace / "agenda" / "research_agenda.json"
    if agenda_path.exists():
        lines += ["", "## Research Agenda", "", f"- Agenda: `{_rel(agenda_path, workspace)}`"]
    lines += ["", "## Hypotheses", ""]
    for hypothesis in hypotheses:
        lines += [
            f"### {hypothesis.hypothesis_id}",
            "",
            f"- Candidate claim: {hypothesis.claim_candidate}",
            f"- Falsification condition: {hypothesis.falsification_condition}",
            f"- Minimum evidence: {hypothesis.minimum_evidence_needed}",
            "",
        ]
    lines += ["## Methods", ""]
    for protocol in protocols:
        lines += [
            f"### {protocol.protocol_id}",
            "",
            f"- Goal: {protocol.scientific_goal}",
            f"- Conditions: {', '.join(protocol.conditions)}",
            f"- Primary metric: `{protocol.primary_metric}`",
            f"- Risks: {', '.join(protocol.protocol_risks) or 'none declared'}",
            "",
        ]
    selection_path = workspace / "datasets" / "selection_rationale.json"
    if selection_path.exists():
        lines += ["## Dataset Selection", "", f"- Selection rationale: `{_rel(selection_path, workspace)}`", ""]
    lines += [
        "## Results And Verification",
        "",
        "- Real run outputs: `runs/`",
        "- Analyses: `analyses/`",
        "- Full claim ledger: `claim_ledger.json`",
        "",
        "## Lucky Loop Audit Trail",
        "",
        "- Prediction log: `predictions/world_model_predictions.jsonl`",
        "- Lab notebook: `notebook.jsonl`",
        "- Generated code validation: `generated/static_validation.json`",
        "- Next decision: `next_decision.json`",
        "",
        "## Supported Claims",
        "",
    ]
    if supported:
        for claim in supported:
            lines.append(f"- **{claim.verdict}**: {claim.claim} Evidence: {', '.join(claim.evidence_ids)}. Reason: {claim.reason}")
    else:
        lines.append("- None.")
    lines += ["", "## Claims Not Supported By This Run", ""]
    if blocked:
        for claim in blocked:
            category = f" `{claim.failure_category}`" if claim.failure_category else ""
            lines += [
                f"- **not supported{category}**: {claim.claim}",
                f"  - Diagnostic: {claim.diagnostic or claim.reason}",
                f"  - Next action: {claim.next_action or 'Run a targeted follow-up before making a stronger claim.'}",
                f"  - Allowed rewrite: {claim.allowed_rewrite or 'Report the result as an observation bounded by the evidence.'}",
            ]
    else:
        lines.append("- None.")
    lines += ["", "## Inconclusive / Observation-Only Findings", ""]
    if observation:
        for claim in observation:
            lines.append(f"- **{claim.verdict}**: {claim.claim}. Reason: {claim.reason}")
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Claim Ledger",
        "",
        "- Full ledger: `claim_ledger.json`",
        "",
        "## Limitations",
        "",
        "- Literature review uses arXiv metadata, abstracts, and curated notes; this run does not parse full PDFs.",
        "- Qwen-AgentWorld predictions are not evidence. They are pre-execution forecasts of lab observations.",
        "- Generated experiment code is executed only after static validation and sandboxed local execution.",
        "",
        "## Reproducibility Commands",
        "",
        "- See `reproducibility.md`.",
    ]
    path = workspace / "final_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (workspace / "claim_ledger.json").write_text(
        json.dumps([claim.model_dump() for claim in claims], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
