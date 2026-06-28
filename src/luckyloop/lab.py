from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .lab_notebook import append_notebook_entry, write_json, write_predictions
from .lab_protocols import (
    SUPPORTED_STUDIES,
    candidate_actions_for_state,
    hypotheses_for,
    infer_study_id,
    make_lab_question,
    protocols_for,
    slugify,
    stop_action,
)
from .lab_reporter import write_final_report, write_reproducibility
from .lab_scientist import decide_next_action, infer_protocol_family
from .lab_verifier import analyze_observation, verify_lab_claims
from .lab_world_model import predict_lab_action
from .literature import synthesize_context, write_context
from .schemas import (
    LabAction,
    LabNotebookEntry,
    LabObservation,
    LabPrediction,
    LabScientistDecision,
    LabStudyResult,
    LabStudyState,
)
from .tasks import ROOT


def _tail(text: str, n: int = 2000) -> str:
    return text[-n:] if text else ""


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_question(workspace: Path, question: str, study_id: str) -> None:
    lines = [
        "# Research Question",
        "",
        question,
        "",
        "## Study",
        "",
        f"- Study id: `{study_id}`",
        "- Domain: `ml_research_validity`",
        "- Qwen-AgentWorld predicts computer-lab observations before execution.",
        "- Python executes the real ML experiment.",
    ]
    (workspace / "question.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _literature_inference_context(context) -> dict:
    return {
        "queries": context.queries,
        "included_sources": [
            {
                "citation_id": paper.citation_id,
                "title": paper.title,
                "url": paper.url,
                "abstract": paper.abstract[:1200],
                "used_for": paper.used_for,
                "relevance_score": paper.relevance_score,
            }
            for paper in context.papers[:10]
        ],
        "gap_findings": [asdict(gap) for gap in context.gap_findings],
        "recommended_metrics": context.recommended_metrics,
        "recommended_experiment_plan": context.recommended_experiment_plan,
    }


def _write_gaps(workspace: Path, study_id: str, context) -> None:
    study_gaps = {
        "split_validity_sensor": {
            "gap_id": "gap_split_validity_sensor",
            "claim": "Random splits can inflate apparent generalization on temporally or subject-correlated sensor data.",
            "protocol": "random_vs_blocked_split",
        },
        "leakage_trap": {
            "gap_id": "gap_leakage_trap",
            "claim": "Preprocessing outside train folds can leak target or test information into the model.",
            "protocol": "proper_vs_leaky_preprocessing",
        },
        "metric_misuse_imbalance": {
            "gap_id": "gap_metric_misuse_imbalance",
            "claim": "Accuracy can hide minority-class failure under imbalance.",
            "protocol": "accuracy_vs_balanced_metrics",
        },
        "seed_variance_claim": {
            "gap_id": "gap_seed_variance_claim",
            "claim": "Single-run winners are not robust claims when seed noise is large.",
            "protocol": "single_run_vs_repeated_seeds",
        },
        "small_data_complexity": {
            "gap_id": "gap_small_data_complexity",
            "claim": "Complex models can fail to produce claimable gains in small-data regimes.",
            "protocol": "simple_vs_complex_small_data",
        },
    }
    payload = {
        "schema_version": "1.0",
        "study_id": study_id,
        "literature_limit": "metadata, abstracts, curated notes, and arXiv IDs; full PDFs are not parsed in v1",
        "study_gap": study_gaps[study_id],
        "autoresearch_gaps": [asdict(gap) for gap in context.gap_findings],
        "source_to_gap_to_protocol": [
            {
                "source_ids": [paper.citation_id for paper in context.papers[:6]],
                "gap_id": study_gaps[study_id]["gap_id"],
                "protocol": study_gaps[study_id]["protocol"],
            }
        ],
    }
    write_json(workspace / "literature" / "gaps.json", payload)
    brief = {
        "schema_version": "1.0",
        "study_id": study_id,
        "known_methods": [
            "safe protocol catalog",
            "repeated-seed evaluation",
            "effect-vs-noise claim verification",
        ],
        "known_failure_modes": [
            "single-run overclaim",
            "data leakage",
            "metric misuse",
            "split-induced overstatement",
            "seed variance",
        ],
        "methodological_risks": study_gaps[study_id]["claim"],
        "candidate_gaps": [payload["study_gap"]],
        "suggested_hypotheses": [study_gaps[study_id]["claim"]],
        "suggested_protocol_families": [study_id],
        "literature_limit": payload["literature_limit"],
    }
    write_json(workspace / "literature" / "literature_brief.json", brief)


def create_literature(workspace: Path, question: str, study_id: str, context=None):
    if context is None:
        context = synthesize_context(f"{question} ML research validity protocol leakage split metric seed variance")
    literature_dir = workspace / "literature"
    write_context(context, literature_dir)
    _write_gaps(workspace, study_id, context)
    return context


def infer_study_from_literature(
    question: str,
    context,
    *,
    requested: str | None,
    planner: str,
    require_agent: bool,
) -> tuple[str, dict]:
    if requested:
        study_id = infer_study_id(question, requested)
        return study_id, {
            "study_id": study_id,
            "source": "user_override",
            "rationale": "`--study` was provided explicitly for a debug or ablation run.",
            "prompt_version": "user_override",
        }
    if planner == "llm":
        inference = infer_protocol_family(
            question,
            _literature_inference_context(context),
            sorted(SUPPORTED_STUDIES),
            planner=planner,
            require_agent=require_agent,
        )
        if inference.get("study_id"):
            return str(inference["study_id"]), inference
    if require_agent:
        raise RuntimeError("Study inference requires the LLM scientist planner in production mode.")
    study_id = infer_study_id(question)
    return study_id, {
        "study_id": study_id,
        "source": "keyword_fallback_dev_only",
        "rationale": "Development fallback used because the LLM scientist planner was not required.",
        "prompt_version": "keyword_fallback_dev_only",
    }


def execute_action(action: LabAction, workspace: Path, timeout: int = 240) -> LabObservation:
    start = time.perf_counter()
    if action.kind == "stop_and_report":
        raw = {"status": "success", "message": "stop_and_report"}
        return LabObservation(
            status="success",
            action_id=action.action_id,
            protocol_id=action.protocol_id,
            raw=raw,
            stdout_tail="stop_and_report: no further lab compute requested",
            runtime_seconds=0.0,
        )
    proc = subprocess.run(
        action.command.replace("python ", f"{sys.executable} ", 1),
        shell=True,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    runtime = round(time.perf_counter() - start, 4)
    raw = {"status": "failed", "returncode": proc.returncode}
    try:
        raw = json.loads(proc.stdout[proc.stdout.find("{") :])
    except Exception:
        raw["stdout_tail"] = _tail(proc.stdout)
        raw["stderr_tail"] = _tail(proc.stderr)
    return LabObservation(
        status=str(raw.get("status", "success" if proc.returncode == 0 else "failed")),
        action_id=action.action_id,
        protocol_id=action.protocol_id,
        raw=raw,
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
        artifacts=[str(item) for item in raw.get("artifacts", [])],
        runtime_seconds=raw.get("runtime_seconds", runtime),
    )


def compare_prediction(prediction: LabPrediction, observation: LabObservation) -> dict:
    observed_text = json.dumps(observation.raw, sort_keys=True)[:3000].lower()
    predicted = " ".join(
        [
            prediction.predicted_terminal_observation,
            prediction.expected_result_pattern,
            " ".join(prediction.protocol_risks),
            " ".join(prediction.failure_modes),
            prediction.runtime_risk,
            prediction.suggested_modification,
            prediction.decision_threshold,
        ]
    ).lower()
    matched_risks = [risk for risk in prediction.protocol_risks if risk.lower() in observed_text]
    warning_text = " ".join(str(item).lower() for item in observation.raw.get("protocol_warnings", []))
    predicted_warning = any(risk.lower() in warning_text for risk in prediction.protocol_risks)
    return {
        "prediction_source": prediction.source,
        "status_match": observation.status == "success" and prediction.recommendation in {"run", "verify"},
        "matched_protocol_risks": matched_risks,
        "predicted_warning_observed": predicted_warning,
        "prediction_had_specific_content": bool(predicted.strip()),
        "prediction_had_compute_signal": bool(
            prediction.runtime_risk.strip()
            or prediction.expected_runtime_seconds is not None
            or prediction.expected_runtime_range_seconds
            or prediction.compute_waste_risk > 0
            or prediction.suggested_modification.strip()
        ),
        "expected_runtime_seconds": prediction.expected_runtime_seconds,
        "compute_waste_risk": prediction.compute_waste_risk,
        "value_of_information": prediction.value_of_information,
    }


def _state_summary(state: LabStudyState) -> str:
    return (
        f"study={state.lab_question.study_id}; completed_actions={len(state.completed_actions)}; "
        f"observations={len(state.observations)}; claims={len(state.claims)}"
    )


def select_action(
    policy: str,
    actions: list[LabAction],
    predictions: list[LabPrediction],
    state: LabStudyState,
    scientist_decision: LabScientistDecision | None = None,
) -> tuple[LabAction, LabPrediction, str]:
    if not actions:
        action = stop_action(len(state.completed_actions) + 1)
        return action, LabPrediction(recommendation="stop_and_report", expected_claim_delta="report_ready"), "No actions remained."
    if policy == "classic_score_chaser":
        return actions[0], predictions[0], "Classic score-chaser runs the first score-producing protocol."
    if policy == "classic_verified":
        return actions[0], predictions[0], "Classic verified runs the protocol and relies on the deterministic verifier after execution."
    by_id = {action.action_id: (action, prediction) for action, prediction in zip(actions, predictions)}
    if scientist_decision and scientist_decision.preferred_action_id in by_id:
        action, prediction = by_id[scientist_decision.preferred_action_id]
        if prediction.recommendation not in {"skip", "modify"}:
            return action, prediction, (
                "Scientist planner selected this action; Qwen prediction did not veto it. "
                f"Planner rationale: {scientist_decision.rationale}"
            )
    scored = []
    for action, prediction in zip(actions, predictions):
        score = 0
        if prediction.expected_claim_delta in {"enables_claim", "blocks_or_rewrites_claim", "reduces_uncertainty"}:
            score += 3
        if prediction.recommendation in {"run", "verify"}:
            score += 2
        if action.protocol_risks:
            score += 1
        if prediction.recommendation in {"skip", "modify"}:
            score -= 2
        scored.append((score, action, prediction))
    scored.sort(key=lambda item: item[0], reverse=True)
    score, action, prediction = scored[0]
    return action, prediction, f"Qwen-conditioned selector chose highest claim-value lab action with score={score}."


def build_research_program(question: str, study_id: str, hypotheses, protocols) -> dict:
    return {
        "schema_version": "1.0",
        "question": question,
        "study_id": study_id,
        "domain": "ml_research_validity",
        "goal": "Run a complete computational ML research lab from literature to verified claims.",
        "hypotheses": [item.model_dump() for item in hypotheses],
        "protocols": [item.model_dump() for item in protocols],
        "pipeline": [
            "research_intake",
            "literature_review",
            "gap_mapping",
            "hypothesis_generation",
            "protocol_design",
            "qwen_prediction_before_action",
            "real_ml_execution",
            "analysis",
            "claim_verification",
            "lab_notebook",
            "final_report",
        ],
    }


def run_lab(
    question: str,
    study_id: str | None = None,
    budget: int = 8,
    require_qwen: bool = True,
    policy: str = "lucky_loop_lab",
    planner: str = "llm",
    require_agent: bool = True,
) -> LabStudyResult:
    if require_agent and planner != "llm":
        raise RuntimeError("--require-agent requires --planner llm.")
    if policy != "lucky_loop_lab" and (require_agent or require_qwen):
        raise RuntimeError("Classic policies are debug ablations; disable required real agents to run them.")
    if study_id is None and policy == "lucky_loop_lab":
        from .open_lab import run_open_lab

        return run_open_lab(question, budget=budget, require_qwen=require_qwen, require_agent=require_agent)
    context = synthesize_context(f"{question} ML research validity protocol leakage split metric seed variance")
    resolved_study_id, study_inference = infer_study_from_literature(
        question,
        context,
        requested=study_id,
        planner=planner,
        require_agent=require_agent,
    )
    lab_question = make_lab_question(question, resolved_study_id, budget, require_qwen)
    slug = slugify(f"{lab_question.study_id}-{question}")
    workspace = ROOT / "reports" / "lab" / slug
    workspace.mkdir(parents=True, exist_ok=True)
    for rel in [
        "notebook.jsonl",
        "claim_ledger.json",
        "final_report.md",
        "reproducibility.md",
        "study_result.json",
    ]:
        path = workspace / rel
        if path.exists():
            path.unlink()
    for rel in ["predictions", "runs", "analyses", "protocols"]:
        path = workspace / rel
        if path.exists():
            shutil.rmtree(path)
    _write_question(workspace, lab_question.question, lab_question.study_id)
    context = create_literature(workspace, lab_question.question, lab_question.study_id, context)
    write_json(workspace / "literature" / "study_inference.json", study_inference)
    literature_brief = json.loads((workspace / "literature" / "literature_brief.json").read_text(encoding="utf-8"))
    literature_brief["study_inference"] = study_inference
    write_json(workspace / "literature" / "literature_brief.json", literature_brief)
    hypotheses = hypotheses_for(lab_question.study_id)
    protocols = protocols_for(lab_question.study_id)
    write_json(workspace / "hypotheses.json", [item.model_dump() for item in hypotheses])
    for index, protocol in enumerate(protocols, start=1):
        write_json(workspace / "protocols" / f"protocol_{index:03d}.json", protocol)
    program = build_research_program(question, lab_question.study_id, hypotheses, protocols)
    program["study_inference"] = study_inference
    write_json(workspace / "research_program.json", program)

    state = LabStudyState(
        lab_question=lab_question,
        hypotheses=hypotheses,
        protocols=protocols,
        summary=f"Literature sources={len(context.papers)}; study={lab_question.study_id}.",
    )
    executed_commands: list[str] = []
    step = 1
    protocol_by_id = {protocol.protocol_id: protocol for protocol in protocols}
    hypothesis_by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}

    while step <= max(1, budget):
        actions = candidate_actions_for_state(state, step, workspace)
        scientist_decision = None
        if policy == "lucky_loop_lab":
            scientist_decision = decide_next_action(
                state,
                actions,
                literature_brief,
                planner=planner,
                require_agent=require_agent,
            )
        predictions = [
            predict_lab_action(action, state, require_qwen=require_qwen and policy == "lucky_loop_lab")
            if policy == "lucky_loop_lab"
            else LabPrediction(
                source="plumbing_not_called",
                predicted_terminal_observation="Baseline policy does not call Qwen-AgentWorld.",
                predicted_artifacts=action.expected_artifacts,
                protocol_risks=action.protocol_risks,
                expected_claim_delta=action.claim_delta_target,
                recommendation="run" if action.kind != "stop_and_report" else "stop_and_report",
            )
            for action in actions
        ]
        prediction_payloads = [
            {"action": action.model_dump(), "prediction": prediction.model_dump()}
            for action, prediction in zip(actions, predictions)
        ]
        write_predictions(workspace, step, prediction_payloads)
        selected, selected_prediction, selection_reason = select_action(policy, actions, predictions, state, scientist_decision)
        executed_commands.append(selected.command)
        observation = execute_action(selected, workspace)
        analysis = None
        claim_updates = []
        if selected.protocol_id and observation.status == "success":
            analysis = analyze_observation(observation.raw, f"analysis_{step:03d}")
            write_json(workspace / "analyses" / f"analysis_{step:03d}.json", analysis)
            protocol = protocol_by_id[selected.protocol_id]
            hypothesis = hypothesis_by_id.get(protocol.hypothesis_id)
            claim_updates = verify_lab_claims(protocol, analysis, hypothesis, f"experiment_{step:03d}")
            state.analyses.append(analysis)
            state.claims.extend(claim_updates)
            if selected.kind == "inspect_dataset":
                state.completed_actions.append(f"inspect_dataset:{protocol.dataset}")
            elif selected.kind == "run_baseline":
                state.completed_actions.append(f"baseline:{protocol.protocol_id}")
            elif selected.kind == "run_protocol":
                state.completed_actions.append(f"main:{protocol.protocol_id}")
            elif selected.kind == "run_replication":
                state.completed_actions.append(f"replication:{protocol.protocol_id}")
            elif selected.kind == "run_ablation":
                state.completed_actions.append(f"control:{protocol.protocol_id}")
        state.observations.append(observation)
        comparison = compare_prediction(selected_prediction, observation)
        next_decision = "stop_and_report" if selected.kind == "stop_and_report" else "continue_or_report_from_verified_evidence"
        entry = LabNotebookEntry(
            step=step,
            hypothesis_id=selected.hypothesis_id,
            state_before=_state_summary(state),
            candidate_actions=actions,
            scientist_decision=scientist_decision,
            qwen_predictions=[payload["prediction"] for payload in prediction_payloads],
            selected_action=selected,
            why_world_model_mattered=selection_reason if selected_prediction.source == "qwen_agentworld" else "No Qwen signal used by this policy.",
            actual_observation=observation,
            prediction_comparison=comparison,
            analysis=analysis,
            claim_updates=claim_updates,
            next_decision=next_decision,
        )
        append_notebook_entry(workspace, entry)
        if selected.kind == "stop_and_report":
            break
        step += 1

    write_json(workspace / "claim_ledger.json", [claim.model_dump() for claim in state.claims])
    write_reproducibility(workspace, executed_commands)
    final_report = write_final_report(workspace, lab_question, hypotheses, protocols, state.claims)
    result = LabStudyResult(
        workspace=_rel(workspace),
        lab_question=lab_question,
        hypotheses=hypotheses,
        protocols=protocols,
        claims=state.claims,
        final_report=_rel(final_report),
    )
    write_json(workspace / "study_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a complete Lucky Loop ML research validity lab.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--study", default=None, help="Debug override. Production runs infer this from the question and literature.")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--require-qwen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--planner", choices=["deterministic", "llm"], default="llm")
    parser.add_argument("--require-agent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--policy",
        choices=["classic_score_chaser", "classic_verified", "lucky_loop_lab"],
        default="lucky_loop_lab",
        help="Debug ablation policy. Production default is lucky_loop_lab.",
    )
    args = parser.parse_args()
    result = run_lab(args.question, args.study, args.budget, args.require_qwen, args.policy, args.planner, args.require_agent)
    print(json.dumps(result.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
