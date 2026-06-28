from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import csv
from pathlib import Path

from .code_safety import write_validated_code
from .dataset_discovery import discover_dataset_candidates, select_and_materialize_dataset
from .lab import _literature_inference_context, _rel, _tail, compare_prediction
from .lab_notebook import append_notebook_entry, write_json, write_predictions
from .lab_reporter import write_final_report, write_reproducibility
from .lab_scientist import decide_next_research_step, generate_experiment_code, generate_open_protocol, generate_research_agenda
from .lab_verifier import analyze_observation, verify_lab_claims
from .lab_world_model import predict_lab_action
from .literature import (
    derive_literature_query_plan,
    synthesize_domain_context,
    synthesize_method_context,
    write_split_context,
)
from .schemas import (
    DatasetAudit,
    GeneratedResearchProtocol,
    DatasetSearchPlan,
    HypothesisCandidate,
    LabAction,
    LabNotebookEntry,
    LabObservation,
    LabPrediction,
    LabQuestion,
    LabStudyResult,
    LabStudyState,
    ProtocolSpec,
    ResearchAgenda,
    ResearchHypothesis,
)
from .tasks import ROOT


def run_open_lab(
    question: str,
    *,
    budget: int = 8,
    require_qwen: bool = True,
    require_agent: bool = True,
) -> LabStudyResult:
    slug = _slug(question)
    workspace = ROOT / "reports" / "lab" / slug
    _clean_workspace(workspace)
    _log_event(workspace, "run_started", "Open auto-research run started.", question=question, budget=budget)
    lab_question = LabQuestion(
        question=question,
        study_id="open_generated_ml_research",
        budget=budget,
        require_qwen=require_qwen,
        success_criteria=[
            "external dataset discovery finds a usable Hugging Face or OpenML dataset",
            "the research protocol is generated from the question, literature, and dataset audit",
            "the experiment code is generated, statically validated, dry-run, executed, analyzed, verified, and reported",
        ],
    )
    _write_open_question(workspace, question)
    _log_event(workspace, "literature_started", "Searching domain and method literature contexts.")
    query_plan = derive_literature_query_plan(question)
    domain_context = synthesize_domain_context(question, query_plan)
    method_context = synthesize_method_context(question, query_plan)
    write_split_context(domain_context, method_context, workspace / "literature", query_plan)
    _log_event(
        workspace,
        "literature_completed",
        "Domain and method literature contexts written.",
        domain_source_count=len(domain_context.papers),
        method_source_count=len(method_context.papers),
    )
    domain_literature_context = _literature_inference_context(domain_context)
    method_literature_context = _literature_inference_context(method_context)

    state = LabStudyState(
        lab_question=lab_question,
        summary=f"Pure auto-research lab; domain_sources={len(domain_context.papers)}; method_sources={len(method_context.papers)}.",
    )
    executed_commands: list[str] = []
    notebook_step = 1

    agenda = _run_logged_stage(
        workspace,
        state,
        notebook_step,
        _stage_action(notebook_step, "draft_protocol", "Build a domain-specific research agenda with multiple hypotheses."),
        lambda: _generate_agenda(question, domain_literature_context, method_literature_context, workspace, require_agent=require_agent),
        require_qwen=require_qwen,
    )
    notebook_step += 1
    selected_hypothesis = _selected_hypothesis(agenda)
    search_plan = _dataset_search_plan(query_plan.dataset_queries, selected_hypothesis)

    candidates = _run_logged_stage(
        workspace,
        state,
        notebook_step,
        _stage_action(notebook_step, "search_datasets", "Search external datasets on Hugging Face and OpenML."),
        lambda: _discover(question, domain_literature_context, workspace, search_plan),
        require_qwen=require_qwen,
    )
    notebook_step += 1

    selected, audit = _run_logged_stage(
        workspace,
        state,
        notebook_step,
        _stage_action(notebook_step, "audit_dataset", "Audit candidate datasets and materialize one supervised classification table."),
        lambda: _select_dataset(candidates, workspace, search_plan, selected_hypothesis.claim_candidate),
        require_qwen=require_qwen,
    )
    notebook_step += 1

    protocol = _run_logged_stage(
        workspace,
        state,
        notebook_step,
        _stage_action(notebook_step, "draft_protocol", "Generate a free-form ML research protocol from the question, literature, and selected dataset."),
        lambda: _generate_protocol(question, domain_literature_context, method_literature_context, audit, selected_hypothesis, workspace, require_agent=require_agent),
        require_qwen=require_qwen,
    )
    notebook_step += 1

    hypothesis = ResearchHypothesis(
        hypothesis_id=selected_hypothesis.hypothesis_id,
        claim_candidate=protocol.hypothesis or selected_hypothesis.claim_candidate,
        why_it_matters=selected_hypothesis.motivation or "The claim was generated from the domain literature and must survive empirical verification.",
        literature_gap_ids=selected_hypothesis.literature_gap_ids,
        falsification_condition=selected_hypothesis.falsification_condition or protocol.claim_blocked_if,
        minimum_evidence_needed=selected_hypothesis.minimum_evidence_needed or protocol.claim_enabled_if,
    )
    verifier_protocol = _to_protocol_spec(protocol, audit)
    state.hypotheses = [hypothesis]
    state.protocols = [verifier_protocol]
    write_json(workspace / "hypotheses.json", [hypothesis.model_dump()])
    write_json(workspace / "protocols" / "generated_protocol.json", verifier_protocol)
    write_json(workspace / "agenda" / "research_agenda.json", agenda)
    write_json(workspace / "research_program.json", _research_program(question, domain_context, method_context, selected, audit, protocol, agenda))

    script_path, validation = _run_logged_stage(
        workspace,
        state,
        notebook_step,
        _stage_action(notebook_step, "generate_code", "Generate and statically validate experiment code for the audited dataset."),
        lambda: _generate_code(protocol, audit, workspace, require_agent=require_agent),
        require_qwen=require_qwen,
    )
    notebook_step += 1
    if validation.status != "accepted":
        raise RuntimeError(f"Generated code was rejected: {validation.reason}")

    dry_dataset = _make_dry_run_dataset(audit, workspace)
    dry_action = _script_action(notebook_step, "dry_run", script_path, audit, workspace, dry_run=True, dataset_csv=dry_dataset)
    dry_observation = _execute_and_record(workspace, state, notebook_step, dry_action, require_qwen=require_qwen)
    executed_commands.append(dry_action.command)
    notebook_step += 1
    if dry_observation.status != "success":
        script_path, validation = _repair_code(protocol, audit, workspace, dry_observation, require_agent=require_agent)
        if validation.status != "accepted":
            raise RuntimeError(f"Repaired code was rejected: {validation.reason}")

    latest_decision = None
    while notebook_step <= max(7, budget):
        run_kind = "run_protocol" if not state.analyses else "run_replication"
        run_action = _script_action(notebook_step, run_kind, script_path, audit, workspace, dry_run=False)
        observation = _execute_and_record(workspace, state, notebook_step, run_action, require_qwen=require_qwen)
        executed_commands.append(run_action.command)
        notebook_step += 1
        if observation.status != "success":
            latest_decision = {
                "decision": "inspect_failure",
                "rationale": f"Generated experiment failed: {observation.stderr_tail or observation.stdout_tail}",
                "next_action_goal": "Inspect and repair generated experiment code.",
            }
            break

        analysis = analyze_observation(observation.raw, f"analysis_{notebook_step:03d}")
        write_json(workspace / "analyses" / f"analysis_{notebook_step:03d}.json", analysis)
        claims = verify_lab_claims(verifier_protocol, analysis, hypothesis, f"experiment_{notebook_step:03d}")
        state.analyses.append(analysis)
        state.claims.extend(claims)
        latest_decision_obj = _decide_next(
            question,
            agenda,
            protocol,
            analysis,
            claims,
            max(0, budget - notebook_step),
            workspace,
            require_agent=require_agent,
        )
        latest_decision = latest_decision_obj.model_dump()
        if latest_decision_obj.decision in {"stop_and_report", "verify_claim"}:
            break
        if latest_decision_obj.decision in {"search_better_dataset", "revise_protocol", "revise_hypothesis"}:
            _log_event(
                workspace,
                "next_decision_deferred",
                "Next decision requires a broader loop step; stopping with explicit follow-up.",
                decision=latest_decision_obj.decision,
            )
            break

    stop_action = _stage_action(notebook_step, "stop_and_report", "Stop and write the evidence-bounded report.")
    _execute_and_record(workspace, state, notebook_step, stop_action, require_qwen=require_qwen)

    if latest_decision:
        write_json(workspace / "next_decision.json", latest_decision)
    write_json(workspace / "claim_ledger.json", [claim.model_dump() for claim in state.claims])
    write_reproducibility(workspace, executed_commands)
    final_report = write_final_report(workspace, lab_question, [hypothesis], [verifier_protocol], state.claims)
    result = LabStudyResult(
        workspace=_rel(workspace),
        lab_question=lab_question,
        hypotheses=[hypothesis],
        protocols=[verifier_protocol],
        claims=state.claims,
        final_report=_rel(final_report),
    )
    write_json(workspace / "study_result.json", result)
    _log_event(workspace, "run_completed", "Open auto-research run completed.", final_report=_rel(final_report))
    return result


def _discover(question: str, literature_context: dict, workspace: Path, search_plan: DatasetSearchPlan):
    candidates = discover_dataset_candidates(question, literature_context, workspace, search_plan=search_plan)
    return candidates


def _select_dataset(candidates, workspace: Path, search_plan: DatasetSearchPlan, hypothesis_text: str):
    return select_and_materialize_dataset(candidates, workspace, search_plan=search_plan, hypothesis_text=hypothesis_text)


def _generate_agenda(
    question: str,
    domain_context: dict,
    method_context: dict,
    workspace: Path,
    *,
    require_agent: bool,
) -> ResearchAgenda:
    agenda = generate_research_agenda(question, domain_context, method_context, require_agent=require_agent)
    write_json(workspace / "agenda" / "research_agenda.json", agenda)
    return agenda


def _selected_hypothesis(agenda: ResearchAgenda) -> HypothesisCandidate:
    for hypothesis in agenda.hypotheses:
        if hypothesis.hypothesis_id == agenda.selected_hypothesis_id:
            return hypothesis
    return max(agenda.hypotheses, key=lambda item: item.priority_score)


def _dataset_search_plan(dataset_queries: list[str], hypothesis: HypothesisCandidate) -> DatasetSearchPlan:
    requirements = [*hypothesis.dataset_requirements]
    if not requirements:
        requirements = ["supervised classification", "tabular or easily materialized features", "public dataset"]
    return DatasetSearchPlan(
        queries=[*dataset_queries, *requirements, hypothesis.claim_candidate],
        required_properties=requirements,
        rejection_criteria=[
            "no plausible supervised target",
            "too few rows or minority class too small",
            "features cannot be materialized into a safe local table",
        ],
        rationale=f"Dataset search plan derived from selected hypothesis `{hypothesis.hypothesis_id}`.",
    )


def _generate_protocol(
    question: str,
    domain_context: dict,
    method_context: dict,
    audit: DatasetAudit,
    hypothesis: HypothesisCandidate,
    workspace: Path,
    *,
    require_agent: bool,
):
    literature_context = {
        "domain_context": domain_context,
        "method_context": method_context,
        "selected_hypothesis": hypothesis.model_dump(),
    }
    protocol = generate_open_protocol(question, literature_context, audit, hypothesis=hypothesis, require_agent=require_agent)
    write_json(workspace / "protocol" / "generated_protocol.json", protocol)
    return protocol


def _decide_next(
    question: str,
    agenda: ResearchAgenda,
    protocol: GeneratedResearchProtocol,
    analysis,
    claims,
    budget_remaining: int,
    workspace: Path,
    *,
    require_agent: bool,
):
    decision = decide_next_research_step(
        question,
        agenda,
        protocol,
        analysis.model_dump(),
        [claim.model_dump() for claim in claims],
        budget_remaining,
        require_agent=require_agent,
    )
    write_json(workspace / "decisions" / f"decision_{analysis.analysis_id}.json", decision)
    _log_event(
        workspace,
        "next_decision_completed",
        "Scientist selected next research decision.",
        decision=decision.decision,
        budget_remaining=budget_remaining,
    )
    return decision


def _generate_code(protocol: GeneratedResearchProtocol, audit: DatasetAudit, workspace: Path, *, require_agent: bool):
    script_path = workspace / "generated" / "experiment.py"
    attempts = []
    last_validation = None
    for attempt in range(1, 4):
        try:
            payload = generate_experiment_code(protocol, audit, {"attempt": attempt, "previous_attempts": attempts}, require_agent=require_agent)
        except Exception as exc:
            last_validation = {"status": "rejected", "reason": f"code_generation_failed: {type(exc).__name__}: {exc}"}
            attempts.append(
                {
                    "attempt": attempt,
                    "validation": last_validation,
                    "rationale": "Agent code generation did not satisfy the strict JSON contract.",
                }
            )
            write_json(workspace / "generated" / "code_generation_attempts.json", attempts)
            continue
        code = _inject_protocol(payload["code"], protocol)
        validation = write_validated_code(code, script_path)
        last_validation = validation.model_dump()
        attempts.append({"attempt": attempt, "validation": validation.model_dump(), "rationale": payload.get("rationale", "")})
        write_json(workspace / "generated" / "code_generation_attempts.json", attempts)
        write_json(workspace / "generated" / "static_validation.json", validation)
        if validation.status == "accepted":
            return script_path, validation
    write_json(workspace / "generated" / "code_generation_attempts.json", attempts)
    write_json(workspace / "generated" / "static_validation.json", last_validation or {"status": "rejected", "reason": "no code attempts completed"})
    raise RuntimeError("Agent-generated experiment code did not pass validation after 3 attempts.")


def _repair_code(protocol, audit, workspace, observation, *, require_agent: bool):
    repair_context = {"failed_observation": observation.model_dump()}
    payload = generate_experiment_code(protocol, audit, repair_context, require_agent=require_agent)
    script_path = workspace / "generated" / "experiment.py"
    validation = write_validated_code(_inject_protocol(payload["code"], protocol), script_path)
    write_json(workspace / "generated" / "repair_validation.json", validation)
    return script_path, validation


def _inject_protocol(code: str, protocol: GeneratedResearchProtocol) -> str:
    marker = "__PROTOCOL_JSON__"
    protocol_json = json.dumps(protocol.model_dump(), sort_keys=True)
    if marker in code:
        return code.replace(marker, protocol_json)
    return f"PROTOCOL = {protocol_json}\n" + code


def _execute_and_record(workspace: Path, state: LabStudyState, step: int, action: LabAction, *, require_qwen: bool) -> LabObservation:
    _log_event(workspace, "qwen_prediction_started", "Calling Qwen-AgentWorld before action.", step=step, action_id=action.action_id, kind=action.kind)
    prediction = predict_lab_action(action, state, require_qwen=require_qwen)
    _log_prediction(workspace, step, action, prediction)
    write_predictions(workspace, step, [{"action": action.model_dump(), "prediction": prediction.model_dump()}])
    _log_qwen_override_if_needed(workspace, step, action, prediction)
    _log_event(workspace, "action_execution_started", "Executing lab action.", step=step, action_id=action.action_id, kind=action.kind, command=action.command)
    observation = _execute_action(action, workspace=workspace)
    _log_event(workspace, "action_execution_completed", "Lab action completed.", step=step, action_id=action.action_id, status=observation.status, runtime_seconds=observation.runtime_seconds)
    state.observations.append(observation)
    entry = LabNotebookEntry(
        step=step,
        hypothesis_id=action.hypothesis_id,
        state_before=f"open_lab_completed_actions={len(state.completed_actions)}; observations={len(state.observations)}",
        candidate_actions=[action],
        scientist_decision=None,
        qwen_predictions=[prediction.model_dump()],
        selected_action=action,
        why_world_model_mattered="Qwen predicted this generated-lab action before real execution.",
        actual_observation=observation,
        prediction_comparison=compare_prediction(prediction, observation),
        analysis=None,
        claim_updates=[],
        next_decision="continue" if action.kind != "stop_and_report" else "stop_and_report",
    )
    append_notebook_entry(workspace, entry)
    state.completed_actions.append(action.kind)
    return observation


def _run_logged_stage(workspace: Path, state: LabStudyState, step: int, action: LabAction, fn, *, require_qwen: bool):
    _log_event(workspace, "qwen_prediction_started", "Calling Qwen-AgentWorld before internal stage.", step=step, action_id=action.action_id, kind=action.kind)
    prediction = predict_lab_action(action, state, require_qwen=require_qwen)
    _log_prediction(workspace, step, action, prediction)
    write_predictions(workspace, step, [{"action": action.model_dump(), "prediction": prediction.model_dump()}])
    _log_qwen_override_if_needed(workspace, step, action, prediction)
    t0 = time.perf_counter()
    _log_event(workspace, "stage_started", "Internal research stage started.", step=step, action_id=action.action_id, kind=action.kind)
    try:
        result = fn()
        raw = _stage_raw(action, result, round(time.perf_counter() - t0, 4))
        status = "success"
        _log_event(workspace, "stage_completed", "Internal research stage completed.", step=step, action_id=action.action_id, kind=action.kind, runtime_seconds=raw.get("runtime_seconds"))
    except Exception as exc:
        raw = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        status = "failed"
        result = None
        _log_event(workspace, "stage_failed", "Internal research stage failed.", step=step, action_id=action.action_id, kind=action.kind, error=raw["error"])
    observation = LabObservation(
        status=status,
        action_id=action.action_id,
        protocol_id=action.protocol_id,
        raw=raw,
        stdout_tail=json.dumps(raw, sort_keys=True)[:2000],
        runtime_seconds=raw.get("runtime_seconds"),
    )
    state.observations.append(observation)
    append_notebook_entry(
        workspace,
        LabNotebookEntry(
            step=step,
            state_before=f"open_lab_completed_actions={len(state.completed_actions)}; observations={len(state.observations)}",
            candidate_actions=[action],
            scientist_decision=None,
            qwen_predictions=[prediction.model_dump()],
            selected_action=action,
            why_world_model_mattered="Qwen predicted this generated-lab stage before execution.",
            actual_observation=observation,
            prediction_comparison=compare_prediction(prediction, observation),
            next_decision="continue",
        ),
    )
    state.completed_actions.append(action.kind)
    if status != "success":
        raise RuntimeError(raw["error"])
    return result


def _stage_raw(action: LabAction, result, runtime: float) -> dict:
    raw = {"status": "success", "stage": action.kind, "runtime_seconds": runtime}
    if isinstance(result, list):
        raw["count"] = len(result)
        raw["items"] = [item.model_dump() if hasattr(item, "model_dump") else str(item) for item in result[:10]]
    elif isinstance(result, tuple):
        raw["items"] = [item.model_dump() if hasattr(item, "model_dump") else str(item) for item in result]
    elif hasattr(result, "model_dump"):
        raw["result"] = result.model_dump()
    else:
        raw["result"] = str(result)
    return raw


def _log_prediction(workspace: Path, step: int, action: LabAction, prediction: LabPrediction) -> None:
    _log_event(
        workspace,
        "qwen_prediction_completed",
        "Qwen-AgentWorld prediction received.",
        step=step,
        action_id=action.action_id,
        kind=action.kind,
        recommendation=prediction.recommendation,
        runtime_risk=prediction.runtime_risk,
        expected_runtime_seconds=prediction.expected_runtime_seconds,
        compute_waste_risk=prediction.compute_waste_risk,
        value_of_information=prediction.value_of_information,
        suggested_modification=prediction.suggested_modification,
        expected_claim_delta=prediction.expected_claim_delta,
    )


def _log_qwen_override_if_needed(workspace: Path, step: int, action: LabAction, prediction: LabPrediction) -> None:
    if prediction.recommendation not in {"skip", "modify"}:
        return
    _log_event(
        workspace,
        "qwen_recommendation_overridden",
        "Controller executed action despite Qwen recommending skip/modify.",
        step=step,
        action_id=action.action_id,
        kind=action.kind,
        recommendation=prediction.recommendation,
        suggested_modification=prediction.suggested_modification,
        rationale=prediction.rationale,
    )


def _execute_action(action: LabAction, timeout: int | None = None, workspace: Path | None = None) -> LabObservation:
    if action.kind == "stop_and_report":
        return LabObservation(status="success", action_id=action.action_id, raw={"status": "success", "message": "stop_and_report"})
    start = time.perf_counter()
    configured_timeout = os.getenv("LUCKYLOOP_EXECUTION_TIMEOUT_SECONDS")
    if timeout is None and configured_timeout:
        timeout = int(configured_timeout)
    proc = subprocess.Popen(
        action.command.replace("python ", f"{sys.executable} ", 1),
        shell=True,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = _communicate_with_live_logs(proc, action, workspace, timeout)
    runtime = round(time.perf_counter() - start, 4)
    raw = {"status": "failed", "returncode": proc.returncode}
    try:
        raw = json.loads(stdout[stdout.find("{") :])
    except Exception:
        raw["stdout_tail"] = _tail(stdout)
        raw["stderr_tail"] = _tail(stderr)
    raw = _normalize_generated_raw(raw)
    artifacts = raw.get("artifacts", [])
    if isinstance(artifacts, dict):
        artifacts = []
    return LabObservation(
        status=str(raw.get("status", "success" if proc.returncode == 0 else "failed")),
        action_id=action.action_id,
        protocol_id=action.protocol_id,
        raw=raw,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
        artifacts=[str(item) for item in artifacts],
        runtime_seconds=raw.get("runtime_seconds", runtime),
    )


def _communicate_with_live_logs(
    proc: subprocess.Popen[str],
    action: LabAction,
    workspace: Path | None,
    timeout: int | None,
) -> tuple[str, str]:
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stdout() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)

    def read_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)
            if workspace is not None and line.strip():
                _log_event(
                    workspace,
                    "action_progress",
                    "Generated experiment progress.",
                    action_id=action.action_id,
                    kind=action.kind,
                    line=line.strip()[:500],
                )

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        if workspace is not None:
            _log_event(workspace, "action_execution_timeout", "Lab action exceeded configured timeout.", action_id=action.action_id, kind=action.kind)
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        return "".join(stdout_lines), "".join(stderr_lines) + "\nTimeoutExpired"
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    return "".join(stdout_lines), "".join(stderr_lines)


def _normalize_generated_raw(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return raw
    if raw.get("status") in {"completed", "succeeded", "ok", "claim_blocked", "claim_supported", "claim_inconclusive"} and (
        raw.get("runs") or raw.get("summary")
    ):
        raw["status"] = "success"
    primary = str(raw.get("primary_metric") or "balanced_accuracy")
    if isinstance(raw.get("runs"), dict):
        rows = []
        for condition, metrics in raw["runs"].items():
            if not isinstance(metrics, dict):
                continue
            values = metrics.get(primary)
            if values is None and "score" in metrics:
                values = metrics.get("score")
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                row = {"condition": str(condition), "model": str(condition), "seed_index": index, primary: value}
                for metric_name, metric_values in metrics.items():
                    if isinstance(metric_values, list) and index < len(metric_values):
                        row[str(metric_name)] = metric_values[index]
                rows.append(row)
        if rows:
            raw["runs"] = rows
    normalized_rows = []
    changed = False
    for row in raw.get("runs") or []:
        if not isinstance(row, dict):
            normalized_rows.append(row)
            continue
        new_row = dict(row)
        nested_metrics = new_row.get("metrics")
        if isinstance(nested_metrics, dict):
            for metric_name, metric_value in nested_metrics.items():
                if metric_name not in new_row:
                    new_row[str(metric_name)] = metric_value
            changed = True
        if "condition" not in new_row and "model" in new_row:
            new_row["condition"] = str(new_row["model"])
            changed = True
        if primary not in new_row and "score" in new_row:
            new_row[primary] = new_row["score"]
            changed = True
        normalized_rows.append(new_row)
    if changed:
        raw["runs"] = normalized_rows
    _recompute_effect_from_runs(raw, primary)
    summary = raw.get("summary")
    if isinstance(summary, dict):
        normalized_summary = {}
        for condition, values in summary.items():
            if isinstance(values, dict) and "mean" in values:
                normalized_summary[condition] = {
                    f"mean_{primary}": values.get("mean"),
                    f"std_{primary}": values.get("std", 0.0),
                    "min": min(values.get("scores", [])) if values.get("scores") else values.get("mean"),
                    "max": max(values.get("scores", [])) if values.get("scores") else values.get("mean"),
                    "n": len(values.get("scores", [])) if values.get("scores") else 1,
                }
            else:
                normalized_summary[condition] = values
        raw["summary"] = normalized_summary
    raw["best_condition"] = _normalize_best_condition(raw.get("best_condition"))
    raw["effect_size"] = _numeric_from_value(raw.get("effect_size"), prefer_key=raw.get("best_condition"))
    raw["seed_noise"] = _numeric_from_value(raw.get("seed_noise"), prefer_key=raw.get("best_condition"))
    raw["effect_to_noise_ratio"] = _numeric_from_value(raw.get("effect_to_noise_ratio"), prefer_key=raw.get("best_condition"))
    return raw


def _recompute_effect_from_runs(raw: dict, primary: str) -> None:
    rows = raw.get("runs")
    if not isinstance(rows, list):
        return
    by_condition: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get(primary)
        condition = row.get("condition") or row.get("model")
        if condition is not None and isinstance(value, (int, float)):
            by_condition.setdefault(str(condition), []).append(float(value))
    if len(by_condition) < 2:
        return
    means = {condition: sum(values) / len(values) for condition, values in by_condition.items() if values}
    ordered = sorted(means, key=means.get, reverse=True)
    best, runner_up = ordered[0], ordered[1]
    best_values = by_condition[best]
    seed_noise = max(best_values) - min(best_values) if len(best_values) > 1 else None
    effect = means[best] - means[runner_up]
    ratio = effect / seed_noise if seed_noise and seed_noise > 0 else None
    raw["best_condition"] = best
    raw["effect_size"] = float(effect)
    raw["seed_noise"] = None if seed_noise is None else float(seed_noise)
    raw["effect_to_noise_ratio"] = None if ratio is None else float(ratio)


def _normalize_best_condition(value):
    if isinstance(value, dict):
        for key in ["model", "condition", "best_condition", "name"]:
            if value.get(key) is not None:
                return str(value[key])
    return value


def _numeric_from_value(value, prefer_key=None):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if prefer_key is not None and prefer_key in value and isinstance(value[prefer_key], (int, float)):
            return float(value[prefer_key])
        for key in ["value", "mean", "effect", "ratio"]:
            if isinstance(value.get(key), (int, float)):
                return float(value[key])
        numeric = [float(item) for item in value.values() if isinstance(item, (int, float))]
        return max(numeric) if numeric else None
    return value


def _make_dry_run_dataset(audit: DatasetAudit, workspace: Path, *, max_rows: int = 320) -> str:
    source = Path(audit.local_path)
    target = workspace / "datasets" / "dry_run_dataset.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline="") as src, target.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        for index, row in enumerate(reader):
            if index > max_rows:
                break
            writer.writerow(row)
    return str(target)


def _script_action(
    step: int,
    kind: str,
    script: Path,
    audit: DatasetAudit,
    workspace: Path,
    *,
    dry_run: bool,
    dataset_csv: str | None = None,
) -> LabAction:
    dataset_path = dataset_csv or audit.local_path
    command = (
        f"python {script.relative_to(ROOT)} --dataset-csv {dataset_path} "
        f"--target-column {audit.target_column} --out-dir {workspace} --step {step}"
    )
    if dry_run:
        command += " --dry-run"
    mode = "reduced dry-run" if dry_run else "full repeated-seed protocol"
    row_count = min(audit.n_rows, 320) if dry_run else audit.n_rows
    heavy_models = {"svc", "random_forest", "hist_gradient_boosting"}
    cost_class = "cheap" if dry_run else ("expensive" if audit.n_rows > 5000 else "moderate")
    protocol_risks = ["generated_code", "dataset_selection_bias", "single_split_overclaim"]
    if not dry_run and heavy_models:
        protocol_risks.append("heavy_model_runtime")
    return LabAction(
        action_id=f"action_{step:03d}_{kind}",
        kind=kind if kind in {"dry_run", "run_protocol", "run_replication", "run_ablation"} else "run_protocol",
        hypothesis_id="H1_generated",
        protocol_id="generated_ml_research_protocol",
        scientific_goal=(
            f"Execute generated experiment code in {mode} mode on approximately {row_count} rows "
            f"and {audit.n_features} features. Dry-run must validate IO/schema quickly; full run may train repeated seeds."
        ),
        command=command,
        expected_artifacts=[f"runs/experiment_{step:03d}.json"],
        primary_metric="balanced_accuracy",
        manipulated_variables=["execution_mode", "model_family"],
        controls=["dataset", "target_column", "split_strategy", "seeds"],
        protocol_risks=protocol_risks,
        claim_delta_target="reduces_uncertainty",
        estimated_cost_class=cost_class,
    )


def _stage_action(step: int, kind: str, goal: str) -> LabAction:
    return LabAction(
        action_id=f"action_{step:03d}_{kind}",
        kind=kind,
        scientific_goal=goal,
        command="python -c \"print('internal stage')\"",
        expected_artifacts=[],
        protocol_risks=["open_ended_research"],
        claim_delta_target="reduces_uncertainty" if kind != "stop_and_report" else "report_ready",
        estimated_cost_class="moderate",
    )


def _to_protocol_spec(protocol: GeneratedResearchProtocol, audit: DatasetAudit) -> ProtocolSpec:
    return ProtocolSpec(
        protocol_id=protocol.protocol_id,
        hypothesis_id="H1_generated",
        scientific_goal=protocol.hypothesis,
        dataset=protocol.dataset_id,
        conditions=protocol.candidate_models,
        controlled_variables=["dataset", "target_column", "split_strategy", "seeds", "preprocessing"],
        manipulated_variable="model_family",
        primary_metric=protocol.primary_metric,
        secondary_metrics=protocol.secondary_metrics,
        seeds=protocol.seeds,
        expected_artifacts=["runs/experiment_<step>.json", "analyses/analysis_<step>.json"],
        protocol_risks=["generated_protocol", "dataset_selection_bias", *audit.warnings],
        claim_enabled_if=protocol.claim_enabled_if,
        claim_blocked_if=protocol.claim_blocked_if,
    )


def _research_program(question, domain_context, method_context, selected, audit, protocol, agenda):
    return {
        "schema_version": "2.0",
        "mode": "pure_auto_research_ml_first",
        "question": question,
        "domain_literature_sources": len(domain_context.papers),
        "method_literature_sources": len(method_context.papers),
        "research_agenda": agenda.model_dump() if hasattr(agenda, "model_dump") else agenda,
        "dataset_selection": {"candidate": selected.model_dump(), "audit": audit.model_dump()},
        "generated_protocol": protocol.model_dump(),
        "pipeline": [
            "domain_literature_review",
            "method_literature_review",
            "research_agenda_generation",
            "hypothesis_prioritization",
            "dataset_discovery_hf_openml",
            "dataset_audit",
            "free_form_protocol_generation",
            "free_form_code_generation",
            "static_code_validation",
            "dry_run",
            "real_ml_execution",
            "analysis",
            "next_research_decision",
            "claim_verification",
            "final_report",
        ],
    }


def _write_open_question(workspace: Path, question: str) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "question.md").write_text(
        "\n".join(
            [
                "# Research Question",
                "",
                question,
                "",
                "## Mode",
                "",
                "- Study id: `open_generated_ml_research`",
                "- Dataset discovery uses external Hugging Face and OpenML datasets.",
                "- DeepSeek generates the empirical protocol and experiment code.",
                "- Qwen-AgentWorld predicts each lab action before execution.",
                "- Python executes the real experiment; the verifier controls claims.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _clean_workspace(workspace: Path) -> None:
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
    for rel in ["predictions", "runs", "analyses", "protocols", "protocol", "generated", "datasets", "agenda", "decisions", "events.jsonl"]:
        path = workspace / rel
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _slug(question: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:72].strip("-")
    return f"open-{slug or 'research'}"


def _log_event(workspace: Path, event: str, message: str, **fields) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "message": message,
        **fields,
    }
    line = json.dumps(payload, sort_keys=True)
    with (workspace / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    detail_keys = {
        "step",
        "kind",
        "status",
        "runtime_seconds",
        "recommendation",
        "runtime_risk",
        "expected_runtime_seconds",
        "compute_waste_risk",
        "value_of_information",
    }
    details = " ".join(f"{key}={value}" for key, value in fields.items() if key in detail_keys and value not in {None, ""})
    print(f"[luckyloop] {event}: {message}" + (f" {details}" if details else ""), file=sys.stderr, flush=True)
