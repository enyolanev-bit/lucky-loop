from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from .schemas import (
    DatasetAudit,
    GeneratedResearchProtocol,
    HypothesisCandidate,
    LabAction,
    LabScientistDecision,
    LabStudyState,
    NextDecision,
    ResearchAgenda,
    VerificationPlan,
)


PROMPT_VERSION = "lab_scientist_planner_v1"
PROTOCOL_INFERENCE_PROMPT_VERSION = "lab_protocol_family_inference_v1"
OPEN_PROTOCOL_PROMPT_VERSION = "open_ml_protocol_generation_v1"
EXPERIMENT_CODE_PROMPT_VERSION = "open_ml_code_generation_v1"

SYSTEM = """You are the scientist planner for Lucky Loop, a computational ML research validity lab.
You are not the world model and not the verifier.

Your job is to choose the next lab action from the provided safe catalog.
You must never invent commands, datasets, files, results, or claims.
Choose only an action_id that appears in candidate_actions.

Return strict JSON only with exactly these keys:
- research_question: string
- working_hypothesis: string
- candidate_action_ids: array of action_id strings from candidate_actions
- preferred_action_id: one action_id from candidate_actions
- rationale: string
- expected_evidence_needed: string
- claim_risk: string
- stop_or_continue: "continue" or "stop_and_report"
"""

PROTOCOL_SYSTEM = """You are the intake scientist for Lucky Loop, a computational ML research validity lab.
You map a natural-language research question and the literature review context to one supported protocol family.

You must choose exactly one study_id from supported_studies.
You must not invent new study IDs.
Return strict JSON only with exactly these keys:
- study_id: one value from supported_studies
- rationale: string
- evidence_from_literature: array of short strings
- rejected_studies: object mapping other study_id values to short rejection reasons
"""

OPEN_PROTOCOL_SYSTEM = """You are the lead ML scientist for Lucky Loop.
Create a concrete empirical ML research protocol from the question, literature context, and audited dataset.

Return strict JSON only. Do not include markdown.
Required keys:
- hypothesis: string
- baseline_models: array of sklearn-compatible model names or short aliases
- candidate_models: array of sklearn-compatible model names or short aliases
- primary_metric: balanced_accuracy
- secondary_metrics: array
- seeds: array of integers
- split_strategy: stratified_train_test_split
- ablations: array of strings
- claim_enabled_if: string
- claim_blocked_if: string
- risk_controls: array of strings
"""

AGENDA_SYSTEM = """You are the lead research scientist for Lucky Loop.
Build a domain-specific ML research agenda from the user's question and separated literature contexts.

Use domain literature for scientific hypotheses. Use method literature only for risk controls.
Preserve the direction of the user's question. If the user asks whether nonlinear models outperform logistic regression, do not invert it into a logistic-regression-superiority claim unless the agenda explicitly labels that as a counter-hypothesis and selects a hypothesis aligned with the user's requested direction.
Prefer hypotheses that can become supported or weakly supported on one audited public dataset before requiring multi-dataset proof.
Return strict JSON only with exactly:
- domain_summary: string
- method_summary: string
- domain_gaps: array of objects with gap_id, claim, source_ids, why_it_matters, testable_question, dataset_requirements, suggested_metrics
- hypotheses: array of objects with hypothesis_id, claim_candidate, motivation, literature_gap_ids, dataset_requirements, expected_signal, falsification_condition, minimum_evidence_needed, scientific_value, compute_risk
- selected_hypothesis_id: string
- selection_rationale: string
"""

NEXT_DECISION_SYSTEM = """You are the post-experiment scientist for Lucky Loop.
Decide the next research action from the current empirical state.

Do not overclaim. If evidence is weak, choose a follow-up unless the budget is exhausted.
Return strict JSON only with exactly:
- decision: one of replicate, ablate, inspect_failure, revise_hypothesis, revise_protocol, search_better_dataset, verify_claim, stop_and_report
- rationale: string
- next_action_goal: string
- expected_evidence_gain: string
- stop_reason: string
"""

CODE_SYSTEM = """You are the experiment-coding scientist for Lucky Loop.
Write one complete Python script for a supervised classification experiment.

Hard requirements:
- Use only these import roots: argparse, json, math, time, warnings, pathlib, numpy, pandas, sklearn, scipy.
- Do not import sys, os, subprocess, socket, requests, urllib, httpx, openai, datasets, huggingface_hub, or shutil.
- Never call open(), eval(), exec(), compile(), input(), or __import__().
- To write the output JSON, use pathlib.Path(...).write_text(...).
- No network, no subprocess, no shell, no secrets, no file writes outside the provided output directory.
- Read --dataset-csv, --target-column, --out-dir, --step, and optional --dry-run.
- Train repeated seeds with baselines and candidates from the protocol.
- If --dry-run is set, it must be a fast schema/IO smoke test: use at most 1 seed, at most 2 models, and at most 300 rows from the provided CSV. It must still write valid output JSON with at least one run row.
- During model/seed execution, write compact progress events to stderr with print(..., file=__import__('sys').stderr, flush=True) is forbidden because __import__ is forbidden; instead import is not allowed for sys, so use warnings.warn("progress: ...") after each seed/model fit.
- Print strict JSON to stdout.
- Write the same JSON to out-dir/runs/experiment_<step>.json.
- Output JSON must include: status, protocol_id, dataset, dataset_source, lab_action, primary_metric, runs, summary, effect_size, seed_noise, effect_to_noise_ratio, best_condition, protocol_warnings, artifacts, runtime_seconds.

Return strict JSON only with exactly:
- code: string
- rationale: string
"""


def agent_configured() -> bool:
    return bool(
        os.getenv("LUCKYLOOP_AGENT_BASE_URL")
        and os.getenv("LUCKYLOOP_AGENT_MODEL")
        and os.getenv("LUCKYLOOP_AGENT_API_KEY")
    )


def _json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                data, _ = decoder.raw_decode(text[match.start() :])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("scientist planner response did not contain JSON")
        return json.loads(match.group(0))


def _candidate_ids(actions: list[LabAction]) -> set[str]:
    return {action.action_id for action in actions}


def validate_decision(decision: LabScientistDecision, actions: list[LabAction]) -> LabScientistDecision:
    ids = _candidate_ids(actions)
    if decision.preferred_action_id not in ids:
        raise ValueError(f"scientist planner preferred unknown action_id: {decision.preferred_action_id}")
    unknown = [action_id for action_id in decision.candidate_action_ids if action_id not in ids]
    if unknown:
        raise ValueError(f"scientist planner referenced unknown action_ids: {unknown}")
    return decision


def infer_protocol_family(
    question: str,
    literature_context: dict,
    supported_studies: list[str],
    *,
    planner: str = "llm",
    require_agent: bool = True,
) -> dict:
    if planner != "llm":
        raise ValueError("protocol-family inference is only available through the LLM scientist planner")
    if not agent_configured():
        if require_agent:
            raise RuntimeError(
                "Scientist planner is required for study inference but LUCKYLOOP_AGENT_BASE_URL, "
                "LUCKYLOOP_AGENT_MODEL, and LUCKYLOOP_AGENT_API_KEY are not configured."
            )
        return {
            "study_id": "",
            "source": "not_configured",
            "rationale": "No scientist planner configured; caller may use a development fallback.",
            "evidence_from_literature": [],
            "rejected_studies": {},
            "prompt_version": PROTOCOL_INFERENCE_PROMPT_VERSION,
        }

    base_url = os.environ["LUCKYLOOP_AGENT_BASE_URL"]
    model = os.environ["LUCKYLOOP_AGENT_MODEL"]
    api_key = os.environ["LUCKYLOOP_AGENT_API_KEY"]
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=0)
    payload = {
        "research_question": question,
        "supported_studies": supported_studies,
        "literature_context": literature_context,
        "contract": {
            "choose_one_supported_study_id": True,
            "do_not_invent_protocols": True,
            "map_from_literature_gaps_to_safe_protocol_catalog": True,
        },
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROTOCOL_SYSTEM},
            {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
        ],
        temperature=0.1,
        max_tokens=700,
    )
    data = _json_from_text(response.choices[0].message.content or "{}")
    study_id = str(data.get("study_id") or "")
    if study_id not in supported_studies:
        raise ValueError(f"scientist planner inferred unsupported study_id: {study_id}")
    return {
        "study_id": study_id,
        "source": "llm",
        "rationale": str(data.get("rationale") or ""),
        "evidence_from_literature": [str(item) for item in data.get("evidence_from_literature", [])],
        "rejected_studies": dict(data.get("rejected_studies") or {}),
        "prompt_version": PROTOCOL_INFERENCE_PROMPT_VERSION,
        "model": model,
    }


def _stop_or_continue(value) -> str:
    text = str(value or "continue").lower()
    return "stop_and_report" if "stop" in text else "continue"


def _client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["LUCKYLOOP_AGENT_BASE_URL"],
        api_key=os.environ["LUCKYLOOP_AGENT_API_KEY"],
        timeout=90,
        max_retries=0,
    )


def _require_agent(require_agent: bool) -> bool:
    if agent_configured():
        return True
    if require_agent:
        raise RuntimeError(
            "Scientist planner is required but LUCKYLOOP_AGENT_BASE_URL, "
            "LUCKYLOOP_AGENT_MODEL, and LUCKYLOOP_AGENT_API_KEY are not configured."
        )
    raise RuntimeError("Open auto-research requires the scientist agent; no non-agent fallback is allowed.")


def generate_open_protocol(
    question: str,
    literature_context: dict,
    dataset_audit: DatasetAudit,
    hypothesis: HypothesisCandidate | None = None,
    *,
    require_agent: bool = True,
) -> GeneratedResearchProtocol:
    _require_agent(require_agent)
    model = os.environ["LUCKYLOOP_AGENT_MODEL"]
    payload = {
        "question": question,
        "literature_context": literature_context,
        "dataset_audit": dataset_audit.model_dump(),
        "selected_hypothesis": hypothesis.model_dump() if hypothesis else None,
        "contract": {
            "use_audited_dataset_only": True,
            "classification_only": True,
            "must_include_repeated_seeds": True,
            "must_include_baseline": True,
            "must_include_claim_blocking_rule": True,
            "models_must_be_sklearn_compatible": True,
            "preserve_user_question_direction": True,
            "avoid_reversing_selected_hypothesis": True,
        },
    }
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": OPEN_PROTOCOL_SYSTEM},
            {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }
    try:
        response = _client().chat.completions.create(**request)
    except Exception:
        request.pop("response_format", None)
        response = _client().chat.completions.create(**request)
    data = _json_from_text(response.choices[0].message.content or "{}")
    if hypothesis:
        data["hypothesis"] = _align_claim_direction(question, str(data.get("hypothesis") or hypothesis.claim_candidate))
    return _protocol_from_data(question, dataset_audit, data)


def generate_research_agenda(
    question: str,
    domain_context: dict,
    method_context: dict,
    *,
    require_agent: bool = True,
) -> ResearchAgenda:
    _require_agent(require_agent)
    model = os.environ["LUCKYLOOP_AGENT_MODEL"]
    payload = {
        "question": question,
        "domain_context": domain_context,
        "method_context": method_context,
        "contract": {
            "domain_literature_drives_hypotheses": True,
            "method_literature_only_for_safeguards": True,
            "produce_multiple_hypotheses": True,
            "select_one_initial_hypothesis": True,
            "preserve_user_question_direction": True,
            "selected_hypothesis_must_answer_user_question": True,
            "avoid_multi_dataset_minimum_when_one_dataset_can_answer": True,
        },
    }
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": AGENDA_SYSTEM},
            {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
        ],
        "temperature": 0.2,
        "max_tokens": 2200,
        "response_format": {"type": "json_object"},
    }
    try:
        response = _client().chat.completions.create(**request)
    except Exception:
        request.pop("response_format", None)
        response = _client().chat.completions.create(**request)
    data = _json_from_text(response.choices[0].message.content or "{}")
    return _agenda_from_data(question, data)


def decide_next_research_step(
    question: str,
    agenda: ResearchAgenda,
    protocol: GeneratedResearchProtocol,
    latest_analysis: dict,
    latest_claims: list[dict],
    budget_remaining: int,
    *,
    require_agent: bool = True,
) -> NextDecision:
    _require_agent(require_agent)
    model = os.environ["LUCKYLOOP_AGENT_MODEL"]
    payload = {
        "question": question,
        "agenda": agenda.model_dump(),
        "protocol": protocol.model_dump(),
        "latest_analysis": latest_analysis,
        "latest_claims": latest_claims,
        "budget_remaining": budget_remaining,
        "contract": {
            "do_not_stop_after_unsupported_claim_unless_budget_exhausted": True,
            "prefer_targeted_followup_for_weak_or_invalid_evidence": True,
            "if_budget_remains_and_claim_blocked_choose_revision_or_followup": True,
            "python_execution_is_ground_truth": True,
        },
    }
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": NEXT_DECISION_SYSTEM},
            {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
        ],
        "temperature": 0.2,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
    }
    try:
        response = _client().chat.completions.create(**request)
    except Exception:
        request.pop("response_format", None)
        response = _client().chat.completions.create(**request)
    data = _json_from_text(response.choices[0].message.content or "{}")
    decision = str(data.get("decision") or "stop_and_report")
    allowed = {
        "replicate",
        "ablate",
        "inspect_failure",
        "revise_hypothesis",
        "revise_protocol",
        "search_better_dataset",
        "verify_claim",
        "stop_and_report",
    }
    if decision not in allowed:
        decision = "stop_and_report"
    has_blocked_claim = any(str(claim.get("verdict") or claim.get("status") or "").lower() == "blocked" for claim in latest_claims)
    if budget_remaining > 0 and has_blocked_claim and decision == "stop_and_report":
        failure_categories = {str(claim.get("failure_category") or "") for claim in latest_claims}
        decision = "search_better_dataset" if "claim_direction_failed" in failure_categories else "revise_hypothesis"
        data["next_action_goal"] = data.get("next_action_goal") or "Use the blocked claim to generate a narrower follow-up that can become supported."
        data["expected_evidence_gain"] = data.get("expected_evidence_gain") or "A targeted follow-up should distinguish unsupported overclaim from a weaker supportable claim."
        data["stop_reason"] = ""
    return NextDecision(
        decision=decision,
        rationale=str(data.get("rationale") or ""),
        next_action_goal=str(data.get("next_action_goal") or ""),
        expected_evidence_gain=str(data.get("expected_evidence_gain") or ""),
        stop_reason=str(data.get("stop_reason") or ""),
    )


def generate_experiment_code(
    protocol: GeneratedResearchProtocol,
    dataset_audit: DatasetAudit,
    repair_context: dict | None = None,
    *,
    require_agent: bool = True,
) -> dict:
    _require_agent(require_agent)
    model = os.environ["LUCKYLOOP_AGENT_MODEL"]
    payload = {
        "protocol": protocol.model_dump(),
        "dataset_audit": dataset_audit.model_dump(),
        "repair_context": repair_context or {},
        "contract": {
            "return_code_only_inside_json_string": True,
            "stdout_strict_json": True,
            "use_path_write_text_not_open": True,
            "no_network": True,
            "no_shell": True,
            "no_sys_import": True,
            "no_secret_access": True,
        },
    }
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": CODE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
        ],
        "temperature": 0.1,
        "max_tokens": 10000,
        "response_format": {"type": "json_object"},
    }
    try:
        response = _client().chat.completions.create(**request)
    except Exception:
        request.pop("response_format", None)
        response = _client().chat.completions.create(**request)
    data = _json_from_text(response.choices[0].message.content or "{}")
    code = str(data.get("code") or "")
    if not code.strip():
        raise ValueError("code generation response had empty code")
    return {"code": code, "rationale": str(data.get("rationale") or ""), "model": model}


def _protocol_from_data(question: str, audit: DatasetAudit, data: dict) -> GeneratedResearchProtocol:
    baselines = [_safe_model_name(item) for item in _strings(data.get("baseline_models")) if _safe_model_name(item)] or ["logistic_regression"]
    candidates = [_safe_model_name(item) for item in _strings(data.get("candidate_models")) if _safe_model_name(item)] or ["logistic_regression", "random_forest", "svc"]
    seeds = []
    for item in data.get("seeds", [0, 1, 2, 3]):
        try:
            seeds.append(int(item))
        except Exception:
            pass
    return GeneratedResearchProtocol(
        question=question,
        hypothesis=str(data.get("hypothesis") or "A model comparison claim is only valid if it survives repeated-seed verification."),
        dataset_id=audit.dataset_id,
        dataset_source=audit.source,
        target_column=audit.target_column or "target",
        feature_columns=audit.feature_columns,
        baseline_models=baselines,
        candidate_models=sorted(set(candidates + baselines)),
        primary_metric="balanced_accuracy",
        secondary_metrics=_strings(data.get("secondary_metrics")) or ["accuracy", "f1_weighted"],
        seeds=seeds[:8] or [0, 1, 2, 3],
        split_strategy="stratified_train_test_split",
        ablations=_strings(data.get("ablations")),
        claim_enabled_if=str(data.get("claim_enabled_if") or "The best candidate beats the strongest baseline by more than seed noise."),
        claim_blocked_if=str(data.get("claim_blocked_if") or "The effect is less than or equal to seed noise."),
        risk_controls=_strings(data.get("risk_controls")),
    )


def _safe_model_name(value) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    blocked = {"", "xgboost", "lightgbm", "catboost", "tensorflow", "torch", "pytorch"}
    return "" if text in blocked else text


def _strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _align_claim_direction(question: str, claim: str) -> str:
    q = question.lower()
    c = claim.lower()
    asks_nonlinear_over_logistic = "nonlinear" in q and "outperform" in q and "logistic" in q
    logistic_superiority = "logistic" in c and any(word in c for word in ["higher", "better", "outperform", "exceed"])
    nonlinear_mentioned = "nonlinear" in c or any(word in c for word in ["random forest", "svc", "svm", "mlp", "tree", "ensemble"])
    if asks_nonlinear_over_logistic and logistic_superiority and nonlinear_mentioned:
        return "A nonlinear model can robustly outperform logistic regression across repeated seeds on a public sensor classification dataset."
    return claim


def _agenda_from_data(question: str, data: dict) -> ResearchAgenda:
    gaps = []
    for index, item in enumerate(data.get("domain_gaps", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        gaps.append(
            {
                "gap_id": str(item.get("gap_id") or f"domain_gap_{index}"),
                "claim": str(item.get("claim") or ""),
                "source_ids": _strings(item.get("source_ids")),
                "why_it_matters": str(item.get("why_it_matters") or ""),
                "testable_question": str(item.get("testable_question") or ""),
                "dataset_requirements": _strings(item.get("dataset_requirements")),
                "suggested_metrics": _strings(item.get("suggested_metrics")),
            }
        )
    hypotheses = []
    for index, item in enumerate(data.get("hypotheses", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        scientific_value = _bounded_float(item.get("scientific_value"), 0.5)
        compute_risk = _bounded_float(item.get("compute_risk"), 0.5)
        hypotheses.append(
            HypothesisCandidate(
                hypothesis_id=str(item.get("hypothesis_id") or f"H{index}"),
                claim_candidate=_align_claim_direction(question, str(item.get("claim_candidate") or question)),
                motivation=str(item.get("motivation") or ""),
                literature_gap_ids=_strings(item.get("literature_gap_ids")),
                dataset_requirements=_strings(item.get("dataset_requirements")),
                expected_signal=str(item.get("expected_signal") or ""),
                falsification_condition=str(item.get("falsification_condition") or "The observed effect does not exceed seed noise."),
                minimum_evidence_needed=str(item.get("minimum_evidence_needed") or "Repeated-seed evidence exceeds noise."),
                scientific_value=scientific_value,
                compute_risk=compute_risk,
                priority_score=scientific_value / max(compute_risk, 0.05),
            )
        )
    if not hypotheses:
        hypotheses.append(
            HypothesisCandidate(
                hypothesis_id="H1",
                claim_candidate=question,
                motivation="Generated as the minimum testable form of the user question.",
                falsification_condition="The experiment does not support the proposed direction beyond seed noise.",
                minimum_evidence_needed="Repeated-seed evidence exceeds seed noise.",
                priority_score=1.0,
            )
        )
    selected = str(data.get("selected_hypothesis_id") or "")
    if selected not in {item.hypothesis_id for item in hypotheses}:
        selected = max(hypotheses, key=lambda item: item.priority_score).hypothesis_id
    return ResearchAgenda(
        question=question,
        domain_summary=str(data.get("domain_summary") or ""),
        method_summary=str(data.get("method_summary") or ""),
        domain_gaps=gaps,
        hypotheses=hypotheses,
        selected_hypothesis_id=selected,
        selection_rationale=str(data.get("selection_rationale") or "Selected by scientific value divided by compute risk."),
    )


def _bounded_float(value, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, parsed))


def _template_protocol(question: str, audit: DatasetAudit) -> GeneratedResearchProtocol:
    return GeneratedResearchProtocol(
        question=question,
        hypothesis="A more complex model is only claimably better than a linear baseline if the repeated-seed effect exceeds seed noise.",
        dataset_id=audit.dataset_id,
        dataset_source=audit.source,
        target_column=audit.target_column or "target",
        feature_columns=audit.feature_columns,
        baseline_models=["logistic_regression"],
        candidate_models=["logistic_regression", "random_forest", "svc"],
        risk_controls=["stratified split", "repeated seeds", "baseline comparison"],
    )


def template_experiment_code() -> str:
    return r'''
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

PROTOCOL = __PROTOCOL_JSON__

def model_for(name, seed):
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"logistic_regression", "logisticregression", "linear_baseline"}:
        return LogisticRegression(max_iter=1000, random_state=seed)
    if normalized in {"random_forest", "randomforest", "randomforestclassifier"}:
        return RandomForestClassifier(n_estimators=160, random_state=seed, n_jobs=-1)
    if normalized in {"svc", "svm", "support_vector_classifier"}:
        return SVC(C=2.0, kernel="rbf", gamma="scale", random_state=seed)
    if normalized in {"hist_gradient_boosting", "histgradientboostingclassifier"}:
        return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, random_state=seed)
    if normalized in {"gradient_boosting", "gradientboostingclassifier"}:
        return GradientBoostingClassifier(n_estimators=120, random_state=seed)
    if normalized in {"mlp", "mlpclassifier", "neural_network"}:
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=seed)
    raise ValueError(f"unknown model: {name}")

def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

def summarize(rows, primary):
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(float(row[primary]))
    summary = {}
    for key, values in by_condition.items():
        arr = np.array(values, dtype=float)
        summary[key] = {
            f"mean_{primary}": float(arr.mean()),
            f"std_{primary}": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(len(arr)),
        }
    ordered = sorted(summary, key=lambda key: summary[key][f"mean_{primary}"], reverse=True)
    if len(ordered) < 2:
        return summary, None, None, None, ordered[0] if ordered else None
    best, runner_up = ordered[0], ordered[1]
    effect = summary[best][f"mean_{primary}"] - summary[runner_up][f"mean_{primary}"]
    best_values = by_condition[best]
    seed_noise = max(best_values) - min(best_values) if len(best_values) > 1 else None
    ratio = effect / seed_noise if seed_noise and seed_noise > 0 else None
    return summary, float(effect), None if seed_noise is None else float(seed_noise), None if ratio is None else float(ratio), best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    frame = pd.read_csv(args.dataset_csv)
    if args.dry_run and len(frame) > 120:
        frame = frame.sample(120, random_state=123)
    target = args.target_column
    y = frame[target].astype(str)
    X = frame[[col for col in frame.columns if col != target]]
    numeric = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical = [col for col in X.columns if col not in numeric]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
    seeds = PROTOCOL.get("seeds", [0, 1, 2, 3])
    if args.dry_run:
        seeds = seeds[:1]
    rows = []
    warnings = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
        for model_name in PROTOCOL.get("candidate_models", ["logistic_regression", "random_forest", "svc"]):
            clf = make_pipeline(pre, model_for(model_name, seed))
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            rows.append({"condition": model_name, "model": model_name, "seed": int(seed), **metrics(y_test, pred)})
    primary = PROTOCOL.get("primary_metric", "balanced_accuracy")
    summary, effect, seed_noise, ratio, best = summarize(rows, primary)
    if len(seeds) < 3:
        warnings.append("dry_run_or_low_seed_count: claim verification requires repeated seeds")
    payload = {
        "status": "success",
        "protocol_id": PROTOCOL.get("protocol_id", "generated_ml_research_protocol"),
        "dataset": PROTOCOL.get("dataset_id", "selected_dataset"),
        "dataset_source": PROTOCOL.get("dataset_source", "unknown"),
        "lab_action": "dry_run" if args.dry_run else "run_experiment",
        "primary_metric": primary,
        "runs": rows,
        "summary": summary,
        "effect_size": effect,
        "seed_noise": seed_noise,
        "effect_to_noise_ratio": ratio,
        "best_condition": best,
        "protocol_warnings": warnings,
        "artifacts": [],
        "runtime_seconds": round(time.perf_counter() - t0, 4),
    }
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"experiment_{args.step:03d}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["artifacts"].append(str(out_path))
    print(json.dumps(payload, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
'''


def deterministic_decision(state: LabStudyState, actions: list[LabAction]) -> LabScientistDecision:
    if not actions:
        raise ValueError("no candidate actions available")
    # Preference order over the kinds candidate_actions_for_state can emit.
    # `run_baseline` is placed after `run_protocol` so existing trajectories are
    # unchanged when both are present; it is included so a baseline-only step is
    # handled instead of crashing. Any other kind falls back to the first action.
    preferred = ["inspect_dataset", "run_protocol", "run_baseline", "run_replication", "run_ablation", "stop_and_report"]
    chosen = None
    chosen_kind = None
    for preferred_kind in preferred:
        for action in actions:
            if action.kind == preferred_kind:
                chosen, chosen_kind = action, preferred_kind
                break
        if chosen is not None:
            break
    if chosen is None:
        # Actions exist but none match the preferred kinds -> proceed gracefully.
        chosen, chosen_kind = actions[0], actions[0].kind
    return LabScientistDecision(
        source="deterministic",
        research_question=state.lab_question.question,
        working_hypothesis=state.hypotheses[0].claim_candidate if state.hypotheses else state.lab_question.question,
        candidate_action_ids=[candidate.action_id for candidate in actions],
        preferred_action_id=chosen.action_id,
        rationale=f"Deterministic planner chose the first {chosen_kind} action for plumbing.",
        expected_evidence_needed="Use the safe lab protocol output and verifier.",
        claim_risk="Deterministic mode is not a demo scientist; it is for local tests only.",
        stop_or_continue="stop_and_report" if chosen.kind == "stop_and_report" else "continue",
        prompt_version=PROMPT_VERSION,
    )


def decide_next_action(
    state: LabStudyState,
    actions: list[LabAction],
    literature_brief: dict,
    *,
    planner: str = "deterministic",
    require_agent: bool = False,
) -> LabScientistDecision:
    if planner == "deterministic":
        return deterministic_decision(state, actions)
    if planner != "llm":
        raise ValueError(f"unsupported lab planner: {planner}")
    if not agent_configured():
        if require_agent:
            raise RuntimeError(
                "Scientist planner is required but LUCKYLOOP_AGENT_BASE_URL, "
                "LUCKYLOOP_AGENT_MODEL, and LUCKYLOOP_AGENT_API_KEY are not configured."
            )
        return deterministic_decision(state, actions)

    base_url = os.environ["LUCKYLOOP_AGENT_BASE_URL"]
    model = os.environ["LUCKYLOOP_AGENT_MODEL"]
    api_key = os.environ["LUCKYLOOP_AGENT_API_KEY"]
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=0)
    payload = {
        "lab_state": state.model_dump(),
        "literature_brief": literature_brief,
        "candidate_actions": [action.model_dump() for action in actions],
        "contract": {
            "choose_action_id_only": True,
            "never_generate_commands": True,
            "qwen_is_world_model_not_planner": True,
            "python_execution_is_ground_truth": True,
            "verifier_controls_final_claims": True,
        },
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    data = _json_from_text(response.choices[0].message.content or "{}")
    decision = LabScientistDecision(
        source="llm",
        research_question=str(data.get("research_question") or state.lab_question.question),
        working_hypothesis=str(data.get("working_hypothesis") or (state.hypotheses[0].claim_candidate if state.hypotheses else "")),
        candidate_action_ids=[str(item) for item in data.get("candidate_action_ids", [])],
        preferred_action_id=str(data.get("preferred_action_id") or ""),
        rationale=str(data.get("rationale") or ""),
        expected_evidence_needed=str(data.get("expected_evidence_needed") or ""),
        claim_risk=str(data.get("claim_risk") or ""),
        stop_or_continue=_stop_or_continue(data.get("stop_or_continue")),
        prompt_version=PROMPT_VERSION,
        model=model,
    )
    return validate_decision(decision, actions)
