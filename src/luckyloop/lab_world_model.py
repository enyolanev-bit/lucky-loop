from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from .schemas import LabAction, LabPrediction, LabStudyState


LAB_WORLD_MODEL_PROMPT_VERSION = "lab_computer_world_model_v2"
LAB_WORLD_MODEL_SCHEMA_VERSION = "lab_prediction_schema_v2"


SYSTEM = """You are Qwen-AgentWorld acting as a computer-lab world model for ML research.
Your job is to predict the next computer-lab observation after a candidate lab action.

You are not the scientific oracle. You do not know the real result before execution.
Predict the likely terminal/output pattern, files/artifacts, runtime or failure risk, protocol risk, and whether the action can change what the final report may honestly claim.

Return strict JSON only with exactly these keys:
- predicted_terminal_observation: string
- expected_result_pattern: string
- predicted_artifacts: array of strings
- failure_modes: array of strings
- protocol_risks: array of strings
- runtime_risk: string
- expected_runtime_seconds: number or null
- expected_runtime_range_seconds: array with two numbers, or empty array
- compute_waste_risk: number from 0 to 1
- value_of_information: number from 0 to 1
- suggested_modification: string
- decision_threshold: string
- expected_claim_delta: one of "none", "adds_observation", "reduces_uncertainty", "enables_claim", "blocks_or_rewrites_claim", "report_ready"
- recommendation: one of "run", "skip", "modify", "verify", "stop_and_report"
- why_not_score_chasing: string
- rationale: string

Be specific. For generated Python/sklearn actions, estimate whether the action is a schema check, reduced dry-run, full fit, repeated-seed fit, or report-only step. If the command contains --dry-run but the action metadata suggests large data or repeated heavy models, recommend "modify" and explain the reduction needed. Mention likely expensive estimators such as SVC, random_forest, and hist_gradient_boosting when relevant.

Do not use markdown. Do not claim actual execution. Predict only."""


def simulator_configured() -> bool:
    return bool(os.getenv("LUCKYWORLD_SIMULATOR_BASE_URL") and os.getenv("LUCKYWORLD_SIMULATOR_MODEL"))


def _json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("Qwen lab prediction did not contain JSON")
        return json.loads(match.group(0))


def _list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _one_of(value, allowed: set[str], default: str) -> str:
    text = str(value or default).lower()
    return text if text in allowed else default


def _float_or_none(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _float_range(value) -> list[float]:
    if not isinstance(value, list):
        return []
    values = [_float_or_none(item) for item in value[:2]]
    return [item for item in values if item is not None]


def _bounded(value, default: float) -> float:
    parsed = _float_or_none(value)
    if parsed is None:
        return default
    return max(0.0, min(1.0, parsed))


def normalize_prediction(data: dict, source: str) -> LabPrediction:
    return LabPrediction(
        source=source,
        predicted_terminal_observation=str(data.get("predicted_terminal_observation") or data.get("observation") or ""),
        expected_result_pattern=str(data.get("expected_result_pattern") or ""),
        predicted_artifacts=_list(data.get("predicted_artifacts")),
        failure_modes=_list(data.get("failure_modes")),
        protocol_risks=_list(data.get("protocol_risks")),
        runtime_risk=str(data.get("runtime_risk") or ""),
        expected_runtime_seconds=_float_or_none(data.get("expected_runtime_seconds")),
        expected_runtime_range_seconds=_float_range(data.get("expected_runtime_range_seconds")),
        compute_waste_risk=_bounded(data.get("compute_waste_risk"), 0.0),
        value_of_information=_bounded(data.get("value_of_information"), 0.5),
        suggested_modification=str(data.get("suggested_modification") or ""),
        decision_threshold=str(data.get("decision_threshold") or ""),
        expected_claim_delta=_one_of(
            data.get("expected_claim_delta"),
            {"none", "adds_observation", "reduces_uncertainty", "enables_claim", "blocks_or_rewrites_claim", "report_ready"},
            "adds_observation",
        ),
        recommendation=_one_of(
            data.get("recommendation"),
            {"run", "skip", "modify", "verify", "stop_and_report"},
            "run",
        ),
        why_not_score_chasing=str(data.get("why_not_score_chasing") or ""),
        rationale=str(data.get("rationale") or ""),
        prompt_version=LAB_WORLD_MODEL_PROMPT_VERSION,
        world_model_schema_version=LAB_WORLD_MODEL_SCHEMA_VERSION,
    )


def plumbing_prediction(action: LabAction) -> LabPrediction:
    return LabPrediction(
        source="plumbing_not_called",
        predicted_terminal_observation="Qwen-AgentWorld was not called; this placeholder is only for non-demo plumbing tests.",
        expected_result_pattern="unknown until real execution",
        predicted_artifacts=action.expected_artifacts,
        failure_modes=[],
        protocol_risks=action.protocol_risks,
        runtime_risk="unknown",
        expected_runtime_seconds=None,
        expected_runtime_range_seconds=[],
        compute_waste_risk=0.0,
        value_of_information=0.5,
        suggested_modification="",
        decision_threshold="",
        expected_claim_delta=action.claim_delta_target,
        recommendation="run" if action.kind != "stop_and_report" else "stop_and_report",
        why_not_score_chasing="Plumbing mode does not provide world-model evidence.",
        rationale="No world-model prediction was requested.",
        prompt_version=LAB_WORLD_MODEL_PROMPT_VERSION,
        world_model_schema_version=LAB_WORLD_MODEL_SCHEMA_VERSION,
    )


def _compact_state(state: LabStudyState) -> dict:
    payload = state.model_dump()
    compact_observations = []
    for observation in state.observations[-4:]:
        raw = observation.raw or {}
        compact_observations.append(
            {
                "status": observation.status,
                "action_id": observation.action_id,
                "protocol_id": observation.protocol_id,
                "runtime_seconds": observation.runtime_seconds,
                "raw_summary": {
                    "status": raw.get("status"),
                    "dataset": raw.get("dataset"),
                    "dataset_source": raw.get("dataset_source"),
                    "primary_metric": raw.get("primary_metric"),
                    "best_condition": raw.get("best_condition"),
                    "effect_size": raw.get("effect_size"),
                    "seed_noise": raw.get("seed_noise"),
                    "effect_to_noise_ratio": raw.get("effect_to_noise_ratio"),
                    "protocol_warnings": raw.get("protocol_warnings", [])[:5] if isinstance(raw.get("protocol_warnings"), list) else [],
                    "run_count": len(raw.get("runs") or []) if isinstance(raw.get("runs"), list) else 0,
                },
                "stdout_tail": observation.stdout_tail[-500:],
                "stderr_tail": observation.stderr_tail[-500:],
            }
        )
    payload["observations"] = compact_observations
    payload["analyses"] = [analysis.model_dump() for analysis in state.analyses[-3:]]
    payload["claims"] = [claim.model_dump() for claim in state.claims[-5:]]
    return payload


def predict_lab_action(action: LabAction, state: LabStudyState, require_qwen: bool = False) -> LabPrediction:
    if not simulator_configured():
        if require_qwen:
            raise RuntimeError(
                "Qwen-AgentWorld is required for lucky_loop_lab but LUCKYWORLD_SIMULATOR_BASE_URL "
                "and LUCKYWORLD_SIMULATOR_MODEL are not configured."
            )
        return plumbing_prediction(action)

    base_url = os.environ["LUCKYWORLD_SIMULATOR_BASE_URL"]
    model = os.environ["LUCKYWORLD_SIMULATOR_MODEL"]
    api_key = os.getenv("LUCKYWORLD_SIMULATOR_API_KEY", "dummy")
    timeout_seconds = float(os.getenv("LUCKYWORLD_SIMULATOR_TIMEOUT_SECONDS", "120"))
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds, max_retries=1)
    payload = {
        "lab_state": _compact_state(state),
        "candidate_action": action.model_dump(),
        "instructions": {
            "predict_environment_not_truth": True,
            "proof_comes_from_real_python_execution": True,
            "do_not_make_claims_from_prediction": True,
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
    text = response.choices[0].message.content or "{}"
    return normalize_prediction(_json_from_text(text), "qwen_agentworld")
