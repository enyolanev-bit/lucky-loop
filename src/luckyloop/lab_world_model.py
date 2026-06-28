from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from .schemas import LabAction, LabPrediction, LabStudyState


LAB_WORLD_MODEL_PROMPT_VERSION = "lab_computer_world_model_v3"
LAB_WORLD_MODEL_SCHEMA_VERSION = "lab_prediction_schema_v3"


SYSTEM = """You are Qwen-AgentWorld, a world model for an ML research lab.
Predict the next lab state before the action executes. Output one JSON object only. No markdown.
Do not use neutral claim_support_probability=0.5.
Compare the action with an alternative such as stop/report, revise claim, search dataset, or reduce seeds.
Dry-runs cannot support scientific claims. Full runs can support, weaken, or block claims. Reports should use recommendation=stop_and_report."""


def simulator_configured() -> bool:
    return bool(os.getenv("LUCKYWORLD_SIMULATOR_BASE_URL") and os.getenv("LUCKYWORLD_SIMULATOR_MODEL"))


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
    if text in allowed:
        return text
    if "stop" in text and "stop_and_report" in allowed:
        return "stop_and_report"
    if any(term in text for term in ["proceed", "run", "execute"]) and "run" in allowed:
        return "run"
    if "verify" in text and "verify" in allowed:
        return "verify"
    if "modify" in text and "modify" in allowed:
        return "modify"
    if "skip" in text and "skip" in allowed:
        return "skip"
    if "report_ready" in allowed and ("report" in text or "ready" in text):
        return "report_ready"
    if "block" in text and "blocks_or_rewrites_claim" in allowed:
        return "blocks_or_rewrites_claim"
    if "enable" in text and "enables_claim" in allowed:
        return "enables_claim"
    if any(term in text for term in ["reduce", "uncertain", "uncertainty", "observation", "improvement", "accuracy"]) and "reduces_uncertainty" in allowed:
        return "reduces_uncertainty"
    return default


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
    if isinstance(value, str):
        text = value.strip().lower()
        qualitative = {
            "none": 0.0,
            "very low": 0.1,
            "low": 0.25,
            "medium": 0.5,
            "moderate": 0.5,
            "high": 0.75,
            "very high": 0.9,
        }
        if text in qualitative:
            return qualitative[text]
    parsed = _float_or_none(value)
    if parsed is None:
        return default
    return max(0.0, min(1.0, parsed))


def _prediction_quality_failures(action: LabAction, prediction: LabPrediction) -> list[str]:
    failures = []
    if abs(prediction.claim_support_probability - 0.5) < 0.001:
        failures.append("claim_support_probability must not be neutral 0.5")
    if abs(prediction.value_of_information - 0.5) < 0.001:
        failures.append("value_of_information must not be neutral 0.5")
    if action.estimated_cost_class != "cheap" and prediction.compute_waste_risk <= 0.0:
        failures.append("compute_waste_risk must be positive for non-cheap actions")
    if not prediction.runtime_risk.strip():
        failures.append("runtime_risk is required")
    if not prediction.discriminative_reason.strip():
        failures.append("discriminative_reason is required")
    if not (prediction.rationale.strip() or prediction.why_not_score_chasing.strip()):
        failures.append("rationale or why_not_score_chasing is required")
    if not (prediction.failure_modes or prediction.protocol_risks):
        failures.append("failure_modes or protocol_risks are required")
    if action.expected_artifacts:
        if not prediction.predicted_artifacts:
            failures.append("predicted_artifacts are required for artifact-producing actions")
        unexpected = [item for item in prediction.predicted_artifacts if item not in action.expected_artifacts]
        if unexpected:
            failures.append(f"predicted_artifacts must only use expected_artifacts; unexpected={unexpected}")
    if action.kind in {"run_protocol", "run_replication", "run_ablation", "dry_run"}:
        if not prediction.preferred_next_action_if_blocked.strip():
            failures.append("preferred_next_action_if_blocked is required for executable protocol actions")
        if not prediction.what_would_change_my_mind.strip():
            failures.append("what_would_change_my_mind is required for executable protocol actions")
    if action.kind == "stop_and_report":
        if prediction.recommendation != "stop_and_report":
            failures.append("stop_and_report action must recommend stop_and_report")
        if prediction.expected_claim_delta != "report_ready":
            failures.append("stop_and_report action must set expected_claim_delta=report_ready")
    return failures


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
        claim_support_probability=_bounded(data.get("claim_support_probability"), 0.5),
        expected_best_model=str(data.get("expected_best_model") or ""),
        preferred_next_action_if_blocked=str(data.get("preferred_next_action_if_blocked") or ""),
        what_would_change_my_mind=str(data.get("what_would_change_my_mind") or ""),
        discriminative_reason=str(data.get("discriminative_reason") or ""),
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
        claim_support_probability=0.5,
        expected_best_model="",
        preferred_next_action_if_blocked="",
        what_would_change_my_mind="",
        discriminative_reason="No world-model call was made.",
        expected_claim_delta=action.claim_delta_target,
        recommendation="run" if action.kind != "stop_and_report" else "stop_and_report",
        why_not_score_chasing="Plumbing mode does not provide world-model evidence.",
        rationale="No world-model prediction was requested.",
        prompt_version=LAB_WORLD_MODEL_PROMPT_VERSION,
        world_model_schema_version=LAB_WORLD_MODEL_SCHEMA_VERSION,
    )


def parse_failed_prediction(action: LabAction, error: Exception) -> LabPrediction:
    expensive = action.estimated_cost_class in {"moderate", "expensive"}
    executable = action.kind in {"run_protocol", "run_replication", "run_ablation", "dry_run"}
    return LabPrediction(
        source="qwen_parse_failed",
        predicted_terminal_observation="World-model response was malformed; controller used action metadata.",
        expected_result_pattern="action may progress, but forecast quality is degraded",
        predicted_artifacts=action.expected_artifacts,
        failure_modes=["malformed_world_model_json"],
        protocol_risks=action.protocol_risks or ["world_model_parse_failure"],
        runtime_risk="higher risk; no parsed world-model forecast",
        expected_runtime_seconds=None,
        expected_runtime_range_seconds=[],
        compute_waste_risk=0.65 if expensive else 0.25,
        value_of_information=0.75 if executable else 0.65,
        suggested_modification="continue, then verify with real execution evidence",
        decision_threshold="do not use malformed forecast as evidence",
        claim_support_probability=0.35 if executable else 0.45,
        expected_best_model="unknown until execution",
        preferred_next_action_if_blocked="verify artifacts and retry world-model call",
        what_would_change_my_mind="successful execution with verified artifacts",
        discriminative_reason=f"Qwen JSON parse failed: {type(error).__name__}",
        expected_claim_delta=action.claim_delta_target,
        recommendation="run" if action.kind != "stop_and_report" else "stop_and_report",
        why_not_score_chasing="fallback is planning-only; verifier still gates claims",
        rationale="Malformed world-model output should not block autoresearch.",
        prompt_version=LAB_WORLD_MODEL_PROMPT_VERSION,
        world_model_schema_version=LAB_WORLD_MODEL_SCHEMA_VERSION,
    )


def _compact_state(state: LabStudyState) -> dict:
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
    return {
        "question": state.lab_question.question,
        "budget": state.lab_question.budget,
        "summary": state.summary,
        "hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
                "claim_candidate": item.claim_candidate,
                "minimum_evidence_needed": item.minimum_evidence_needed,
                "falsification_condition": item.falsification_condition,
            }
            for item in state.hypotheses[-3:]
        ],
        "protocols": [
            {
                "protocol_id": item.protocol_id,
                "scientific_goal": item.scientific_goal,
                "dataset": item.dataset,
                "conditions": item.conditions,
                "seeds": item.seeds,
                "claim_enabled_if": item.claim_enabled_if,
                "claim_blocked_if": item.claim_blocked_if,
                "protocol_risks": item.protocol_risks,
            }
            for item in state.protocols[-3:]
        ],
        "observations": compact_observations,
        "analyses": [analysis.model_dump() for analysis in state.analyses[-3:]],
        "claims": [claim.model_dump() for claim in state.claims[-5:]],
    }


def _compact_action(action: LabAction) -> dict:
    return {
        "action_id": action.action_id,
        "kind": action.kind,
        "scientific_goal": action.scientific_goal,
        "expected_artifacts": action.expected_artifacts,
        "protocol_risks": action.protocol_risks,
        "estimated_cost_class": action.estimated_cost_class,
        "claim_delta_target": action.claim_delta_target,
        "command_hint": "dry_run" if "--dry-run" in action.command else "python_execution" if action.command.startswith("python ") else "internal_stage",
        "controls": action.controls,
        "manipulated_variables": action.manipulated_variables,
        "primary_metric": action.primary_metric,
    }


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
        "candidate_action": _compact_action(action),
        "implicit_alternatives": [
            "stop_and_report",
            "search_better_dataset",
            "revise_claim",
            "add_or_reduce_seeds",
            "simplify_protocol",
        ],
        "instructions": {
            "predict_environment_not_truth": True,
            "proof_comes_from_real_python_execution": True,
            "do_not_make_claims_from_prediction": True,
            "must_be_action_specific": True,
            "must_compare_against_alternatives": True,
            "neutral_defaults_are_invalid": True,
            "claim_support_probability_must_not_equal_0_5": True,
            "prediction_is_rejected_if_generic": True,
            "required_action_specific_fields": [
                "runtime_risk",
                "claim_support_probability",
                "failure_modes_or_protocol_risks",
                "preferred_next_action_if_blocked",
                "what_would_change_my_mind",
                "discriminative_reason",
                "rationale_or_why_not_score_chasing",
            ],
        },
    }
    user_prompt = (
        "Predict the next ML lab state for this action. Return JSON only with keys: "
        "predicted_terminal_observation, expected_result_pattern, predicted_artifacts, failure_modes, "
        "protocol_risks, runtime_risk, expected_runtime_seconds, expected_runtime_range_seconds, "
        "compute_waste_risk, value_of_information, suggested_modification, decision_threshold, "
        "claim_support_probability, expected_best_model, preferred_next_action_if_blocked, "
        "what_would_change_my_mind, discriminative_reason, expected_claim_delta, recommendation, "
        "why_not_score_chasing, rationale. "
        "Keep every string under 12 words. Use only listed artifacts. "
        "Use numeric decimals for compute_waste_risk and value_of_information, not low/medium/high. "
        "Allowed expected_claim_delta: none, adds_observation, reduces_uncertainty, enables_claim, "
        "blocks_or_rewrites_claim, report_ready. Allowed recommendation: run, skip, modify, verify, "
        "stop_and_report. Payload: "
        f"{json.dumps(payload, separators=(',', ':'), sort_keys=True)}"
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    last_failures = []
    last_prediction = None
    for attempt in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=4000,
        )
        text = response.choices[0].message.content or "{}"
        try:
            prediction = normalize_prediction(_json_from_text(text), "qwen_agentworld")
        except Exception as exc:
            return parse_failed_prediction(action, exc)
        failures = _prediction_quality_failures(action, prediction)
        if not failures:
            return prediction
        last_failures = failures
        last_prediction = prediction
        messages.extend(
            [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Your prediction failed the world-model quality gate for "
                        f"{action.action_id}: {last_failures}. Return strict JSON again. "
                        "Do not use claim_support_probability=0.5. Compare this action against at least one "
                        "alternative and state how the action changes claim support, compute risk, and next decision."
                    ),
                },
            ]
        )
    if require_qwen:
        raise RuntimeError(f"Qwen prediction failed quality gate for {action.action_id}: {last_failures}")
    return last_prediction or plumbing_prediction(action)
