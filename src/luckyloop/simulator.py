from __future__ import annotations
import json, os, re
from openai import OpenAI
from .schemas import Prediction, PredictedNextState, ProposedAction

PROMPT_VERSION = "world_model_prompt_v4_research_action"
SCHEMA_VERSION = "prediction_schema_v4_research_action"

SYSTEM = """You are Qwen-AgentWorld acting as a language world model for ML autoresearch.
Your job is not to plan after the fact. Given state s_t and candidate research action a_t, predict the next experimental observation o_t+1, the next research state s_t+1, and whether this action changes what the final report can honestly claim before compute is spent.

Be specific to the action and dataset. Prefer concrete scientific signals over generic cautions.
For sklearn tabular classification:
- mention feature scaling when the action is logistic regression or SVC and scaling is relevant
- mention small-dataset variance when a single split could overstate a model win
- mention overfitting or variance for tree ensembles when appropriate
- mention robustness sweeps before strong claims when a high single-run score exists
- mention that label-noise sweeps test claim robustness, not leaderboard performance
- when the action is top_model_verification, predict whether top single-run models are likely robustly separated or tied across seeds
- for sonar, mention small acoustic/sonar sample size, mine-vs-rock fragility, and high seed variance
- for eeg_eye_state, mention neural time-series measurements and possible protocol fragility from random splits
- for har, mention smartphone sensor features, high dimensionality, runtime cost, and split/protocol sensitivity
- if the action is unlikely to change what the final report can claim, say so and recommend skip or stop_and_report
- if the action is needed to turn observations into evidence, use claim_impact=high and recommendation=verify
- estimate value of information: prefer actions that reduce uncertainty or resolve claim status per unit compute
- explicitly say whether this action changes the final report, just adds a score, or should stop the loop
- explicitly contrast the action with classic score-chasing autoresearch
- protocol_probe actions test whether an impressive result should be blocked by protocol, metric, split, or leakage risk

Return strict JSON only, with exactly these keys:
- expected_metric: string, e.g. "accuracy around 0.94-0.97"
- expected_runtime_seconds: string, e.g. "under 5"
- expected_metric_range: two floats or null
- expected_runtime_range_seconds: two floats or null
- risks: array of strings
- recommendation: one of "run", "skip", "modify", "verify", "stop_and_report"
- rationale: string
- action_specific_signal: string
- claim_risk: string
- claim_impact: one of "low", "medium", "high"
- compute_value: one of "low", "medium", "high"
- why_this_action_changes_claims: string
- why_this_action_may_be_wasteful: string
- risk_predictions: array of strings chosen from ["seed_variance", "overfitting", "scaling_sensitivity", "runtime_cost", "single_split_overclaim", "metric_misuse", "data_leakage", "low_claim_impact"]
- stop_condition: string
- predicted_next_state: object with keys:
  - likely_best_model: string or null
  - expected_metric_delta: string
  - uncertainty_reduction: one of "low", "medium", "high"
  - claim_status_after_action: one of "observation_only", "needs_verification", "likely_inconclusive", "likely_supported", "report_ready"
  - likely_next_open_questions: array of strings
  - recommended_followup: string
  - expected_compute_cost_seconds: float or null
  - expected_research_value: one of "low", "medium", "high"
- expected_value_of_information: float from 0.0 to 1.0
- expected_claim_resolution: float from 0.0 to 1.0
- cost_aware_recommendation_reason: string
- predicted_observation: string
- expected_claim_delta: one of "none", "adds_observation", "reduces_uncertainty", "enables_claim", "blocks_or_rewrites_claim", "report_ready"
- protocol_risks: array of strings
- compute_waste_risk: float from 0.0 to 1.0
- why_not_classic_autoresearch: string
Do not use markdown. Do not claim actual execution. Predict only."""

FEW_SHOTS = [
    {
        "example_id": "fewshot_scaling_helped",
        "state": "breast_cancer baseline logistic_regression without scaling reached about 0.951 accuracy.",
        "action": "run scaled logistic_regression",
        "prediction": "Scaling should improve or stabilize logistic regression because breast_cancer numeric features have different scales; claim impact is medium until repeated seeds.",
        "actual": "scaled logistic_regression reached about 0.986 single-run accuracy.",
        "lesson": "Scaling was a useful intervention, but the result remained an observation until robustness verification.",
    },
    {
        "example_id": "fewshot_random_forest_overestimated",
        "state": "scaled linear models were already strong and top models were close.",
        "action": "run random_forest",
        "prediction": "Random forest may be competitive, but claim impact is low if it only adds another single-run score.",
        "actual": "random_forest underperformed the best scaled linear/SVC runs on the observed split.",
        "lesson": "Tree ensembles can be useful comparisons, but after a close top-model state they may be non-claimable score chasing.",
    },
    {
        "example_id": "fewshot_verifier_blocked",
        "state": "top models are close and a single-run winner exists.",
        "action": "run top_model_verification",
        "prediction": "Claim impact is high; matched seeds may show the effect is smaller than seed noise.",
        "actual": "verifier returned inconclusive because effect/noise was below the trust threshold.",
        "lesson": "Verification blocked the robust best-model claim and should trigger honest reporting or stop_and_report under a strict claim objective.",
    },
]


def heuristic_prediction(action: ProposedAction, state: str) -> Prediction:
    m = action.model
    if m == "logistic_regression" and action.params.get("scale"):
        return Prediction(expected_metric="accuracy around 0.95-0.98", expected_runtime_seconds="under 5", expected_metric_range=[0.95, 0.98], expected_runtime_range_seconds=[0.0, 5.0], risks=["minor convergence warning possible"], recommendation="run", rationale="Scaling usually helps logistic regression on tabular numeric features.", action_specific_signal="Feature scaling is the direct intervention for a linear model after an unscaled baseline.", claim_risk="Single-run best score should not be claimed as robust without repeated seeds.", claim_impact="medium", compute_value="high", why_this_action_changes_claims="It tests whether the unscaled baseline was limited by feature scale.", why_this_action_may_be_wasteful="It becomes wasteful after multiple scaled linear variants are already known.", risk_predictions=["scaling_sensitivity", "single_split_overclaim"], stop_condition="Stop chasing scaled variants once top models require multi-seed verification.", prompt_version=PROMPT_VERSION, world_model_schema_version=SCHEMA_VERSION)
    if m == "logistic_regression":
        return Prediction(expected_metric="accuracy around 0.90-0.97", expected_runtime_seconds="under 5", expected_metric_range=[0.90, 0.97], expected_runtime_range_seconds=[0.0, 5.0], risks=["unscaled features can slow convergence or underperform when feature scales differ"], recommendation="run", rationale="A simple linear baseline is cheap and informative for sklearn tabular classification.", action_specific_signal="Establish a baseline before testing scaling, nonlinear models, or sweeps.", claim_risk="Baseline-only evidence cannot support a best-model claim.", claim_impact="medium", compute_value="high", why_this_action_changes_claims="It anchors later interventions and defines the baseline.", why_this_action_may_be_wasteful="It is redundant after a baseline already exists.", risk_predictions=["scaling_sensitivity"], stop_condition="After baseline, test the most direct intervention rather than repeat unscaled runs.", prompt_version=PROMPT_VERSION, world_model_schema_version=SCHEMA_VERSION)
    if m == "random_forest":
        return Prediction(expected_metric="accuracy around 0.90-0.98", expected_runtime_seconds="under 10", expected_metric_range=[0.90, 0.98], expected_runtime_range_seconds=[0.0, 10.0], risks=["overfitting or split variance if depth is unconstrained"], recommendation="run", rationale="Tree ensembles test a different inductive bias, but may not beat a strong scaled linear model on small tabular datasets.", action_specific_signal="Use random forest to test whether nonlinear interactions improve over the linear baseline.", claim_risk="A single tree-ensemble score should be treated as observation until checked across seeds.", claim_impact="low" if "needs_robustness_verification" in state else "medium", compute_value="medium", why_this_action_changes_claims="It can falsify the assumption that a linear model is sufficient.", why_this_action_may_be_wasteful="If top models are already close, another single-run tree score cannot support a robust claim.", risk_predictions=["overfitting", "single_split_overclaim"], stop_condition="Skip tree variants when verification is already required.", prompt_version=PROMPT_VERSION, world_model_schema_version=SCHEMA_VERSION)
    if "boost" in m:
        return Prediction(expected_metric="accuracy around 0.90-0.98", expected_runtime_seconds="under 15", expected_metric_range=[0.90, 0.98], expected_runtime_range_seconds=[0.0, 15.0], risks=["can overfit with too many estimators"], recommendation="run", rationale="Boosting is a useful late comparison when linear and bagged-tree baselines are known.", action_specific_signal="Use boosting to test a staged-tree alternative under the remaining budget.", claim_risk="Do not report a robust winner from one boosted run.", claim_impact="low" if "needs_robustness_verification" in state else "medium", compute_value="medium", why_this_action_changes_claims="It tests a staged-tree alternative.", why_this_action_may_be_wasteful="It is low claim value after a verifier is already needed.", risk_predictions=["overfitting", "runtime_cost", "single_split_overclaim"], stop_condition="Skip if expected claim impact is low and verification is pending.", prompt_version=PROMPT_VERSION, world_model_schema_version=SCHEMA_VERSION)
    if m == "svc":
        return Prediction(expected_metric="accuracy around 0.90-0.99", expected_runtime_seconds="under 10", expected_metric_range=[0.90, 0.99], expected_runtime_range_seconds=[0.0, 10.0], risks=["sensitive to scaling and C"], recommendation="run", rationale="Scaled SVC can be strong on small numeric classification tasks but needs careful comparison.", action_specific_signal="SVC tests a margin-based nonlinear hypothesis after linear/tree baselines.", claim_risk="SVC should not be declared best without repeated-seed evidence.", claim_impact="medium", compute_value="medium", why_this_action_changes_claims="It tests whether a margin-based nonlinear classifier beats scaled linear baselines.", why_this_action_may_be_wasteful="It is weak evidence if it only adds a single split to a close top-model race.", risk_predictions=["scaling_sensitivity", "single_split_overclaim"], stop_condition="Verify top models once SVC and linear models are close.", prompt_version=PROMPT_VERSION, world_model_schema_version=SCHEMA_VERSION)
    if m == "verification_sweep":
        return Prediction(expected_metric="accuracy around 0.85-0.98", expected_runtime_seconds="under 35", expected_metric_range=[0.85, 0.98], expected_runtime_range_seconds=[5.0, 35.0], risks=["label noise can make small hyperparameter differences non-robust", "seed variance may exceed apparent gains"], recommendation="verify", rationale="A multi-seed sweep is useful for the deterministic verifier: it tests whether an apparent gain survives effect-vs-noise scrutiny.", action_specific_signal="Run a robustness sweep before allowing a best-hyperparameter claim.", claim_risk="The apparent winner may be blocked if effect size is smaller than seed noise.", claim_impact="high", compute_value="high", why_this_action_changes_claims="It directly determines whether a hyperparameter claim can survive the trust ladder.", why_this_action_may_be_wasteful="It is wasteful only if no claim candidate exists yet.", risk_predictions=["seed_variance", "single_split_overclaim"], stop_condition="If effect/noise is below threshold, stop making robust winner claims.", prompt_version=PROMPT_VERSION, world_model_schema_version=SCHEMA_VERSION)
    if m == "top_model_verification":
        return Prediction(expected_metric="accuracy around 0.90-0.99", expected_runtime_seconds="under 45", expected_metric_range=[0.90, 0.99], expected_runtime_range_seconds=[10.0, 45.0], risks=["top single-run models may be tied across seeds", "seed variance may exceed the observed top-model gap"], recommendation="verify", rationale="A multi-seed top-model comparison is required before reporting a robust best model.", action_specific_signal="Verify the top observed models on matched seeds before allowing a best-model claim.", claim_risk="Best observed single-run model may not be a robust winner.", claim_impact="high", compute_value="high", why_this_action_changes_claims="It is the gate between a best observed score and an allowed robust best-model claim.", why_this_action_may_be_wasteful="It is wasteful only before top candidates exist.", risk_predictions=["seed_variance", "single_split_overclaim"], stop_condition="If verification is inconclusive, stop_and_report instead of running extra score-chasing models.", prompt_version=PROMPT_VERSION, world_model_schema_version=SCHEMA_VERSION)
    if m == "protocol_probe":
        return Prediction(expected_metric="best mean accuracy may look high but should be claim-blocked by protocol warning", expected_runtime_seconds="under 35", expected_metric_range=[0.80, 0.99], expected_runtime_range_seconds=[5.0, 35.0], risks=["metric-only or protocol-fragile result can look claimable without a verifier", "protocol warning should block a strong scientific claim"], recommendation="verify", rationale="A protocol probe intentionally tests whether the lab can reject an impressive-looking but scientifically fragile result.", action_specific_signal="Probe protocol risk before allowing the report to treat a score as a discovery.", claim_risk="Classic autoresearch may report the apparent winner and miss the protocol warning.", claim_impact="high", compute_value="high", why_this_action_changes_claims="It can block or rewrite a tempting claim when the protocol is unsafe.", why_this_action_may_be_wasteful="It is wasteful before there is any score that might enter the report.", risk_predictions=["metric_misuse", "single_split_overclaim"], stop_condition="If protocol warning fires, report the limitation instead of claiming a valid winner.", prompt_version=PROMPT_VERSION, world_model_schema_version=SCHEMA_VERSION)
    if m == "stop_and_report":
        return Prediction(expected_metric="no metric; reporting action", expected_runtime_seconds="under 1", expected_metric_range=None, expected_runtime_range_seconds=[0.0, 1.0], risks=["stopping may leave only inconclusive findings"], recommendation="stop_and_report", rationale="If the verifier has already blocked a robust best-model claim, further score-chasing is unlikely to change what can be honestly reported under the current objective.", action_specific_signal="Stop after verifier evidence instead of spending compute on non-claimable score chasing.", claim_risk="The final report must state the robust claim was blocked or remains inconclusive.", claim_impact="high", compute_value="high", why_this_action_changes_claims="It prevents unsupported claims and closes the loop with an honest report.", why_this_action_may_be_wasteful="It is premature before a verifier result exists.", risk_predictions=["low_claim_impact", "single_split_overclaim"], stop_condition="Stop when the verifier is inconclusive and remaining candidates only add single-run observations.", prompt_version=PROMPT_VERSION, world_model_schema_version=SCHEMA_VERSION)
    return Prediction(expected_metric="unknown", expected_runtime_seconds="unknown", risks=["no calibrated prior"], recommendation="run", rationale="Fallback prediction.", action_specific_signal="No action-specific prior available.", claim_risk="No claim should be made from this prediction alone.", claim_impact="low", compute_value="low", why_this_action_changes_claims="", why_this_action_may_be_wasteful="No calibrated prior means this action may not change claim status.", risk_predictions=["low_claim_impact"], stop_condition="Stop if no claim-relevant candidate is available.", prompt_version=PROMPT_VERSION, world_model_schema_version=SCHEMA_VERSION)


def _json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in model response")
    return json.loads(m.group(0))


def _normalize_prediction(data: dict) -> dict:
    if "expected_metric" not in data and "metric_range" in data:
        data["expected_metric"] = str(data["metric_range"])
    if "expected_runtime_seconds" not in data:
        for key in ("runtime_seconds_range", "runtime_estimate", "runtime_seconds"):
            if key in data:
                data["expected_runtime_seconds"] = str(data[key])
                break
    if "expected_metric" in data and not isinstance(data["expected_metric"], str):
        data["expected_metric"] = str(data["expected_metric"])
    if "expected_runtime_seconds" in data and not isinstance(data["expected_runtime_seconds"], str):
        data["expected_runtime_seconds"] = str(data["expected_runtime_seconds"])
    if "expected_metric_range" not in data:
        for key in ("metric_range", "expected_metric_interval"):
            if key in data:
                data["expected_metric_range"] = data[key]
                break
    if "expected_runtime_range_seconds" not in data:
        for key in ("runtime_seconds_range", "runtime_range", "expected_runtime_interval_seconds"):
            if key in data:
                data["expected_runtime_range_seconds"] = data[key]
                break
    rec = str(data.get("recommendation", "run")).lower()
    if rec not in {"run", "skip", "modify", "verify", "stop_and_report"}:
        rec = "modify" if any(word in rec for word in ["scale", "change", "adjust", "modify"]) else "run"
    data["recommendation"] = rec
    if "risks" not in data or data["risks"] is None:
        data["risks"] = []
    if isinstance(data["risks"], str):
        data["risks"] = [data["risks"]]
    data.setdefault("rationale", "")
    data.setdefault("action_specific_signal", "")
    data.setdefault("claim_risk", "")
    data["expected_metric_range"] = _clean_float_pair(data.get("expected_metric_range"))
    data["expected_runtime_range_seconds"] = _clean_float_pair(data.get("expected_runtime_range_seconds"))
    data["claim_impact"] = _one_of(data.get("claim_impact"), {"low", "medium", "high"}, "medium")
    data["compute_value"] = _one_of(data.get("compute_value"), {"low", "medium", "high"}, "medium")
    data.setdefault("why_this_action_changes_claims", "")
    data.setdefault("why_this_action_may_be_wasteful", "")
    data["risk_predictions"] = _string_list(data.get("risk_predictions", []))
    data.setdefault("stop_condition", "")
    data["predicted_next_state"] = _normalize_next_state(data)
    data["expected_value_of_information"] = _bounded_float(
        data.get("expected_value_of_information"),
        _default_value_of_information(data),
    )
    data["expected_claim_resolution"] = _bounded_float(
        data.get("expected_claim_resolution"),
        _default_claim_resolution(data),
    )
    data.setdefault("cost_aware_recommendation_reason", "")
    data.setdefault("predicted_observation", data.get("rationale", ""))
    data["expected_claim_delta"] = _one_of(
        data.get("expected_claim_delta"),
        {"none", "adds_observation", "reduces_uncertainty", "enables_claim", "blocks_or_rewrites_claim", "report_ready"},
        _default_claim_delta(data),
    )
    data["protocol_risks"] = _string_list(data.get("protocol_risks", data.get("risk_predictions", [])))
    data["compute_waste_risk"] = _bounded_float(
        data.get("compute_waste_risk"),
        _default_compute_waste_risk(data),
    )
    data.setdefault("why_not_classic_autoresearch", _default_why_not_classic(data))
    data.setdefault("prompt_version", PROMPT_VERSION)
    data.setdefault("world_model_schema_version", SCHEMA_VERSION)
    return data


def _normalize_next_state(data: dict) -> dict:
    raw = data.get("predicted_next_state")
    if not isinstance(raw, dict):
        raw = {}
    recommendation = str(data.get("recommendation", "run"))
    claim_impact = str(data.get("claim_impact", "medium"))
    compute_value = str(data.get("compute_value", "medium"))
    if recommendation == "verify":
        default_status = "needs_verification"
        default_followup = "Use verifier output to allow, block, or rewrite the best-model claim."
    elif recommendation == "stop_and_report":
        default_status = "report_ready"
        default_followup = "Write the honest report from current evidence and avoid further score-chasing."
    elif claim_impact == "high":
        default_status = "needs_verification"
        default_followup = "Check whether the result changes allowed claims."
    else:
        default_status = "observation_only"
        default_followup = "Update the evidence table and reconsider whether verification is now needed."
    if compute_value == "high" or claim_impact == "high":
        default_value = "high"
    elif compute_value == "low" and claim_impact == "low":
        default_value = "low"
    else:
        default_value = "medium"
    normalized = {
        "likely_best_model": raw.get("likely_best_model"),
        "expected_metric_delta": str(raw.get("expected_metric_delta") or "unknown"),
        "uncertainty_reduction": _one_of(raw.get("uncertainty_reduction"), {"low", "medium", "high"}, "medium"),
        "claim_status_after_action": _one_of(
            raw.get("claim_status_after_action"),
            {"observation_only", "needs_verification", "likely_inconclusive", "likely_supported", "report_ready"},
            default_status,
        ),
        "likely_next_open_questions": _string_list(raw.get("likely_next_open_questions", [])),
        "recommended_followup": str(raw.get("recommended_followup") or default_followup),
        "expected_compute_cost_seconds": _optional_float(raw.get("expected_compute_cost_seconds")),
        "expected_research_value": _one_of(raw.get("expected_research_value"), {"low", "medium", "high"}, default_value),
    }
    return PredictedNextState(**normalized).model_dump()


def _clean_float_pair(value) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", value)]
    elif isinstance(value, (list, tuple)):
        nums = []
        for item in value:
            try:
                nums.append(float(item))
            except (TypeError, ValueError):
                pass
    else:
        return None
    if len(nums) < 2:
        return None
    return [min(nums[:2]), max(nums[:2])]


def _one_of(value, allowed: set[str], default: str) -> str:
    text = str(value or default).lower()
    return text if text in allowed else default


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded_float(value, default: float) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        parsed = default
    return max(0.0, min(1.0, parsed))


def _default_value_of_information(data: dict) -> float:
    if data.get("recommendation") in {"verify", "stop_and_report"}:
        return 0.9
    if data.get("claim_impact") == "high" or data.get("compute_value") == "high":
        return 0.75
    if data.get("claim_impact") == "low" and data.get("compute_value") == "low":
        return 0.2
    return 0.5


def _default_claim_resolution(data: dict) -> float:
    if data.get("recommendation") in {"verify", "stop_and_report"}:
        return 0.9
    if data.get("claim_impact") == "high":
        return 0.75
    if data.get("claim_impact") == "low":
        return 0.2
    return 0.5


def _default_claim_delta(data: dict) -> str:
    rec = data.get("recommendation")
    claim_impact = data.get("claim_impact")
    status = (data.get("predicted_next_state") or {}).get("claim_status_after_action")
    if rec == "stop_and_report" or status == "report_ready":
        return "report_ready"
    if rec == "verify":
        return "blocks_or_rewrites_claim"
    if claim_impact == "high":
        return "enables_claim"
    if data.get("expected_value_of_information", 0.5) >= 0.7:
        return "reduces_uncertainty"
    if claim_impact == "low":
        return "none"
    return "adds_observation"


def _default_compute_waste_risk(data: dict) -> float:
    if data.get("recommendation") in {"skip", "stop_and_report"}:
        return 0.9
    if data.get("claim_impact") == "low" and data.get("compute_value") == "low":
        return 0.8
    if data.get("expected_claim_resolution", 0.5) >= 0.75:
        return 0.15
    return 0.45


def _default_why_not_classic(data: dict) -> str:
    if data.get("recommendation") == "verify":
        return "Classic score-chasing might launch another model run, while this action resolves whether an existing result can become a claim."
    if data.get("recommendation") == "stop_and_report":
        return "Classic score-chasing might spend remaining budget on non-claimable scores, while this action closes the loop honestly."
    if data.get("claim_impact") == "low":
        return "A classic policy may overvalue the immediate metric even though this action is unlikely to change claim status."
    return "This action is evaluated by predicted research value and claim impact, not only by immediate score."


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _prompt_context(memory_examples: list[dict] | None) -> str:
    memory_examples = memory_examples or []
    return json.dumps(
        {
            "few_shot_examples": FEW_SHOTS,
            "retrieved_similar_past_cases": memory_examples,
            "instructions": (
                "Use few-shot examples as calibration priors. Use retrieved cases only as past lessons; "
                "do not treat them as the actual result of the candidate being predicted."
            ),
        },
        indent=2,
    )


def _enrich_prediction(prediction: Prediction) -> Prediction:
    return Prediction(**_normalize_prediction(prediction.model_dump(exclude_defaults=True)))


def predict(action: ProposedAction, state: str, memory_examples: list[dict] | None = None) -> Prediction:
    base_url = os.getenv("LUCKYWORLD_SIMULATOR_BASE_URL")
    model = os.getenv("LUCKYWORLD_SIMULATOR_MODEL")
    api_key = os.getenv("LUCKYWORLD_SIMULATOR_API_KEY", "dummy")
    timeout_seconds = float(os.getenv("LUCKYWORLD_SIMULATOR_TIMEOUT_SECONDS", "45"))
    if not base_url or not model:
        prediction = _enrich_prediction(heuristic_prediction(action, state))
        prediction.memory_example_ids = [item.get("memory_id", "") for item in (memory_examples or []) if item.get("memory_id")]
        prediction.few_shot_example_ids = [item["example_id"] for item in FEW_SHOTS]
        return prediction
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds, max_retries=0)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Current state:\n{state}\n\n"
                        f"Few-shot and retrieved calibration context:\n{_prompt_context(memory_examples)}\n\n"
                        f"Proposed action:\n{action.model_dump_json(indent=2)}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=700,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = resp.choices[0].message.content or "{}"
        data = _normalize_prediction(_json_from_text(content))
        data["memory_example_ids"] = [item.get("memory_id", "") for item in (memory_examples or []) if item.get("memory_id")]
        data["few_shot_example_ids"] = [item["example_id"] for item in FEW_SHOTS]
        return Prediction(**data)
    except Exception as e:
        hp = _enrich_prediction(heuristic_prediction(action, state))
        hp.memory_example_ids = [item.get("memory_id", "") for item in (memory_examples or []) if item.get("memory_id")]
        hp.few_shot_example_ids = [item["example_id"] for item in FEW_SHOTS]
        hp.rationale += f" Fallback used because simulator endpoint failed: {type(e).__name__}: {e}"
        hp.risks.append("simulator endpoint unavailable; heuristic fallback used")
        return hp
