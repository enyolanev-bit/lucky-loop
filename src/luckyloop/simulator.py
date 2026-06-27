from __future__ import annotations
import json, os, re
from openai import OpenAI
from .schemas import Prediction, ProposedAction

SYSTEM = """You are a language world model simulating a real ML research environment.
Given a TaskSpec, the current research state, and one proposed sklearn experiment, predict the next experimental observation before any compute is spent.

Be specific to the action and dataset. Prefer concrete scientific signals over generic cautions.
For sklearn tabular classification:
- mention feature scaling when the action is logistic regression or SVC and scaling is relevant
- mention small-dataset variance when a single split could overstate a model win
- mention overfitting or variance for tree ensembles when appropriate
- mention robustness sweeps before strong claims when a high single-run score exists
- mention that label-noise sweeps test claim robustness, not leaderboard performance
- when the action is top_model_verification, predict whether top single-run models are likely robustly separated or tied across seeds

Return strict JSON only, with exactly these keys:
- expected_metric: string, e.g. "accuracy around 0.94-0.97"
- expected_runtime_seconds: string, e.g. "under 5"
- risks: array of strings
- recommendation: one of "run", "skip", "modify"
- rationale: string
- action_specific_signal: string
- claim_risk: string
Do not use markdown. Do not claim actual execution. Predict only."""


def heuristic_prediction(action: ProposedAction, state: str) -> Prediction:
    m = action.model
    if m == "logistic_regression" and action.params.get("scale"):
        return Prediction(expected_metric="accuracy around 0.95-0.98", expected_runtime_seconds="under 5", risks=["minor convergence warning possible"], recommendation="run", rationale="Scaling usually helps logistic regression on tabular numeric features.", action_specific_signal="Feature scaling is the direct intervention for a linear model after an unscaled baseline.", claim_risk="Single-run best score should not be claimed as robust without repeated seeds.")
    if m == "logistic_regression":
        return Prediction(expected_metric="accuracy around 0.90-0.97", expected_runtime_seconds="under 5", risks=["unscaled features can slow convergence or underperform when feature scales differ"], recommendation="run", rationale="A simple linear baseline is cheap and informative for sklearn tabular classification.", action_specific_signal="Establish a baseline before testing scaling, nonlinear models, or sweeps.", claim_risk="Baseline-only evidence cannot support a best-model claim.")
    if m == "random_forest":
        return Prediction(expected_metric="accuracy around 0.90-0.98", expected_runtime_seconds="under 10", risks=["overfitting or split variance if depth is unconstrained"], recommendation="run", rationale="Tree ensembles test a different inductive bias, but may not beat a strong scaled linear model on small tabular datasets.", action_specific_signal="Use random forest to test whether nonlinear interactions improve over the linear baseline.", claim_risk="A single tree-ensemble score should be treated as observation until checked across seeds.")
    if "boost" in m:
        return Prediction(expected_metric="accuracy around 0.90-0.98", expected_runtime_seconds="under 15", risks=["can overfit with too many estimators"], recommendation="run", rationale="Boosting is a useful late comparison when linear and bagged-tree baselines are known.", action_specific_signal="Use boosting to test a staged-tree alternative under the remaining budget.", claim_risk="Do not report a robust winner from one boosted run.")
    if m == "svc":
        return Prediction(expected_metric="accuracy around 0.90-0.99", expected_runtime_seconds="under 10", risks=["sensitive to scaling and C"], recommendation="run", rationale="Scaled SVC can be strong on small numeric classification tasks but needs careful comparison.", action_specific_signal="SVC tests a margin-based nonlinear hypothesis after linear/tree baselines.", claim_risk="SVC should not be declared best without repeated-seed evidence.")
    if m == "verification_sweep":
        return Prediction(expected_metric="accuracy around 0.85-0.98", expected_runtime_seconds="under 35", risks=["label noise can make small hyperparameter differences non-robust", "seed variance may exceed apparent gains"], recommendation="run", rationale="A multi-seed sweep is useful for the deterministic verifier: it tests whether an apparent gain survives effect-vs-noise scrutiny.", action_specific_signal="Run a robustness sweep before allowing a best-hyperparameter claim.", claim_risk="The apparent winner may be blocked if effect size is smaller than seed noise.")
    if m == "top_model_verification":
        return Prediction(expected_metric="accuracy around 0.90-0.99", expected_runtime_seconds="under 45", risks=["top single-run models may be tied across seeds", "seed variance may exceed the observed top-model gap"], recommendation="run", rationale="A multi-seed top-model comparison is required before reporting a robust best model.", action_specific_signal="Verify the top observed models on matched seeds before allowing a best-model claim.", claim_risk="Best observed single-run model may not be a robust winner.")
    return Prediction(expected_metric="unknown", expected_runtime_seconds="unknown", risks=["no calibrated prior"], recommendation="run", rationale="Fallback prediction.", action_specific_signal="No action-specific prior available.", claim_risk="No claim should be made from this prediction alone.")


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
    rec = str(data.get("recommendation", "run")).lower()
    if rec not in {"run", "skip", "modify"}:
        rec = "modify" if any(word in rec for word in ["scale", "change", "adjust", "modify"]) else "run"
    data["recommendation"] = rec
    if "risks" not in data or data["risks"] is None:
        data["risks"] = []
    if isinstance(data["risks"], str):
        data["risks"] = [data["risks"]]
    data.setdefault("rationale", "")
    data.setdefault("action_specific_signal", "")
    data.setdefault("claim_risk", "")
    return data


def predict(action: ProposedAction, state: str) -> Prediction:
    base_url = os.getenv("LUCKYWORLD_SIMULATOR_BASE_URL")
    model = os.getenv("LUCKYWORLD_SIMULATOR_MODEL")
    api_key = os.getenv("LUCKYWORLD_SIMULATOR_API_KEY", "dummy")
    if not base_url or not model:
        return heuristic_prediction(action, state)
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Current state:\n{state}\n\n"
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
        return Prediction(**data)
    except Exception as e:
        hp = heuristic_prediction(action, state)
        hp.rationale += f" Fallback used because simulator endpoint failed: {type(e).__name__}: {e}"
        hp.risks.append("simulator endpoint unavailable; heuristic fallback used")
        return hp
