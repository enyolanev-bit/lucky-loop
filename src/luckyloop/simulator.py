from __future__ import annotations
import json, os, re
from openai import OpenAI
from .schemas import Prediction, ProposedAction

SYSTEM = """You are a language world model simulating a research coding environment.
Given the current research state and a proposed ML experiment, predict likely metric range, runtime, failure modes, and whether to run it.
Return strict JSON only, with exactly these keys:
- expected_metric: string, e.g. "accuracy around 0.94-0.97"
- expected_runtime_seconds: string, e.g. "under 5"
- risks: array of strings
- recommendation: one of "run", "skip", "modify"
- rationale: string
Do not use markdown. Do not claim actual execution. Predict only."""


def heuristic_prediction(action: ProposedAction, state: str) -> Prediction:
    m = action.model
    if m == "logistic_regression" and action.params.get("scale"):
        return Prediction(expected_metric="accuracy around 0.95-0.98", expected_runtime_seconds="under 5", risks=["minor convergence warning possible"], recommendation="run", rationale="Scaling usually helps logistic regression on tabular medical data.")
    if m == "logistic_regression":
        return Prediction(expected_metric="accuracy around 0.92-0.96", expected_runtime_seconds="under 5", risks=["unscaled features can slow convergence"], recommendation="run", rationale="Good baseline for breast cancer dataset.")
    if m == "random_forest":
        return Prediction(expected_metric="accuracy around 0.94-0.98", expected_runtime_seconds="under 10", risks=["overfitting if max_depth is unconstrained"], recommendation="run", rationale="Tree ensembles are strong baselines and robust to scaling.")
    if "boost" in m:
        return Prediction(expected_metric="accuracy around 0.94-0.98", expected_runtime_seconds="under 15", risks=["can overfit with too many estimators"], recommendation="run", rationale="Boosting often performs well but needs tuning.")
    if m == "svc":
        return Prediction(expected_metric="accuracy around 0.94-0.98", expected_runtime_seconds="under 10", risks=["sensitive to scaling and C"], recommendation="run", rationale="Scaled RBF SVC is strong on small tabular datasets.")
    if m == "verification_sweep":
        return Prediction(expected_metric="accuracy around 0.94-0.98", expected_runtime_seconds="under 25", risks=["label noise can make small hyperparameter differences non-robust", "seed variance may exceed apparent gains"], recommendation="run", rationale="A multi-seed sweep is useful for the deterministic verifier: it tests whether an apparent gain survives effect-vs-noise scrutiny.")
    if m == "weak_effect":
        return Prediction(expected_metric="accuracy around 0.94-0.99", expected_runtime_seconds="under 5", risks=["effect may be smaller than seed noise", "claim should likely be weak or blocked"], recommendation="run", rationale="A weak-effect scenario tests whether the verifier refuses to overclaim a small apparent improvement.")
    if m == "real_effect":
        return Prediction(expected_metric="accuracy around 0.76-0.88", expected_runtime_seconds="under 5", risks=["large effect should be checked against seed noise"], recommendation="run", rationale="A real-effect scenario should produce a supported claim when the effect/noise ratio is high.")
    if m == "data_leakage_trap":
        return Prediction(expected_metric="accuracy around 0.95-1.00", expected_runtime_seconds="under 5", risks=["suspiciously high accuracy may indicate data leakage", "protocol warning should block a strong claim"], recommendation="run", rationale="A leakage trap tests whether the system blocks claims from invalid protocols even when metrics look excellent.")
    if m == "metric_misuse":
        return Prediction(expected_metric="balanced_accuracy around 0.50-0.73", expected_runtime_seconds="under 5", risks=["accuracy may be misleading under class imbalance", "balanced accuracy or F1 should be preferred"], recommendation="run", rationale="Metric misuse tests whether the report avoids accuracy-only claims on an imbalanced setting.")
    return Prediction(expected_metric="unknown", expected_runtime_seconds="unknown", risks=["no calibrated prior"], recommendation="run", rationale="Fallback prediction.")


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
