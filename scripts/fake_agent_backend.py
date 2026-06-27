#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def choose_action(request: dict) -> dict:
    task = request["task"]
    state = request["state"]
    candidates = request["candidate_catalog"]
    traces = request.get("prior_traces_summary", [])

    def first_where(predicate):
        for candidate in candidates:
            if predicate(candidate):
                return candidate
        return candidates[0]

    top_summary = state.get("top_model_summary") or {}
    if top_summary.get("needs_robustness_verification"):
        preferred = first_where(lambda c: c["model"] == "top_model_verification")
        hypothesis = "The best observed single-run model may not be robust across matched seeds."
        evidence = "Run top-model verification before allowing a robust best-model claim."
        risk = "A single-split leaderboard can overstate model quality."
    elif not traces:
        preferred = first_where(lambda c: c["model"] == "logistic_regression" and not c.get("params", {}).get("scale"))
        hypothesis = "A simple baseline should anchor the research loop before interventions."
        evidence = "Measure baseline accuracy, f1, runtime, and prediction error."
        risk = "Baseline-only evidence cannot support a best-model claim."
    elif not any(t["model"] == "logistic_regression" and t.get("params", {}).get("scale") for t in traces):
        preferred = first_where(lambda c: c["model"] == "logistic_regression" and c.get("params", {}).get("scale"))
        hypothesis = "Feature scaling should improve or stabilize a scale-sensitive linear model."
        evidence = "Compare scaled logistic regression against the unscaled baseline."
        risk = "A single scaled run remains observational until repeated-seed verification."
    else:
        tested = {t["model"] for t in traces}
        preferred = first_where(lambda c: c["model"] not in tested and c["model"] not in {"verification_sweep", "top_model_verification"})
        hypothesis = "A new model family tests whether the current evidence depends on inductive bias."
        evidence = "Run a real experiment from an untested model family."
        risk = "Do not report a robust winner from a single split."

    candidate_ids = [candidate.get("action_id") or "" for candidate in candidates[:6]]
    if preferred.get("action_id") not in candidate_ids:
        candidate_ids.insert(0, preferred.get("action_id") or "")
    return {
        "research_question": task.get("goal") or f"Maximize {task.get('primary_metric', 'accuracy')} on {task['dataset']} while avoiding unsupported claims.",
        "working_hypothesis": hypothesis,
        "candidate_action_ids": candidate_ids,
        "preferred_action_id": preferred.get("action_id") or "",
        "rationale": f"Fake command backend selected {preferred.get('action_id')} from the safe catalog for state {state['state_id']}.",
        "expected_evidence_needed": evidence,
        "claim_risk": risk,
        "stop_or_continue": "continue",
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: fake_agent_backend.py REQUEST_PATH RESPONSE_PATH")
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response = choose_action(request)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
