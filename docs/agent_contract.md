# Lucky Loop Agent Contract

Lucky Loop separates the autoresearch planner from the world model.

- The planner agent chooses an experiment from a safe catalog.
- Qwen-AgentWorld predicts the outcome before compute.
- Lucky Loop runs the real experiment and verifies claims.

## Request

In `planner_mode=agent_handoff`, Lucky Loop writes:

```text
agent_io/<task_id>/<state_id>.request.json
```

The request contains:

- `task`: dataset, metric, budget, candidate space.
- `state`: current evidence, budget, open questions, top-model summary.
- `candidate_catalog`: allowed actions. The agent must choose from these IDs.
- `prior_traces_summary`: compact history of previous runs.
- `planner_contract`: safety rules.
- `response_schema`: required response shape.

## Response

The external agent writes:

```text
agent_io/<task_id>/<state_id>.response.json
```

Response JSON:

```json
{
  "research_question": "Maximize validation accuracy while avoiding unsupported claims.",
  "working_hypothesis": "Feature scaling should improve a scale-sensitive linear baseline.",
  "candidate_action_ids": ["action_logistic_regression_C-1p0_scale-True"],
  "preferred_action_id": "action_logistic_regression_C-1p0_scale-True",
  "rationale": "The unscaled baseline is known; scaling is the next cheapest causal intervention.",
  "expected_evidence_needed": "Compare scaled logistic regression against the baseline.",
  "claim_risk": "A single improvement is observational until repeated-seed verification.",
  "stop_or_continue": "continue"
}
```

Rules:

- `preferred_action_id` must come from `candidate_catalog`.
- `candidate_action_ids` must only contain catalog IDs.
- Do not invent shell commands.
- Do not claim execution happened.
- Recommend top-model verification when single-run top models are close.

Lucky Loop validates the response with Pydantic. Invalid actions are rejected or overridden by the safety selector.

## Command Mode

If an agent system exposes a CLI, use:

```bash
export LUCKYLOOP_AGENT_COMMAND="your-agent {request_path} {response_path}"

PYTHONPATH=src python -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode agent_command \
  --agent-backend generic_command
```

For local testing:

```bash
export LUCKYLOOP_AGENT_COMMAND=".venv/bin/python scripts/fake_agent_backend.py {request_path} {response_path}"
```

## Handoff Mode

If the agent is interactive, use:

```bash
PYTHONPATH=src python -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode agent_handoff \
  --agent-backend codex_handoff
```

Lucky Loop waits for the response file. The external agent writes it, then Lucky Loop continues.
