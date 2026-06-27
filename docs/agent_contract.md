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

Lucky Loop validates the response with Pydantic. For external agents (`agent_handoff` / `agent_command`), invalid action_ids are **overridden by the safety selector** (`validate_agent_decision(..., override_on_invalid=True)`): an invalid `preferred_action_id` is replaced with a safe catalog action, unknown `candidate_action_ids` are dropped, and the override is recorded in the rationale for audit. The strict `llm` planner still rejects invalid actions by raising.

## Command Mode

> Security: `LUCKYLOOP_AGENT_COMMAND` is run via the shell and is **trusted input only** — set it yourself, never from an untrusted source. Request/response paths are shell-quoted.


If an agent system exposes a CLI, use:

```bash
export LUCKYLOOP_AGENT_COMMAND="your-agent {request_path} {response_path}"

PYTHONPATH=src python -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode agent_command \
  --agent-backend custom_cli_agent
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

## Agent-In-Repo Mode

For Codex, Claude Code, OpenClaw, Hermes, or another coding agent that is already operating in the repository, use `program.md` as the runbook and keep the backend trace label explicit:

```bash
PYTHONPATH=src python -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode operator_driven \
  --agent-backend codex_operator
```

This mode uses the same catalog-only contract and trace schema as the API and handoff modes. It is the credential-free path for showing the backend without pretending that the Python process can call the currently attached coding agent as a private API.
