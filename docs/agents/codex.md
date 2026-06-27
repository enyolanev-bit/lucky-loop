# Codex Handoff

Use Codex as the autoresearch planner without an API key.

1. Start Lucky Loop:

```bash
PYTHONPATH=src python -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode agent_handoff \
  --agent-backend codex_handoff \
  --agent-timeout-seconds 900
```

2. Lucky Loop writes:

```text
agent_io/<task_id>/<state_id>.request.json
```

3. Ask Codex to read the request and write:

```text
agent_io/<task_id>/<state_id>.response.json
```

Codex must follow `docs/agent_contract.md` and choose only one action ID from the catalog.

The trace will show:

```text
planner_mode = agent_handoff
agent_backend = codex_handoff
```
