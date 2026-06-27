# OpenClaw Handoff

Lucky Loop does not depend on OpenClaw internals. It exposes a stable request/response contract.

If OpenClaw is available as a CLI, configure:

```bash
export LUCKYLOOP_AGENT_COMMAND="openclaw run-agent {request_path} {response_path}"

PYTHONPATH=src python -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode agent_command \
  --agent-backend openclaw_command
```

If OpenClaw is interactive, use file handoff:

```bash
PYTHONPATH=src python -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode agent_handoff \
  --agent-backend openclaw_handoff
```

The agent must read `agent_io/<task_id>/<state_id>.request.json` and write `agent_io/<task_id>/<state_id>.response.json` following `docs/agent_contract.md`.
