# Claude Code Handoff

Lucky Loop can use Claude Code through the same file contract as Codex.

Recommended command shape, if a CLI wrapper is available:

```bash
export LUCKYLOOP_AGENT_COMMAND="claude-code-agent {request_path} {response_path}"

PYTHONPATH=src python -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode agent_command \
  --agent-backend claude_code_command
```

If no CLI wrapper is available, run:

```bash
PYTHONPATH=src python -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode agent_handoff \
  --agent-backend claude_code_handoff
```

Then ask Claude Code to read the request JSON and write the response JSON described in `docs/agent_contract.md`.
