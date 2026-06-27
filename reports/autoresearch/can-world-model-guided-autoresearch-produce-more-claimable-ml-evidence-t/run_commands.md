# Run Commands

## Full Agent-Operated Research

```bash
export PYTHONPATH=src
export LUCKYWORLD_SIMULATOR_BASE_URL=http://134.199.205.222:8000/v1
export LUCKYWORLD_SIMULATOR_MODEL=Qwen/Qwen-AgentWorld-35B-A3B
export LUCKYWORLD_SIMULATOR_API_KEY=dummy
export LUCKYWORLD_SIMULATOR_TIMEOUT_SECONDS=25

python -m luckyloop.autoresearch \
  --question "Can world-model-guided autoresearch produce more claimable ML evidence than classic autoresearch?" \
  --agent codex_operator \
  --tasks configs/tasks/breast_cancer_accuracy.json configs/tasks/wine_accuracy.json configs/tasks/digits_accuracy.json \
  --execute
```

## Re-run Core Ablation Only

```bash
PYTHONPATH=src python scripts/run_ablation_suite.py --world-model auto --operator-agent codex_operator
PYTHONPATH=src python scripts/run_budgeted_compute_evaluation.py
```

## Validate

```bash
PYTHONPATH=src python scripts/validate_artifacts.py --check-ablations --require-qwen
```
