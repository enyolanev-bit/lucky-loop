# Lucky Loop

Predict before compute. Verify before claim.

Lucky Loop is an ML-first auto-research backend built around a language world model. It starts from a natural-language research question, reviews domain literature, finds a public dataset, generates a hypothesis and experiment protocol, asks Qwen-AgentWorld to forecast each lab action, runs real Python experiments, and reports only claims that pass verification.

## What It Does

```text
question
-> domain literature review
-> method/safety literature review
-> research agenda + hypotheses
-> Hugging Face/OpenML dataset search
-> dataset audit + materialization
-> DeepSeek protocol + experiment code
-> Qwen-AgentWorld pre-action forecast
-> static validation + reduced dry-run
-> real sklearn execution
-> analysis + claim verification
-> next research decision
-> final report
```

## Why The World Model Matters

Most auto-research systems are reactive: they run an experiment, read the score, then decide what to do next. Lucky Loop adds a predictive step before compute.

Qwen-AgentWorld is a central component of the loop. For every lab action, it receives the current research state and candidate action, then predicts:

- likely terminal/output observation
- expected artifacts
- runtime and compute-waste risk
- likely failure modes
- protocol risks
- expected claim impact
- whether to run, modify, verify, skip, or stop

That prediction does not count as scientific evidence. It is a planning signal. The evidence still comes from real Python execution, and final claims still go through the verifier.

This is the main differentiation: Lucky Loop is not just an agent that runs ML experiments. It is a research loop that predicts the consequence of an experiment before spending compute, checks that prediction against reality, and uses the mismatch to make the next decision better.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` and set:

```bash
LUCKYLOOP_AGENT_BASE_URL=...
LUCKYLOOP_AGENT_MODEL=...
LUCKYLOOP_AGENT_API_KEY=...
LUCKYWORLD_SIMULATOR_BASE_URL=...
LUCKYWORLD_SIMULATOR_MODEL=...
LUCKYWORLD_SIMULATOR_API_KEY=...
```

## Run The Auto-Research Lab

```bash
set -a; source .env; set +a
PYTHONPATH=src python -m luckyloop.lab \
  --question "Can a nonlinear model robustly outperform logistic regression on public sensor classification data across repeated seeds?" \
  --budget 8
```

Outputs are generated under `reports/lab/<slug>/` and are ignored by git.

Key artifacts:

```text
literature/domain_related_work.md
literature/method_related_work.md
agenda/research_agenda.json
datasets/selection_rationale.json
protocol/generated_protocol.json
generated/experiment.py
runs/*.json
analyses/*.json
predictions/world_model_predictions.jsonl
claim_ledger.json
next_decision.json
final_report.md
```

## Validate A Run

```bash
PYTHONPATH=src python scripts/validate_lab_artifacts.py \
  --workspace reports/lab/<slug> \
  --require-qwen \
  --require-agent
```

Generate a readable demo summary:

```bash
PYTHONPATH=src python scripts/write_complete_demo.py \
  --workspace reports/lab/<slug> \
  --out reports/demo_complete_lab.md
```

## Legacy Benchmarks

The older benchmark/ablation path is still available for comparison work:

```bash
PYTHONPATH=src python scripts/run_ablation_suite.py --world-model auto
PYTHONPATH=src python scripts/validate_artifacts.py --check-ablations --require-qwen
```

Those commands also write to ignored `reports/` and `runs/` directories.

## Tests

```bash
PYTHONPATH=src pytest
```

## Repo Layout

```text
src/luckyloop/        backend package
experiments/         local sklearn experiment runners
configs/tasks/       legacy benchmark task specs
scripts/             validation, demo, and ablation commands
tests/               regression tests
reports/             generated reports, ignored except .gitkeep
runs/                generated run traces, ignored except .gitkeep
```

## Current Constraints

- The lab is ML-first, focused on sklearn-compatible supervised experiments.
- Generated code is sandboxed by static validation and a reduced dry-run.
- There is no production UI in this repo.
- No fallback templates are used in the open auto-research path; failed LLM contracts fail explicitly.
