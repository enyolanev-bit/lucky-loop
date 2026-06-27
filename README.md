# Lucky Loop: World-Model-Guided Autonomous Research with Claim-Calibrated Reporting

> Paris Research Hackathon (TUM.ai x Iterate), Track 3 "Lucky Loop". Team Pegasus.

## Tagline

**Predict before compute. Verify before claim.**

## One-Liner

Lucky Loop is a world-model-guided autonomous research loop: an autoresearch agent proposes experiments, Qwen-AgentWorld predicts what should happen before compute is spent, real code tests reality, and a deterministic verifier decides which claims survive the evidence.

## Why This Matters

Most autoresearch agents act by trial and error: they choose an experiment, run code, inspect the metric, and write a report. That is useful, but it misses two research-grade behaviors:

1. **Foresight before action.** A research agent should estimate likely outcomes, risks, and runtime before spending compute.
2. **Discipline after action.** A research agent should not turn fragile or noisy results into confident scientific claims.

Lucky Loop adds both layers. The world model is the signature: Qwen-AgentWorld acts as an experimental simulator. The verifier is the trust gate: it prevents unsupported findings from appearing as claims.

## Core Loop

```text
research state
-> API-backed autoresearch agent proposes candidate experiments
-> Qwen-AgentWorld predicts outcome / runtime / risks for each candidate
-> safety selector validates the agent choice using world-model and evidence-risk signals
-> executor runs the real sklearn experiment
-> comparator measures prediction vs reality
-> verifier gates claims with evidence
-> claim ledger + honest report + demo UI
```

The important distinction:

- **Autoresearch agent:** proposes actions and decides what to run.
- **Qwen-AgentWorld:** predicts future observations for candidate actions.
- **Executor:** runs real code and logs measured metrics.
- **Comparator:** records whether the prediction matched reality.
- **Verifier:** deterministic claim gate, not an LLM judge.

Operating mode:

- **Product path:** the autoresearch agent is an OpenAI-compatible planner API. It proposes hypotheses and chooses action IDs from a safe catalog.
- **Agent-in-repo path:** `--planner-mode operator_driven` represents Codex, Claude Code, OpenClaw, Hermes, or another coding agent operating from `program.md` against the same backend protocol.
- **No-key agent path:** `--planner-mode agent_handoff` lets Codex, Claude Code, OpenClaw, or another external agent read a request JSON and write an `AgentDecision` response.
- **Automatic agent path:** `--planner-mode agent_command` calls any CLI adapter through `LUCKYLOOP_AGENT_COMMAND`.
- **Test path:** `--planner-mode replay` uses a local fixture-like planner for smoke tests only.
- **Compatibility path:** `--planner-mode selector` keeps the previous transparent selector available, but it is not the main demo path.

Qwen-AgentWorld is never treated as the research agent or verifier. It is the world model that forecasts candidate-action outcomes.

## What Works Now

- Qwen-AgentWorld-35B-A3B is served through vLLM on Team Pegasus MI300X.
- OpenAI-compatible endpoint: `http://134.199.205.222:8000/v1`.
- `src/luckyloop/` contains the runnable task-agnostic ML research loop:
  - task specs for sklearn benchmarks
  - API-first, agent-in-repo, and external-agent planner interfaces
  - adaptive candidate generation from dataset/model/search-space config
  - world-model prediction
  - real experiment execution
  - prediction-vs-actual comparison
  - automatic top-model detection
  - matched multi-seed top-model verification
  - real ablation against classic autoresearch and classic verified baselines
  - JSON evidence traces
  - Markdown report
- Benchmark tasks exist for real sklearn datasets:
  - `breast_cancer_accuracy`
  - `wine_accuracy`
  - `digits_accuracy`
- Each benchmark uses real sklearn training commands, detects the top observed models, asks the world model whether a best-model claim needs verification, and runs matched multi-seed top-model checks before claims are allowed. The legacy controlled probes are not the primary demo path.

## Demo Message

```text
Most AI scientists hallucinate after the experiment.
Lucky Loop makes a prediction before the experiment,
runs the real code, compares prediction with reality,
and only claims what survives verification.
```

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH=src
export LUCKYWORLD_SIMULATOR_BASE_URL=http://134.199.205.222:8000/v1
export LUCKYWORLD_SIMULATOR_MODEL=Qwen/Qwen-AgentWorld-35B-A3B
export LUCKYWORLD_SIMULATOR_API_KEY=dummy

python3 -m luckyloop.autoresearch \
  --question "Can world-model-guided autoresearch produce more claimable ML evidence than classic autoresearch?" \
  --agent codex_operator \
  --execute
```

This is the main agent-operated path: Codex, Claude Code, OpenClaw, Hermes, or an API planner acts as the autoresearch agent in the repo. Lucky Loop provides the world-model prediction layer, real execution harness, comparator, verifier, claim ledger, and evidence reports.

The autoresearch workspace includes an auditable literature layer:

- official arXiv Atom API search with curated fallback
- deduplicated `sources.json`
- stable arXiv IDs/versions where available
- source -> gap -> metric -> experiment mapping
- citations carried into the final report

Search arXiv directly:

```bash
PYTHONPATH=src python3 scripts/search_arxiv.py "Qwen-AgentWorld language world models" --max-results 5
```

To run one task directly:

```bash
python3 -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode operator_driven \
  --agent-backend codex_operator
```

If the simulator endpoint is unavailable, Lucky Loop keeps a deterministic heuristic fallback for local smoke tests. The live presentation path uses Qwen-AgentWorld.

The future API-backed autoresearch agent uses:

```bash
export LUCKYLOOP_AGENT_BASE_URL=...
export LUCKYLOOP_AGENT_MODEL=...
export LUCKYLOOP_AGENT_API_KEY=...

PYTHONPATH=src python3 -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode llm
```

Use Codex/Claude Code/OpenClaw as an interactive handoff planner without an API key:

```bash
PYTHONPATH=src python3 -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode agent_handoff \
  --agent-backend codex_handoff
```

Lucky Loop writes `agent_io/<task_id>/<state_id>.request.json` and waits for `*.response.json`.
The response contract is documented in `docs/agent_contract.md`.

Custom CLI-backed agents are supported through the same contract:

```bash
export LUCKYLOOP_AGENT_COMMAND="your-agent {request_path} {response_path}"
PYTHONPATH=src python3 -m luckyloop.loop \
  --task configs/tasks/breast_cancer_accuracy.json \
  --planner-mode agent_command \
  --agent-backend custom_cli_agent
```

Run all real benchmark tasks:

```bash
PYTHONPATH=src python3 scripts/run_benchmark_suite.py \
  --planner-mode operator_driven \
  --agent-backend codex_operator
```

Run the backend ablation suite:

```bash
PYTHONPATH=src python3 scripts/run_ablation_suite.py --world-model auto
```

Validate presentation artifacts:

```bash
PYTHONPATH=src python3 scripts/validate_artifacts.py --check-ablations --require-qwen
```

## Artifacts

```text
runs/<task_id>/run_001.json ...
reports/<task_id>/final_report.md
reports/<task_id>/demo_summary.md
reports/<task_id>/world_model_calibration.md
reports/<task_id>/claim_ledger.json
reports/benchmark_summary.md
reports/ablations/world_model_ablation.md
reports/ablations/classic_vs_lucky_loop.md
reports/autoresearch/<question_slug>/
reports/autoresearch/<question_slug>/literature/sources.json
reports/pitch_backend_summary.md
app/streamlit_app.py
```

The root `runs/` and `reports/` directories may also contain the latest single-task local run.
