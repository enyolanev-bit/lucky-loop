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
-> autoresearch agent proposes candidate experiments
-> Qwen-AgentWorld predicts outcome / runtime / risks for each candidate
-> planner selects an action using the world-model signal
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

- **Hackathon implementation:** the autoresearch agent is the planner/selector in `src/luckyloop/`. It proposes actions, asks Qwen-AgentWorld for predictions, runs experiments, and records evidence.
- **Autonomous extension:** the same role can be backed by a planner API while preserving the same state/action/prediction trace format.

Qwen-AgentWorld is never treated as the research agent or verifier. It is the world model that forecasts candidate-action outcomes.

## What Works Now

- Qwen-AgentWorld-35B-A3B is served through vLLM on Team Pegasus MI300X.
- OpenAI-compatible endpoint: `http://134.199.205.222:8000/v1`.
- `src/luckyloop/` contains the runnable research loop:
  - world-model prediction
  - real experiment execution
  - prediction-vs-actual comparison
  - JSON evidence traces
  - Markdown report
- Six real runs exist under `runs/`.
- The current verifier already blocks an overclaim in `run_005`:
  - `effect_size = 0.020979`
  - `seed_noise = 0.027972`
  - verdict: `inconclusive`

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

python3 -m luckyloop.loop --max-experiments 6
```

If the simulator endpoint is unavailable, Lucky Loop keeps a deterministic heuristic fallback for local smoke tests. The live presentation path uses Qwen-AgentWorld.

## Artifacts

```text
runs/run_001.json ... run_006.json
reports/final_report.md
app/streamlit_app.py
```

Next build targets:

- trace schema v2 with explicit state, candidates, predictions, decision trace, and claim ledger updates
- multi-candidate world-model prediction before selection
- world-model calibration report
- trust ladder verifier
- claim ledger
- judge-ready Streamlit timeline
