# LuckyWorld

LuckyWorld is the working implementation of **Lucky Loop: World-Model-Guided Autonomous Research with Claim-Calibrated Reporting**.

Tagline: **Predict before compute. Verify before claim.**

LuckyWorld wraps an autoresearch loop with a language world model. The agent proposes candidate experiments, Qwen-AgentWorld predicts likely metrics, runtime, risks, and recommendations before compute, real sklearn code tests the selected action, and a deterministic verifier gates which claims can appear in the final report.

## Roles

```text
Autoresearch agent / planner
-> proposes candidate experiments and decides the next action

Qwen-AgentWorld
-> predicts what should happen for each candidate action

Executor
-> runs real sklearn experiments

Comparator
-> measures prediction-vs-reality

Verifier
-> blocks unsupported scientific claims

Claim ledger / reporter / UI
-> expose evidence and write only allowed claims
```

For the hackathon build, Codex can operate as the autoresearch agent while LuckyWorld records the same trace shape expected from an autonomous planner.

## What Works Now

- Qwen-AgentWorld-35B-A3B served through vLLM on Team Pegasus MI300X.
- OpenAI-compatible endpoint: `http://134.199.205.222:8000/v1`.
- End-to-end loop verified on `sklearn` breast cancer.
- Six real experiments generated JSON traces and a Markdown report.
- Controlled noisy-label multi-seed sweep produces an honest verifier verdict.
- Streamlit reads prediction, actual metric, and verifier fields.

Observed best result:

```text
scaled logistic regression accuracy = 0.9860
```

Honest verifier moment:

```text
run_005 noisy-label C sweep
effect_size = 0.020979
seed_noise = 0.027972
verdict = inconclusive
blocked claim = C=0.1 is robustly better
allowed claim = C=0.1 had the best mean, but the effect was smaller than seed noise
```

## Architecture

Current implementation:

```text
Goal
-> world-model prediction with Qwen-AgentWorld
-> real sklearn execution
-> prediction-vs-actual comparison
-> deterministic effect-vs-noise verifier for sweep claims
-> JSON evidence trace
-> final report
```

Target implementation:

```text
Goal
-> explicit research state
-> candidate actions
-> Qwen-AgentWorld prediction for each candidate
-> selector decision trace
-> real sklearn execution
-> prediction-vs-actual comparison
-> verifier + claim ledger
-> calibration report
-> final report + judge UI
```

The key point is that Qwen-AgentWorld is not the research agent and not the verifier. It is the world model: it predicts experimental consequences before compute.

## Run Locally

```bash
cd luckyworld
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH=src
export LUCKYWORLD_SIMULATOR_BASE_URL=http://134.199.205.222:8000/v1
export LUCKYWORLD_SIMULATOR_MODEL=Qwen/Qwen-AgentWorld-35B-A3B
export LUCKYWORLD_SIMULATOR_API_KEY=dummy

python3 -m luckyworld.loop --max-experiments 6
```

If no simulator endpoint is configured, LuckyWorld falls back to a deterministic heuristic predictor so local smoke tests still run. The presentation path uses live Qwen-AgentWorld.

## Evidence

Generated traces:

```text
runs/run_001.json
runs/run_002.json
runs/run_003.json
runs/run_004.json
runs/run_005.json
runs/run_006.json
```

Report:

```text
reports/final_report.md
```

Pitch outline:

```text
reports/pitch_outline.md
```

## Streamlit UI

```bash
cd luckyworld
. .venv/bin/activate
streamlit run app/streamlit_app.py --server.headless true
```

## Next Build Targets

- Trace schema v2: state, candidates, predictions, decision trace, comparison, verifier, claim ledger updates.
- Candidate planner: predict multiple possible futures before selecting an experiment.
- World-model calibration report: interval coverage, misses, runtime error, useful decisions.
- Trust ladder verifier: inconclusive, weakly supported, supported, strongly supported.
- Claim ledger: supported, weak, and blocked claims with evidence run IDs.
- Judge-ready Streamlit UI: World model said / Planner did / Reality showed / Verifier allowed or blocked.
