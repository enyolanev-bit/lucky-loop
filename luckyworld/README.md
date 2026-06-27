# LuckyWorld

World-model-guided autonomous research loops for the Paris Research Hackathon.

Tagline: **Predict before you compute.**

LuckyWorld asks a language world model to predict an ML experiment before spending real compute, then runs the experiment for real, compares prediction vs reality, and writes auditable traces.

## What works now

- Qwen-AgentWorld-35B-A3B served through vLLM on Team Pegasus MI300X.
- OpenAI-compatible endpoint: `http://134.199.205.222:8000/v1`.
- End-to-end loop verified on `sklearn` breast cancer.
- Five real experiments generated JSON traces and a Markdown report.
- A Streamlit timeline UI reads the traces.

Observed best result:

```text
scaled logistic regression accuracy = 0.9860
```

## Architecture

```text
Goal
-> world-model prediction with Qwen-AgentWorld
-> real sklearn execution
-> prediction-vs-actual comparison
-> next experiment decision
-> JSON evidence trace
-> final report
```

The important bit is that the model is not allowed to claim a result directly. It predicts. The executor measures. The comparator records the gap.

## Run locally

```bash
cd luckyworld
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH=src
export LUCKYWORLD_SIMULATOR_BASE_URL=http://134.199.205.222:8000/v1
export LUCKYWORLD_SIMULATOR_MODEL=Qwen/Qwen-AgentWorld-35B-A3B
export LUCKYWORLD_SIMULATOR_API_KEY=dummy

python -m luckyworld.loop --max-experiments 5
```

If no simulator endpoint is configured, LuckyWorld falls back to a deterministic heuristic predictor so the loop still runs.

## Evidence

Generated traces:

```text
runs/run_001.json
runs/run_002.json
runs/run_003.json
runs/run_004.json
runs/run_005.json
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

## Next step

Add a controlled perturbation scenario, for example noisy labels, bad split, data leakage trap, or timeout. This will make the demo closer to the Qwen-AgentWorld framing: a world model should anticipate environment traps before execution.
