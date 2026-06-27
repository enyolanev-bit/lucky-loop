# Lucky Loop pitch outline

## One-liner
Lucky Loop is a world-model-guided autonomous research loop: it predicts experiment outcomes before running real compute, then compares prediction vs reality and writes an auditable report.

## Tagline
Predict before you compute.

## Problem
Autonomous research agents are usually reactive. They launch code, observe failures or weak metrics, patch, and repeat. That wastes compute and produces reports that are hard to audit.

## Insight
Qwen-AgentWorld shows that language world models can simulate agent environments such as terminal, search, code editing, and tool/API calls. We apply that idea directly to research automation.

## Solution
Lucky Loop inserts Qwen-AgentWorld between planning and execution:

1. propose a research experiment
2. ask Qwen-AgentWorld to predict metric range, runtime, risks, and recommendation
3. execute the experiment for real
4. compare predicted vs actual metrics/logs
5. let that comparison choose the next experiment
6. produce JSON evidence traces and a final report

## Demo flow
Goal: run world-model-guided autoresearch on real sklearn benchmark tasks, then block unsupported claims.

Observed loop:

| Step | What happened |
|---|---|
| 1 | A TaskSpec declares dataset, metric, candidate models, search space, and compute budget. |
| 2 | The planner generates candidate actions from the spec, not from a hardcoded breast_cancer script. |
| 3 | Qwen-AgentWorld predicts each candidate's metric range, runtime, risks, and claim risk before compute. |
| 4 | The selector records whether the decision came from a world-model signal, selector policy, or both. |
| 5 | After real search, Lucky Loop detects the top observed models and asks Qwen whether a best-model claim needs verification. |
| 6 | The executor runs matched multi-seed top-model verification. |
| 7 | The comparator logs prediction hits and misses. The verifier allows or blocks the best-model claim. |

Current benchmark suite:

| Task | Runs | Best model | Best metric | Top-model verification |
|---|---:|---|---:|---:|
| breast_cancer_accuracy | 7 | logistic_regression | 0.9860 | 1 |
| wine_accuracy | 7 | logistic_regression | 1.0000 | 1 |
| digits_accuracy | 7 | svc | 0.9800 | 1 |

## Why this is not just AutoML
AutoML searches configurations. Lucky Loop records a scientific loop:

- hypothesis
- prediction
- real execution
- prediction-vs-actual delta
- lesson
- next decision

Every run is saved as JSON evidence under `runs/<task_id>/`, and each task gets a report, calibration file, demo summary, and claim ledger under `reports/<task_id>/`.

## Why it matters
Research agents need more than tool calls. They need a predictive model of the environment, so they can anticipate failures, choose better experiments, and produce auditable evidence.

## Architecture
User goal -> Planner -> Qwen-AgentWorld simulator -> real executor -> comparator -> next experiment -> report.

## Current working artifact
- Qwen-AgentWorld-35B-A3B served through vLLM on Team Pegasus MI300X, endpoint `http://134.199.205.222:8000/v1`.
- CLI benchmark suite verified on three real sklearn datasets.
- Streamlit UI verified locally at `http://127.0.0.1:8501`.
- Evidence files:
  - `runs/<task_id>/run_001.json` ...
  - `reports/<task_id>/final_report.md`
  - `reports/benchmark_summary.md`
  - `reports/pitch_outline.md`
