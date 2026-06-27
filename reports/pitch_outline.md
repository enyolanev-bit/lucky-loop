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
Goal: maximize validation accuracy on sklearn breast cancer in five experiments.

Observed loop:

| Step | What happened |
|---|---|
| 1 | Unscaled logistic regression baseline was predicted around 0.93-0.96 accuracy and actually scored 0.9510. |
| 2 | Qwen-AgentWorld warned about scaling, so the policy selected scaled logistic regression next. It scored 0.9860. |
| 3 | Since the linear baseline was strong, the policy tested random forest for a different inductive bias. It scored 0.9580. |
| 4 | Because the tree prediction was overestimated, the policy tried a scaled RBF SVC. It also scored 0.9860. |
| 5 | Final conservative gradient boosting scored 0.9510. |

Best observed model: scaled logistic regression, accuracy 0.9860.

## Why this is not just AutoML
AutoML searches configurations. Lucky Loop records a scientific loop:

- hypothesis
- prediction
- real execution
- prediction-vs-actual delta
- lesson
- next decision

Every run is saved as JSON evidence under `runs/`, and the final report is generated at `reports/final_report.md`.

## Why it matters
Research agents need more than tool calls. They need a predictive model of the environment, so they can anticipate failures, choose better experiments, and produce auditable evidence.

## Architecture
User goal -> Planner -> Qwen-AgentWorld simulator -> real executor -> comparator -> next experiment -> report.

## Current working artifact
- Qwen-AgentWorld-35B-A3B served through vLLM on Team Pegasus MI300X, endpoint `http://134.199.205.222:8000/v1`.
- CLI loop verified with five real experiments.
- Streamlit UI verified locally at `http://127.0.0.1:8501`.
- Evidence files:
  - `runs/run_001.json` ... `runs/run_005.json`
  - `reports/final_report.md`
  - `reports/pitch_outline.md`
