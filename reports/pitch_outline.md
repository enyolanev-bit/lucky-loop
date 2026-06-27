# Lucky Loop pitch outline

## One-liner

Lucky Loop is a world-model-guided autonomous research loop: it predicts experiment outcomes before running real compute, compares prediction vs reality, and only lets evidence-backed claims survive.

## Tagline

**Predict before compute. Verify before claim.**

## Problem

Autonomous research agents are often reactive. They run code, inspect the metric, patch, repeat, then write a confident report. That wastes compute and creates a second failure mode: weak or protocol-invalid results can be turned into strong scientific claims.

## Insight

Qwen-AgentWorld can act as a language world model for agent environments. Lucky Loop applies that idea to research automation: before the agent spends compute, Qwen-AgentWorld predicts likely metrics, runtime, risks, and whether the action is worth running.

## Solution

Lucky Loop inserts a world-model and a claim gate into the research loop:

1. maintain explicit research state,
2. propose candidate experiments,
3. ask Qwen-AgentWorld to predict metric range, runtime, risks, recommendation,
4. execute the selected experiment for real,
5. compare predicted vs actual observations,
6. update the next decision from that evidence,
7. gate scientific claims with a deterministic verifier and claim ledger.

## Demo flow

Goal: maximize validation accuracy on sklearn breast cancer while keeping the research trace auditable.

| Moment | What to show | Message |
|---|---|---|
| Predict | Qwen-AgentWorld predicts metric/runtime/risks for candidate actions | The agent has foresight before compute |
| Execute | Real sklearn run produces measured metrics | This is not simulated evidence |
| Compare | Prediction hit/miss is recorded | Misses stay visible |
| Verify | Trust ladder blocks fragile claims | High score is not automatically a discovery |
| Ledger | Claims are allowed, blocked, or rewritten | The report is auditable |

## Evidence numbers

| Evidence | Result |
|---|---:|
| Real traces | 10 runs |
| Metric interval coverage | 80% |
| Runtime interval coverage | 90% |
| Useful world-model decisions | 10/10 |
| Prediction misses | 3 |
| Claim ledger entries | 5 |
| Strongly supported claims | 1 |
| Blocked overclaims | 4 |
| Independent IC95 cross-check agreement | 2/5 |

## Key demo examples

| Run | Judge point |
|---|---|
| `run_004` | apparent `C=0.1` winner is blocked because effect size is smaller than seed noise |
| `run_005` | large real effect becomes the one strongly supported claim |
| `run_006` | weak positive metric is blocked by the conservative trust ladder |
| `run_007` | data leakage trap is blocked even though the metric looks excellent |
| `run_008` | metric misuse trap avoids an accuracy-only overclaim |

## Why this is not AutoML

AutoML searches configurations. Lucky Loop records a scientific loop:

- hypothesis,
- predicted observation,
- real observation,
- prediction-vs-actual delta,
- lesson,
- next decision,
- claim verdict.

The final score matters less than the audit trail: the system measures its own predictive reliability and refuses unsupported claims.

## Cross-check framing

The additive IC95 cross-check reports `2/5 verdicts concordent`. Present it as a strength:

- It confirms the strong positive case.
- It rejects the noisy C sweep.
- It reveals where the trust ladder is stricter than pure metric significance.
- It shows protocol warnings matter: a statistically positive result can still be scientifically invalid.

## Closing line

Most AI scientists hallucinate after the experiment. Lucky Loop makes a prediction before the experiment, runs real code, compares prediction with reality, and only claims what survives verification.

## Artifacts

| File | Purpose |
|---|---|
| `runs/run_*.json` | evidence traces |
| `reports/final_report.md` | full generated report |
| `reports/world_model_calibration.md` | prediction-vs-reality calibration |
| `reports/claim_ledger.json` | allowed/blocked claims |
| `reports/verifier_crosscheck.md` | independent IC95 cross-check |
| `reports/selection_brief.md` | judge-ready one-pager |
| `app/streamlit_app.py` | timeline demo UI |
