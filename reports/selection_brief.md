# Lucky Loop selection brief

**Team Pegasus** — Paris Research Hackathon, Track 3 “Lucky Loop”.

## Core claim

Lucky Loop is not just an AutoML loop. It is an auditable research loop with two guardrails most autoresearch demos miss:

1. **Predict before compute**: Qwen-AgentWorld predicts metric range, runtime, risks, and recommendation before a candidate experiment is executed.
2. **Verify before claim**: a deterministic verifier and claim ledger block unsupported or protocol-invalid scientific claims.

Tagline: **Predict before compute. Verify before claim.**

## What the demo proves

| Evidence | Result |
|---|---:|
| Real experiment traces | 10 runs under `runs/` |
| World-model metric interval coverage | 80% |
| Runtime interval coverage | 90% |
| Useful world-model decision signals | 10/10 |
| Prediction misses kept visible | 3 |
| Claim ledger entries | 5 |
| Strong supported claim | 1 |
| Blocked overclaims | 4 |
| Independent IC95 cross-check agreement | 2/5 |

The important result is not “the world model is always right”. The important result is that Lucky Loop measures when it is right, exposes when it is wrong, and prevents weak evidence from becoming a confident claim.

## Judge-ready story

### 1. Foresight before action

The loop keeps explicit research state, proposes candidate experiments, and asks Qwen-AgentWorld what should happen before spending compute. The prediction is structured:

| Field | Example |
|---|---|
| metric | `accuracy around 0.94-0.98` |
| runtime | `under 25` |
| risks | `seed variance may exceed apparent gains` |
| recommendation | `run` |
| rationale | why this action is useful now |

### 2. Reality check

The executor runs real sklearn experiments. The comparator records whether the prediction matched reality. Misses remain in the report:

| Miss | Why it matters |
|---|---|
| run_002 metric exceeded predicted interval | world model was useful but not omniscient |
| run_004 runtime exceeded prediction | compute-cost miss remains auditable |
| run_010 metric exceeded predicted interval | report does not hide favorable misses |

### 3. Claim discipline

The verifier separates observations from claims. A high score is not automatically a discovery.

| Case | Apparent result | Lucky Loop verdict |
|---|---|---|
| noisy C sweep | `C=0.1` had best mean | blocked, effect smaller than seed noise |
| real effect | candidate beat baseline by 0.1000 | strongly supported |
| weak effect | candidate mean was higher | blocked, effect/noise too low |
| data leakage trap | leaky protocol scored ~1.0 | blocked by protocol warning |
| metric misuse | balanced objective won | blocked from overclaim due metric/protocol warning |

### 4. Independent cross-check

An additive script, `experiments/verifier_crosscheck.py`, recomputes a paired 95% CI best-vs-second and writes `reports/verifier_crosscheck.md`.

Result: **2/5 verdicts concordent**.

That is useful, not bad:

- It confirms the strong positive case.
- It rejects the noisy C sweep.
- It shows where the trust ladder is deliberately stricter than pure metric significance, especially protocol warnings.

## Demo line

> Most AI scientists hallucinate after the experiment. Lucky Loop makes a prediction before the experiment, runs real code, compares prediction with reality, and only claims what survives verification.

## What to show live

1. Open the Streamlit timeline.
2. Show one run card with: world model prediction → planner decision → actual result → verifier verdict.
3. Jump to run_004: apparent C winner blocked because effect < seed noise.
4. Jump to run_007: leaky winner blocked despite high metric.
5. Show claim ledger and cross-check report.
6. End with the slogan: **Predict before compute. Verify before claim.**

## Files to point judges to

| File | Purpose |
|---|---|
| `README.md` | main framing and run command |
| `runs/run_*.json` | auditable evidence traces |
| `reports/final_report.md` | full generated research report |
| `reports/world_model_calibration.md` | prediction-vs-reality calibration |
| `reports/claim_ledger.json` | supported/blocked claims |
| `reports/verifier_crosscheck.md` | independent IC95 cross-check |
| `app/streamlit_app.py` | judge-facing timeline UI |
