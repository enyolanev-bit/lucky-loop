# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 42.86%
- Runtime interval coverage: 85.71%
- Mean metric absolute error outside interval: 0.0045
- Mean runtime relative error above bound: 49.23%
- Prediction miss count: 5
- Risk recall approximation: 14.29%
- Useful decision signals: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.92-0.95 | 0.9622 | no | under 5 | 3.62s | yes | accuracy 0.9622 outside predicted range 0.92-0.95 |
| run_002 | logistic_regression | accuracy around 0.96-0.97 | 0.9711 | no | under 5 | 0.69s | yes | accuracy 0.9711 outside predicted range 0.96-0.97 |
| run_003 | svc | accuracy around 0.96-0.97 | 0.9778 | no | under 5 | 0.11s | yes | accuracy 0.9778 outside predicted range 0.96-0.97 |
| run_004 | logistic_regression | accuracy around 0.97-0.98 | 0.9778 | yes | under 5 | 0.62s | yes |  |
| run_005 | logistic_regression | accuracy around 0.975-0.98 | 0.9644 | no | under 5 | 0.38s | yes | accuracy 0.9644 outside predicted range 0.97-0.98 |
| run_006 | top_model_verification | accuracy around 0.975-0.98 with small seed variance | 0.9764 | yes | under 10 | 44.46s | no | runtime 44.46s exceeded predicted 10s |
| run_007 | svc | accuracy around 0.975-0.98 | 0.9800 | yes | under 5 | 0.10s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | logistic regression without feature scaling may underperform on pixel intensity features; single-run best score may not justify a scientific claim | accuracy 0.9622 outside predicted range 0.92-0.95 | no |
| run_002 | single-run best score may not justify a scientific claim; feature scaling is relevant for logistic regression and should be verified across seeds | accuracy 0.9711 outside predicted range 0.96-0.97 | no |
| run_003 | SVC requires feature scaling; scaling is set to true; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win on a single split | accuracy 0.9778 outside predicted range 0.96-0.97 | no |
| run_004 | single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win; robustness sweeps required before strong claims | none | no |
| run_005 | single-run best score may not justify a scientific claim; label-noise sweeps test claim robustness, not leaderboard performance | accuracy 0.9644 outside predicted range 0.97-0.98 | no |
| run_006 | Top models are tied on single-split evidence; robust best-model claim requires multi-seed verification.; single-run best score may not justify a scientific claim | runtime 44.46s exceeded predicted 10s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win; robustness sweeps needed before strong claims | none | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=162.0; autoresearch agent preferred this catalog action; world model recommended modification, so the action remains informative but lower priority; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple unscaled linear baseline should anchor the search before interventions.. World model predicted accuracy around 0.92-0.95, runtime under 5, recommendation=modify, risks=logistic regression without feature scaling may underperform on pixel intensity features, single-run best score may not justify a scientific claim. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize scale-sensitive linear classification.. World model predicted accuracy around 0.96-0.97, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, feature scaling is relevant for logistic regression and should be verified across seeds. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=90.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.96-0.97, runtime under 5, recommendation=run, risks=SVC requires feature scaling; scaling is set to true, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win on a single split. Causal signal type: mixed. |
| run_004 | logistic_regression | yes | yes | Selected logistic_regression because score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win, robustness sweeps required before strong claims. Causal signal type: mixed. |
| run_005 | logistic_regression | yes | yes | Selected logistic_regression because score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.975-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, label-noise sweeps test claim robustness, not leaderboard performance. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=152.0; autoresearch agent preferred this catalog action; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.975-0.98 with small seed variance, runtime under 10, recommendation=run, risks=Top models are tied on single-split evidence; robust best-model claim requires multi-seed verification., single-run best score may not justify a scientific claim. Causal signal type: mixed. |
| run_007 | svc | yes | yes | Selected svc because score=60.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.975-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win, robustness sweeps needed before strong claims. Causal signal type: mixed. |