# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 28.57%
- Runtime interval coverage: 85.71%
- Mean metric absolute error outside interval: 0.0049
- Mean runtime relative error above bound: 21.43%
- Prediction miss count: 6
- Risk recall approximation: 14.29%
- Useful decision signals: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.93-0.95 | 0.9510 | no | under 5 | 0.62s | yes | accuracy 0.9510 outside predicted range 0.93-0.95 |
| run_002 | logistic_regression | accuracy around 0.95-0.97 | 0.9790 | no | under 5 | 0.05s | yes | accuracy 0.9790 outside predicted range 0.95-0.97 |
| run_003 | svc | accuracy around 0.96-0.98 | 0.9720 | yes | under 5 | 0.02s | yes |  |
| run_004 | gradient_boosting | accuracy around 0.96-0.98 | 0.9441 | no | under 5 | 0.39s | yes | accuracy 0.9441 outside predicted range 0.96-0.98 |
| run_005 | logistic_regression | accuracy around 0.97-0.98 | 0.9860 | no | under 5 | 0.05s | yes | accuracy 0.9860 outside predicted range 0.97-0.98 |
| run_006 | top_model_verification | accuracy around 0.97-0.99 with seed variance of ~0.01-0.02 | 0.9706 | yes | under 15 | 37.50s | no | runtime 37.50s exceeded predicted 15s |
| run_007 | random_forest | accuracy around 0.96-0.98 | 0.9580 | no | under 5 | 0.31s | yes | accuracy 0.9580 outside predicted range 0.96-0.98 |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | logistic regression without feature scaling may underperform on breast_cancer due to unscaled features; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win with a single split | accuracy 0.9510 outside predicted range 0.93-0.95 | no |
| run_002 | single-run best score may not justify a scientific claim; logistic regression requires feature scaling for meaningful coefficient interpretation and performance; small-dataset variance may cause a single split to overstate a model win | accuracy 0.9790 outside predicted range 0.95-0.97 | no |
| run_003 | single-run best score may not justify a scientific claim; SVC with RBF kernel can overfit on small tabular datasets if C is too high; feature scaling is required for SVC and is applied, but variance across seeds should be checked | none | no |
| run_004 | single-run best score may not justify a scientific claim; gradient boosting may overfit or show variance on small splits; robustness sweeps needed before strong claims | accuracy 0.9441 outside predicted range 0.96-0.98 | no |
| run_005 | single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win; label-noise sweeps test claim robustness, not leaderboard performance | accuracy 0.9860 outside predicted range 0.97-0.98 | no |
| run_006 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; Top-model gap 0.0070 is within margin 0.0100; verify before claiming a winner. | runtime 37.50s exceeded predicted 15s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; world-model predictions may miss quantitative details | accuracy 0.9580 outside predicted range 0.96-0.98 | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=162.0; autoresearch agent preferred this catalog action; world model recommended modification, so the action remains informative but lower priority; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple unscaled linear baseline should anchor the search before interventions.. World model predicted accuracy around 0.93-0.95, runtime under 5, recommendation=modify, risks=logistic regression without feature scaling may underperform on breast_cancer due to unscaled features, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win with a single split. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize scale-sensitive linear classification.. World model predicted accuracy around 0.95-0.97, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, logistic regression requires feature scaling for meaningful coefficient interpretation and performance, small-dataset variance may cause a single split to overstate a model win. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=86.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; world model predicted overfitting risk. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.96-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, SVC with RBF kernel can overfit on small tabular datasets if C is too high, feature scaling is required for SVC and is applied, but variance across seeds should be checked. Causal signal type: mixed. |
| run_004 | gradient_boosting | yes | yes | Selected gradient_boosting because score=91.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.96-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, gradient boosting may overfit or show variance on small splits, robustness sweeps needed before strong claims. Causal signal type: mixed. |
| run_005 | logistic_regression | yes | yes | Selected logistic_regression because score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win, label-noise sweeps test claim robustness, not leaderboard performance. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=152.0; autoresearch agent preferred this catalog action; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.97-0.99 with seed variance of ~0.01-0.02, runtime under 15, recommendation=run, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, Top-model gap 0.0070 is within margin 0.0100; verify before claiming a winner.. Causal signal type: mixed. |
| run_007 | random_forest | yes | yes | Selected random_forest because score=83.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; new model family increases search coverage; strong linear baseline makes a different inductive bias useful; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.96-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, world-model predictions may miss quantitative details. Causal signal type: mixed. |