# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 28.57%
- Runtime interval coverage: 85.71%
- Mean metric absolute error outside interval: 0.0049
- Mean runtime relative error above bound: 20.40%
- Prediction miss count: 6
- Risk recall approximation: 14.29%
- Useful decision signals: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.93-0.95 | 0.9510 | no | under 2 | 0.46s | yes | accuracy 0.9510 outside predicted range 0.93-0.95 |
| run_002 | logistic_regression | accuracy around 0.95-0.97 | 0.9790 | no | under 5 | 0.04s | yes | accuracy 0.9790 outside predicted range 0.95-0.97 |
| run_003 | svc | accuracy around 0.96-0.98 | 0.9720 | yes | under 5 | 0.03s | yes |  |
| run_004 | random_forest | accuracy around 0.96-0.98 | 0.9580 | no | under 5 | 0.28s | yes | accuracy 0.9580 outside predicted range 0.96-0.98 |
| run_005 | gradient_boosting | accuracy around 0.96-0.98 | 0.9441 | no | under 5 | 0.41s | yes | accuracy 0.9441 outside predicted range 0.96-0.98 |
| run_006 | top_model_verification | accuracy around 0.97-0.98 for logistic_regression_scaled_C=0.1, 0.96-0.97 for svc_scaled_C=0.5_kernel=rbf, 0.95-0.96 for random_forest_n=100 | 0.9636 | yes | under 15 | 36.42s | no | runtime 36.42s exceeded predicted 15s |
| run_007 | logistic_regression | accuracy around 0.97-0.98 | 0.9860 | no | under 5 | 0.02s | yes | accuracy 0.9860 outside predicted range 0.97-0.98 |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | logistic regression without feature scaling may underperform on breast_cancer due to unscaled features; single-run score may overstate performance; repeated seeds needed for robust claims | accuracy 0.9510 outside predicted range 0.93-0.95 | no |
| run_002 | single-run best score may not justify a scientific claim; logistic regression requires feature scaling for comparable performance; unscaled baselines can mislead; small-dataset variance: breast_cancer is small, so a single split could overstate a model win | accuracy 0.9790 outside predicted range 0.95-0.97 | no |
| run_003 | small-dataset variance with a single split could overstate a model win; robustness sweeps needed before strong claims when a high single-run score exists | none | no |
| run_004 | small-dataset variance for a single split could overstate a model win; tree ensembles may show overfitting or variance on small datasets | accuracy 0.9580 outside predicted range 0.96-0.98 | no |
| run_005 | single-run best score may not justify a scientific claim; world-model predictions may miss quantitative details; tree ensembles can overfit on small datasets without proper depth control | accuracy 0.9441 outside predicted range 0.96-0.98 | no |
| run_006 | small-dataset variance may cause single-run best score to overstate a model win; top-model gap 0.0070 is within margin 0.0100; verify before claiming a winner; label-noise sweeps test claim robustness, not leaderboard performance | runtime 36.42s exceeded predicted 15s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | feature scaling is required for logistic regression and SVC; small-dataset variance may cause single-split results to overstate a model win; robustness sweeps are needed before strong claims when a high single-run score exists | accuracy 0.9860 outside predicted range 0.97-0.98 | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=162.0; autoresearch agent preferred this catalog action; world model recommended modification, so the action remains informative but lower priority; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple baseline should anchor the research loop before interventions.. World model predicted accuracy around 0.93-0.95, runtime under 2, recommendation=modify, risks=logistic regression without feature scaling may underperform on breast_cancer due to unscaled features, single-run score may overstate performance; repeated seeds needed for robust claims. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize a scale-sensitive linear model.. World model predicted accuracy around 0.95-0.97, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, logistic regression requires feature scaling for comparable performance; unscaled baselines can mislead, small-dataset variance: breast_cancer is small, so a single split could overstate a model win. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=90.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.96-0.98, runtime under 5, recommendation=run, risks=small-dataset variance with a single split could overstate a model win, robustness sweeps needed before strong claims when a high single-run score exists. Causal signal type: mixed. |
| run_004 | random_forest | yes | yes | Selected random_forest because score=86.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; world model predicted overfitting risk. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.96-0.98, runtime under 5, recommendation=run, risks=small-dataset variance for a single split could overstate a model win, tree ensembles may show overfitting or variance on small datasets. Causal signal type: mixed. |
| run_005 | gradient_boosting | yes | yes | Selected gradient_boosting because score=91.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.96-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, world-model predictions may miss quantitative details, tree ensembles can overfit on small datasets without proper depth control. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=152.0; autoresearch agent preferred this catalog action; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best observed single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.97-0.98 for logistic_regression_scaled_C=0.1, 0.96-0.97 for svc_scaled_C=0.5_kernel=rbf, 0.95-0.96 for random_forest_n=100, runtime under 15, recommendation=run, risks=small-dataset variance may cause single-run best score to overstate a model win, top-model gap 0.0070 is within margin 0.0100; verify before claiming a winner, label-noise sweeps test claim robustness, not leaderboard performance. Causal signal type: mixed. |
| run_007 | logistic_regression | yes | yes | Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=feature scaling is required for logistic regression and SVC, small-dataset variance may cause single-split results to overstate a model win, robustness sweeps are needed before strong claims when a high single-run score exists. Causal signal type: mixed. |