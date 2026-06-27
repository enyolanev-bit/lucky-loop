# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 42.86%
- Runtime interval coverage: 85.71%
- Mean metric absolute error outside interval: 0.0051
- Mean runtime relative error above bound: 61.48%
- Prediction miss count: 5
- Risk recall approximation: 14.29%
- Useful decision signals: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.92-0.95 | 0.9622 | no | under 5 | 3.10s | yes | accuracy 0.9622 outside predicted range 0.92-0.95 |
| run_002 | logistic_regression | accuracy around 0.96-0.97 | 0.9711 | no | under 5 | 0.54s | yes | accuracy 0.9711 outside predicted range 0.96-0.97 |
| run_003 | svc | accuracy around 0.96-0.98 | 0.9778 | yes | under 5 | 0.12s | yes |  |
| run_004 | random_forest | accuracy around 0.97-0.98 | 0.9600 | no | under 5 | 0.47s | yes | accuracy 0.9600 outside predicted range 0.97-0.98 |
| run_005 | hist_gradient_boosting | accuracy around 0.97-0.98 | 0.9578 | no | under 5 | 4.94s | yes | accuracy 0.9578 outside predicted range 0.97-0.98 |
| run_006 | top_model_verification | accuracy around 0.97-0.98 with small seed variance | 0.9764 | yes | under 10 | 53.04s | no | runtime 53.04s exceeded predicted 10s |
| run_007 | logistic_regression | accuracy around 0.97-0.98 | 0.9778 | yes | under 5 | 0.67s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | logistic regression without feature scaling may underperform on pixel features; single-run best score may not justify a scientific claim | accuracy 0.9622 outside predicted range 0.92-0.95 | no |
| run_002 | single-run best score may not justify a scientific claim; feature scaling is relevant for logistic regression and should be verified across seeds | accuracy 0.9711 outside predicted range 0.96-0.97 | no |
| run_003 | single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win; SVC performance is sensitive to scaling and C; robustness sweeps needed before strong claims | none | no |
| run_004 | small-dataset variance for a single split could overstate a model win; tree ensembles may show overfitting or variance on small splits | accuracy 0.9600 outside predicted range 0.97-0.98 | no |
| run_005 | single-run best score may not justify a scientific claim; hist_gradient_boosting may overfit or show variance on small splits; robustness sweeps needed before strong claims | accuracy 0.9578 outside predicted range 0.97-0.98 | no |
| run_006 | single-run best score may not justify a scientific claim; Top-model gap 0.0067 is within margin 0.0100; verify before claiming a winner.; small-dataset variance could overstate a model win across seeds | runtime 53.04s exceeded predicted 10s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | feature scaling is required for logistic regression to achieve competitive accuracy; small-dataset variance means a single split could overstate a model win; robustness sweeps are needed before strong claims when a high single-run score exists | none | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=162.0; autoresearch agent preferred this catalog action; world model recommended modification, so the action remains informative but lower priority; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple baseline should anchor the research loop before interventions.. World model predicted accuracy around 0.92-0.95, runtime under 5, recommendation=modify, risks=logistic regression without feature scaling may underperform on pixel features, single-run best score may not justify a scientific claim. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize a scale-sensitive linear model.. World model predicted accuracy around 0.96-0.97, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, feature scaling is relevant for logistic regression and should be verified across seeds. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=90.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.96-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win, SVC performance is sensitive to scaling and C; robustness sweeps needed before strong claims. Causal signal type: mixed. |
| run_004 | random_forest | yes | yes | Selected random_forest because score=86.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; world model predicted overfitting risk. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=small-dataset variance for a single split could overstate a model win, tree ensembles may show overfitting or variance on small splits. Causal signal type: mixed. |
| run_005 | hist_gradient_boosting | yes | yes | Selected hist_gradient_boosting because score=86.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; world model predicted overfitting risk. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, hist_gradient_boosting may overfit or show variance on small splits, robustness sweeps needed before strong claims. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=152.0; autoresearch agent preferred this catalog action; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best observed single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.97-0.98 with small seed variance, runtime under 10, recommendation=run, risks=single-run best score may not justify a scientific claim, Top-model gap 0.0067 is within margin 0.0100; verify before claiming a winner., small-dataset variance could overstate a model win across seeds. Causal signal type: mixed. |
| run_007 | logistic_regression | yes | yes | Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=feature scaling is required for logistic regression to achieve competitive accuracy, small-dataset variance means a single split could overstate a model win, robustness sweeps are needed before strong claims when a high single-run score exists. Causal signal type: mixed. |