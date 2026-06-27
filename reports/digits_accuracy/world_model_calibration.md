# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 100.00%
- Runtime interval coverage: 85.71%
- Mean metric absolute error outside interval: 0.0000
- Mean runtime relative error above bound: 1.31%
- Prediction miss count: 1
- Risk recall approximation: 14.29%
- Useful decision signals: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.90-0.97 | 0.9622 | yes | under 5 | 2.08s | yes |  |
| run_002 | logistic_regression | accuracy around 0.95-0.98 | 0.9711 | yes | under 5 | 0.68s | yes |  |
| run_003 | svc | accuracy around 0.90-0.99 | 0.9778 | yes | under 10 | 0.16s | yes |  |
| run_004 | random_forest | accuracy around 0.90-0.98 | 0.9600 | yes | under 10 | 0.46s | yes |  |
| run_005 | hist_gradient_boosting | accuracy around 0.90-0.98 | 0.9578 | yes | under 15 | 3.90s | yes |  |
| run_006 | top_model_verification | accuracy around 0.90-0.99 | 0.9764 | yes | under 45 | 49.13s | no | runtime 49.13s exceeded predicted 45s |
| run_007 | logistic_regression | accuracy around 0.95-0.98 | 0.9778 | yes | under 5 | 0.31s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | unscaled features can slow convergence or underperform when feature scales differ | none | no |
| run_002 | minor convergence warning possible | none | no |
| run_003 | sensitive to scaling and C | none | no |
| run_004 | overfitting or split variance if depth is unconstrained | none | no |
| run_005 | can overfit with too many estimators | none | no |
| run_006 | top single-run models may be tied across seeds; seed variance may exceed the observed top-model gap | runtime 49.13s exceeded predicted 45s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | minor convergence warning possible | none | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=170.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple unscaled linear baseline should anchor the search before interventions.. World model predicted accuracy around 0.90-0.97, runtime under 5, recommendation=run, risks=unscaled features can slow convergence or underperform when feature scales differ. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize scale-sensitive linear classification.. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=90.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage. Agent hypothesis: Testing a new model family can reveal whether the current best score depends on inductive bias.. World model predicted accuracy around 0.90-0.99, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Causal signal type: mixed. |
| run_004 | random_forest | yes | yes | Selected random_forest because score=86.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; world model predicted overfitting risk. Agent hypothesis: Testing a new model family can reveal whether the current best score depends on inductive bias.. World model predicted accuracy around 0.90-0.98, runtime under 10, recommendation=run, risks=overfitting or split variance if depth is unconstrained. Causal signal type: mixed. |
| run_005 | hist_gradient_boosting | yes | yes | Selected hist_gradient_boosting because score=86.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; world model predicted overfitting risk. Agent hypothesis: Testing a new model family can reveal whether the current best score depends on inductive bias.. World model predicted accuracy around 0.90-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=152.0; autoresearch agent preferred this catalog action; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.90-0.99, runtime under 45, recommendation=run, risks=top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap. Causal signal type: mixed. |
| run_007 | logistic_regression | yes | yes | Selected logistic_regression because score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |