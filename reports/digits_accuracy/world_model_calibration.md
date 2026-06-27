# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 100.00%
- Runtime interval coverage: 85.71%
- Mean metric absolute error outside interval: 0.0000
- Mean runtime relative error above bound: 0.28%
- Prediction miss count: 1
- Risk recall approximation: 14.29%
- Useful decision signals: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.90-0.97 | 0.9622 | yes | under 5 | 3.41s | yes |  |
| run_002 | logistic_regression | accuracy around 0.95-0.98 | 0.9711 | yes | under 5 | 1.07s | yes |  |
| run_003 | logistic_regression | accuracy around 0.95-0.98 | 0.9778 | yes | under 5 | 0.81s | yes |  |
| run_004 | logistic_regression | accuracy around 0.95-0.98 | 0.9644 | yes | under 5 | 0.24s | yes |  |
| run_005 | svc | accuracy around 0.90-0.99 | 0.9778 | yes | under 10 | 0.13s | yes |  |
| run_006 | random_forest | accuracy around 0.90-0.98 | 0.9600 | yes | under 10 | 0.43s | yes |  |
| run_007 | top_model_verification | accuracy around 0.90-0.99 | 0.9764 | yes | under 45 | 45.90s | no | runtime 45.90s exceeded predicted 45s |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | unscaled features can slow convergence or underperform when feature scales differ | none | no |
| run_002 | minor convergence warning possible | none | no |
| run_003 | minor convergence warning possible | none | no |
| run_004 | minor convergence warning possible | none | no |
| run_005 | sensitive to scaling and C | none | no |
| run_006 | overfitting or split variance if depth is unconstrained | none | no |
| run_007 | top single-run models may be tied across seeds; seed variance may exceed the observed top-model gap | runtime 45.90s exceeded predicted 45s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=128.0; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.90-0.97, runtime under 5, recommendation=run, risks=unscaled features can slow convergence or underperform when feature scales differ. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |
| run_003 | logistic_regression | yes | yes | Selected logistic_regression because score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |
| run_004 | logistic_regression | yes | yes | Selected logistic_regression because score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |
| run_005 | svc | yes | yes | Selected svc because score=48.0; world model recommended run; new model family increases search coverage. World model predicted accuracy around 0.90-0.99, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Causal signal type: mixed. |
| run_006 | random_forest | yes | yes | Selected random_forest because score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk. World model predicted accuracy around 0.90-0.98, runtime under 10, recommendation=run, risks=overfitting or split variance if depth is unconstrained. Causal signal type: mixed. |
| run_007 | top_model_verification | yes | yes | Selected top_model_verification because score=110.0; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. World model predicted accuracy around 0.90-0.99, runtime under 45, recommendation=run, risks=top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap. Causal signal type: mixed. |