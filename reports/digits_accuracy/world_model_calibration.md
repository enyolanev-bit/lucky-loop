# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 83.33%
- Runtime interval coverage: 100.00%
- Mean metric absolute error outside interval: 0.0001
- Mean runtime relative error above bound: 0.00%
- Prediction miss count: 1
- Risk recall approximation: 16.67%
- Useful decision signals: 5/6

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.90-0.97 | 0.9622 | yes | under 5 | 2.83s | yes |  |
| run_002 | logistic_regression | accuracy around 0.95-0.98 | 0.9778 | yes | under 5 | 0.47s | yes |  |
| run_003 | svc | accuracy around 0.90-0.99 | 0.9800 | yes | under 10 | 0.17s | yes |  |
| run_004 | random_forest | accuracy around 0.90-0.98 | 0.9689 | yes | under 10 | 1.43s | yes |  |
| run_005 | verification_sweep | accuracy around 0.85-0.98 | 0.9806 | no | under 35 | 30.38s | yes | accuracy 0.9806 outside predicted range 0.85-0.98 |
| run_006 | hist_gradient_boosting | accuracy around 0.90-0.98 | 0.9600 | yes | under 15 | 5.46s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | unscaled features can slow convergence or underperform when feature scales differ | none | no |
| run_002 | minor convergence warning possible | none | no |
| run_003 | sensitive to scaling and C | none | no |
| run_004 | overfitting or split variance if depth is unconstrained | none | no |
| run_005 | label noise can make small hyperparameter differences non-robust; seed variance may exceed apparent gains | accuracy 0.9806 outside predicted range 0.85-0.98; Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery. | yes |
| run_006 | can overfit with too many estimators | none | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=110; world model recommended run; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.90-0.97, runtime under 5, recommendation=run, risks=unscaled features can slow convergence or underperform when feature scales differ. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=75; world model recommended run; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=30; world model recommended run. World model predicted accuracy around 0.90-0.99, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Causal signal type: world_model_prediction. |
| run_004 | random_forest | yes | yes | Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.90-0.98, runtime under 10, recommendation=run, risks=overfitting or split variance if depth is unconstrained. Causal signal type: mixed. |
| run_005 | verification_sweep | yes | yes | Selected verification_sweep because score=90; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk. World model predicted accuracy around 0.85-0.98, runtime under 35, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. Causal signal type: mixed. |
| run_006 | hist_gradient_boosting | yes | no | Selected hist_gradient_boosting because score=26; world model recommended run; world model predicted overfitting risk. World model predicted accuracy around 0.90-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. Causal signal type: world_model_prediction. |