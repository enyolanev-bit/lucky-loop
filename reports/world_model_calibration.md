# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 66.67%
- Runtime interval coverage: 100.00%
- Mean metric absolute error outside interval: 0.0020
- Mean runtime relative error above bound: 0.00%
- Prediction miss count: 2
- Risk recall approximation: 16.67%
- Useful decision signals: 6/6

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.92-0.96 | 0.9510 | yes | under 5 | 0.35s | yes |  |
| run_002 | logistic_regression | accuracy around 0.95-0.98 | 0.9860 | no | under 5 | 0.02s | yes | accuracy 0.9860 outside predicted range 0.95-0.98 |
| run_003 | random_forest | accuracy around 0.94-0.98 | 0.9580 | yes | under 10 | 0.71s | yes |  |
| run_004 | verification_sweep | accuracy around 0.94-0.98 | 0.9615 | yes | under 25 | 24.16s | yes |  |
| run_005 | gradient_boosting | accuracy around 0.94-0.98 | 0.9510 | yes | under 15 | 0.58s | yes |  |
| run_006 | svc | accuracy around 0.94-0.98 | 0.9860 | no | under 10 | 0.02s | yes | accuracy 0.9860 outside predicted range 0.94-0.98 |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | unscaled features can slow convergence | none | no |
| run_002 | minor convergence warning possible | accuracy 0.9860 outside predicted range 0.95-0.98 | no |
| run_003 | overfitting if max_depth is unconstrained | none | no |
| run_004 | label noise can make small hyperparameter differences non-robust; seed variance may exceed apparent gains | Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery. | yes |
| run_005 | can overfit with too many estimators | none | no |
| run_006 | sensitive to scaling and C | accuracy 0.9860 outside predicted range 0.94-0.98 | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=110; world model recommended run; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.92-0.96, runtime under 5, recommendation=run, risks=unscaled features can slow convergence. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=75; world model recommended run; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. |
| run_003 | random_forest | yes | yes | Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=overfitting if max_depth is unconstrained. |
| run_004 | verification_sweep | yes | yes | Selected verification_sweep because score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist. World model predicted accuracy around 0.94-0.98, runtime under 25, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. |
| run_005 | gradient_boosting | yes | yes | Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. |
| run_006 | svc | yes | yes | Selected svc because score=30; world model recommended run. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=sensitive to scaling and C. |