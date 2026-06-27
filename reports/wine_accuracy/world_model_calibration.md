# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 42.86%
- Runtime interval coverage: 100.00%
- Mean metric absolute error outside interval: 0.0076
- Mean runtime relative error above bound: 0.00%
- Prediction miss count: 4
- Risk recall approximation: 28.57%
- Useful decision signals: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.90-0.97 | 0.9778 | no | under 5 | 0.48s | yes | accuracy 0.9778 outside predicted range 0.90-0.97 |
| run_002 | logistic_regression | accuracy around 0.95-0.98 | 1.0000 | no | under 5 | 0.01s | yes | accuracy 1.0000 outside predicted range 0.95-0.98 |
| run_003 | random_forest | accuracy around 0.90-0.98 | 1.0000 | no | under 10 | 0.73s | yes | accuracy 1.0000 outside predicted range 0.90-0.98 |
| run_004 | svc | accuracy around 0.90-0.99 | 0.9778 | yes | under 10 | 0.01s | yes |  |
| run_005 | gradient_boosting | accuracy around 0.90-0.98 | 0.9556 | yes | under 15 | 0.51s | yes |  |
| run_006 | top_model_verification | accuracy around 0.90-0.99 | 0.9956 | no | under 45 | 31.66s | yes | accuracy 0.9956 outside predicted range 0.90-0.99 |
| run_007 | verification_sweep | accuracy around 0.85-0.98 | 0.9667 | yes | under 35 | 21.24s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | unscaled features can slow convergence or underperform when feature scales differ | accuracy 0.9778 outside predicted range 0.90-0.97 | no |
| run_002 | minor convergence warning possible | accuracy 1.0000 outside predicted range 0.95-0.98 | no |
| run_003 | overfitting or split variance if depth is unconstrained | accuracy 1.0000 outside predicted range 0.90-0.98 | no |
| run_004 | sensitive to scaling and C | none | no |
| run_005 | can overfit with too many estimators | none | no |
| run_006 | top single-run models may be tied across seeds; seed variance may exceed the observed top-model gap | accuracy 0.9956 outside predicted range 0.90-0.99; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | label noise can make small hyperparameter differences non-robust; seed variance may exceed apparent gains | Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery. | yes |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=110; world model recommended run; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.90-0.97, runtime under 5, recommendation=run, risks=unscaled features can slow convergence or underperform when feature scales differ. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=75; world model recommended run; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |
| run_003 | random_forest | yes | yes | Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.90-0.98, runtime under 10, recommendation=run, risks=overfitting or split variance if depth is unconstrained. Causal signal type: mixed. |
| run_004 | svc | yes | yes | Selected svc because score=60; world model recommended run; tree prediction missed, so a scaled margin model tests a different hypothesis. World model predicted accuracy around 0.90-0.99, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Causal signal type: mixed. |
| run_005 | gradient_boosting | yes | yes | Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.90-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=110; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. World model predicted accuracy around 0.90-0.99, runtime under 45, recommendation=run, risks=top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap. Causal signal type: mixed. |
| run_007 | verification_sweep | yes | yes | Selected verification_sweep because score=90; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk. World model predicted accuracy around 0.85-0.98, runtime under 35, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. Causal signal type: mixed. |