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
| run_001 | logistic_regression | accuracy around 0.90-0.97 | 0.9778 | no | under 5 | 1.32s | yes | accuracy 0.9778 outside predicted range 0.90-0.97 |
| run_002 | logistic_regression | accuracy around 0.95-0.98 | 1.0000 | no | under 5 | 0.02s | yes | accuracy 1.0000 outside predicted range 0.95-0.98 |
| run_003 | svc | accuracy around 0.90-0.99 | 0.9778 | yes | under 10 | 0.01s | yes |  |
| run_004 | gradient_boosting | accuracy around 0.90-0.98 | 0.9556 | yes | under 15 | 0.32s | yes |  |
| run_005 | random_forest | accuracy around 0.90-0.98 | 1.0000 | no | under 10 | 0.31s | yes | accuracy 1.0000 outside predicted range 0.90-0.98 |
| run_006 | top_model_verification | accuracy around 0.90-0.99 | 0.9956 | no | under 45 | 31.14s | yes | accuracy 0.9956 outside predicted range 0.90-0.99 |
| run_007 | verification_sweep | accuracy around 0.85-0.98 | 0.9667 | yes | under 35 | 24.07s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | unscaled features can slow convergence or underperform when feature scales differ | accuracy 0.9778 outside predicted range 0.90-0.97 | no |
| run_002 | minor convergence warning possible | accuracy 1.0000 outside predicted range 0.95-0.98 | no |
| run_003 | sensitive to scaling and C | none | no |
| run_004 | can overfit with too many estimators | none | no |
| run_005 | overfitting or split variance if depth is unconstrained | accuracy 1.0000 outside predicted range 0.90-0.98 | no |
| run_006 | top single-run models may be tied across seeds; seed variance may exceed the observed top-model gap | accuracy 0.9956 outside predicted range 0.90-0.99; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | label noise can make small hyperparameter differences non-robust; seed variance may exceed apparent gains | Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery. | yes |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=170.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple unscaled linear baseline should anchor the search before interventions.. World model predicted accuracy around 0.90-0.97, runtime under 5, recommendation=run, risks=unscaled features can slow convergence or underperform when feature scales differ. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize scale-sensitive linear classification.. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=82.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.90-0.99, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Causal signal type: mixed. |
| run_004 | gradient_boosting | yes | yes | Selected gradient_boosting because score=83.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.90-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. Causal signal type: mixed. |
| run_005 | random_forest | yes | yes | Selected random_forest because score=79.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; new model family increases search coverage; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.90-0.98, runtime under 10, recommendation=run, risks=overfitting or split variance if depth is unconstrained. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=152.0; autoresearch agent preferred this catalog action; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.90-0.99, runtime under 45, recommendation=run, risks=top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap. Causal signal type: mixed. |
| run_007 | verification_sweep | yes | yes | Selected verification_sweep because score=90.0; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.85-0.98, runtime under 35, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. Causal signal type: mixed. |