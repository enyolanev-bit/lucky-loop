# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 85.71%
- Runtime interval coverage: 100.00%
- Mean metric absolute error outside interval: 0.0009
- Mean runtime relative error above bound: 0.00%
- Prediction miss count: 1
- Risk recall approximation: 28.57%
- Useful decision signals: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.90-0.97 | 0.9510 | yes | under 5 | 0.57s | yes |  |
| run_002 | logistic_regression | accuracy around 0.95-0.98 | 0.9790 | yes | under 5 | 0.02s | yes |  |
| run_003 | logistic_regression | accuracy around 0.95-0.98 | 0.9860 | no | under 5 | 0.03s | yes | accuracy 0.9860 outside predicted range 0.95-0.98 |
| run_004 | random_forest | accuracy around 0.90-0.98 | 0.9580 | yes | under 10 | 0.32s | yes |  |
| run_005 | top_model_verification | accuracy around 0.90-0.99 | 0.9706 | yes | under 45 | 40.45s | yes |  |
| run_006 | verification_sweep | accuracy around 0.85-0.98 | 0.9615 | yes | under 35 | 31.22s | yes |  |
| run_007 | logistic_regression | accuracy around 0.95-0.98 | 0.9720 | yes | under 5 | 0.04s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | unscaled features can slow convergence or underperform when feature scales differ | none | no |
| run_002 | minor convergence warning possible | none | no |
| run_003 | minor convergence warning possible | accuracy 0.9860 outside predicted range 0.95-0.98 | no |
| run_004 | overfitting or split variance if depth is unconstrained | none | no |
| run_005 | top single-run models may be tied across seeds; seed variance may exceed the observed top-model gap | Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_006 | label noise can make small hyperparameter differences non-robust; seed variance may exceed apparent gains | Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery. | yes |
| run_007 | minor convergence warning possible | none | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=128.0; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.90-0.97, runtime under 5, recommendation=run, risks=unscaled features can slow convergence or underperform when feature scales differ. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |
| run_003 | logistic_regression | yes | yes | Selected logistic_regression because score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |
| run_004 | random_forest | yes | yes | Selected random_forest because score=71.0; world model recommended run; new model family increases search coverage; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk; high best score increases claim risk for additional single-run score chasing. World model predicted accuracy around 0.90-0.98, runtime under 10, recommendation=run, risks=overfitting or split variance if depth is unconstrained. Causal signal type: mixed. |
| run_005 | top_model_verification | yes | yes | Selected top_model_verification because score=110.0; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. World model predicted accuracy around 0.90-0.99, runtime under 45, recommendation=run, risks=top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap. Causal signal type: mixed. |
| run_006 | verification_sweep | yes | yes | Selected verification_sweep because score=90.0; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk. World model predicted accuracy around 0.85-0.98, runtime under 35, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. Causal signal type: mixed. |
| run_007 | logistic_regression | yes | yes | Selected logistic_regression because score=55.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; high best score increases claim risk for additional single-run score chasing. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed. |