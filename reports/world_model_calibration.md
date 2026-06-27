# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 80.00%
- Runtime interval coverage: 90.00%
- Mean metric absolute error outside interval: 0.0012
- Mean runtime relative error above bound: 1.36%
- Prediction miss count: 3
- Risk recall approximation: 50.00%
- Useful decision signals: 10/10

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.92-0.96 | 0.9510 | yes | under 5 | 0.52s | yes |  |
| run_002 | logistic_regression | accuracy around 0.95-0.98 | 0.9860 | no | under 5 | 0.04s | yes | accuracy 0.9860 outside predicted range 0.95-0.98 |
| run_003 | random_forest | accuracy around 0.94-0.98 | 0.9580 | yes | under 10 | 0.82s | yes |  |
| run_004 | verification_sweep | accuracy around 0.94-0.98 | 0.9615 | yes | under 25 | 28.40s | no | runtime 28.40s exceeded predicted 25s |
| run_005 | real_effect | accuracy around 0.76-0.88 | 0.8675 | yes | under 5 | 0.00s | yes |  |
| run_006 | weak_effect | accuracy around 0.94-0.99 | 0.9750 | yes | under 5 | 0.00s | yes |  |
| run_007 | data_leakage_trap | accuracy around 0.95-1.00 | 0.9992 | yes | under 5 | 0.00s | yes |  |
| run_008 | metric_misuse | balanced_accuracy around 0.50-0.73 | 0.7175 | yes | under 5 | 0.00s | yes |  |
| run_009 | gradient_boosting | accuracy around 0.94-0.98 | 0.9510 | yes | under 15 | 0.64s | yes |  |
| run_010 | svc | accuracy around 0.94-0.98 | 0.9860 | no | under 10 | 0.02s | yes | accuracy 0.9860 outside predicted range 0.94-0.98 |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | unscaled features can slow convergence | none | no |
| run_002 | minor convergence warning possible | accuracy 0.9860 outside predicted range 0.95-0.98 | no |
| run_003 | overfitting if max_depth is unconstrained | none | no |
| run_004 | label noise can make small hyperparameter differences non-robust; seed variance may exceed apparent gains | runtime 28.40s exceeded predicted 25s; Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery. | yes |
| run_005 | large effect should be checked against seed noise | none | yes |
| run_006 | effect may be smaller than seed noise; claim should likely be weak or blocked | Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery. | yes |
| run_007 | suspiciously high accuracy may indicate data leakage; protocol warning should block a strong claim | Protocol warning blocks the claim even though the metric effect may look large: suspiciously high score; label-derived feature was included before the split | yes |
| run_008 | accuracy may be misleading under class imbalance; balanced accuracy or F1 should be preferred | Protocol warning blocks the claim even though the metric effect may look large: accuracy is misleading on an imbalanced dataset; balanced_accuracy is the verifier metric | yes |
| run_009 | can overfit with too many estimators | none | no |
| run_010 | sensitive to scaling and C | accuracy 0.9860 outside predicted range 0.94-0.98 | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=110; world model recommended run; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.92-0.96, runtime under 5, recommendation=run, risks=unscaled features can slow convergence. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=75; world model recommended run; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. |
| run_003 | random_forest | yes | yes | Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=overfitting if max_depth is unconstrained. |
| run_004 | verification_sweep | yes | yes | Selected verification_sweep because score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist. World model predicted accuracy around 0.94-0.98, runtime under 25, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. |
| run_005 | real_effect | yes | yes | Selected real_effect because score=54; world model recommended run; real-effect scenario should produce a supported trust-ladder case. World model predicted accuracy around 0.76-0.88, runtime under 5, recommendation=run, risks=large effect should be checked against seed noise. |
| run_006 | weak_effect | yes | yes | Selected weak_effect because score=48; world model recommended run; weak-effect scenario demonstrates the lower rung of the trust ladder. World model predicted accuracy around 0.94-0.99, runtime under 5, recommendation=run, risks=effect may be smaller than seed noise, claim should likely be weak or blocked. |
| run_007 | data_leakage_trap | yes | yes | Selected data_leakage_trap because score=52; world model recommended run; data leakage trap tests whether protocol warnings block suspicious wins. World model predicted accuracy around 0.95-1.00, runtime under 5, recommendation=run, risks=suspiciously high accuracy may indicate data leakage, protocol warning should block a strong claim. |
| run_008 | metric_misuse | yes | yes | Selected metric_misuse because score=50; world model recommended run; metric misuse scenario tests whether the system avoids accuracy-only claims. World model predicted balanced_accuracy around 0.50-0.73, runtime under 5, recommendation=run, risks=accuracy may be misleading under class imbalance, balanced accuracy or F1 should be preferred. |
| run_009 | gradient_boosting | yes | yes | Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. |
| run_010 | svc | yes | yes | Selected svc because score=30; world model recommended run. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=sensitive to scaling and C. |