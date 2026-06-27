# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 42.86%
- Runtime interval coverage: 71.43%
- Mean metric absolute error outside interval: 0.0189
- Mean runtime relative error above bound: 75.38%
- Prediction miss count: 6
- Risk recall approximation: 28.57%
- Useful decision signals: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.85-0.90 | 0.9778 | no | under 5 | 0.61s | yes | accuracy 0.9778 outside predicted range 0.85-0.90 |
| run_002 | logistic_regression | accuracy around 0.97-0.98 | 1.0000 | no | under 5 | 0.08s | yes | accuracy 1.0000 outside predicted range 0.97-0.98 |
| run_003 | svc | accuracy around 0.95-0.98 | 0.9778 | yes | under 5 | 0.02s | yes |  |
| run_004 | gradient_boosting | accuracy around 0.97-0.98 | 0.9556 | no | under 5 | 0.40s | yes | accuracy 0.9556 outside predicted range 0.97-0.98 |
| run_005 | random_forest | accuracy around 0.97-0.98 | 1.0000 | no | under 5 | 0.30s | yes | accuracy 1.0000 outside predicted range 0.97-0.98 |
| run_006 | top_model_verification | accuracy around 0.97-1.0 for both logistic_regression_scaled_C=0.1 and random_forest_n=100 across seeds | 0.9956 | yes | under 10 | 44.24s | no | runtime 44.24s exceeded predicted 10s |
| run_007 | verification_sweep | accuracy around 0.97-0.99 across C values, with C=0.1 and C=1.0 likely near 1.0 but showing seed variance | 0.9667 | yes | under 10 | 28.53s | no | runtime 28.53s exceeded predicted 10s |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | feature scaling is relevant for logistic regression and not enabled in this run; single-run best score may not justify a scientific claim; small-dataset variance on a single split could overstate a model win | accuracy 0.9778 outside predicted range 0.85-0.90 | no |
| run_002 | feature scaling is relevant for logistic regression; small-dataset variance may cause single-run scores to overstate a model win; robustness sweeps needed before strong claims | accuracy 1.0000 outside predicted range 0.97-0.98 | no |
| run_003 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; feature scaling is required for SVC and is applied | none | no |
| run_004 | small-dataset variance for a single split could overstate a model win; gradient boosting may overfit on small tabular datasets if not carefully tuned; robustness sweeps across seeds needed before claiming superiority over logistic regression or SVC | accuracy 0.9556 outside predicted range 0.97-0.98 | no |
| run_005 | small-dataset variance may cause a single split to overstate a model win; tree ensembles can overfit on small tabular datasets if depth is unbounded; robustness sweeps needed before strong claims when a high single-run score exists | accuracy 1.0000 outside predicted range 0.97-0.98 | no |
| run_006 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; Top models are tied on single-split evidence; robust best-model claim requires multi-seed verification. | runtime 44.24s exceeded predicted 10s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; label-noise sweeps test claim robustness, not leaderboard performance | runtime 28.53s exceeded predicted 10s; Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery. | yes |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=162.0; autoresearch agent preferred this catalog action; world model recommended modification, so the action remains informative but lower priority; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple unscaled linear baseline should anchor the search before interventions.. World model predicted accuracy around 0.85-0.90, runtime under 5, recommendation=modify, risks=feature scaling is relevant for logistic regression and not enabled in this run, single-run best score may not justify a scientific claim, small-dataset variance on a single split could overstate a model win. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize scale-sensitive linear classification.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=feature scaling is relevant for logistic regression, small-dataset variance may cause single-run scores to overstate a model win, robustness sweeps needed before strong claims. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=82.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, feature scaling is required for SVC and is applied. Causal signal type: mixed. |
| run_004 | gradient_boosting | yes | yes | Selected gradient_boosting because score=83.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=small-dataset variance for a single split could overstate a model win, gradient boosting may overfit on small tabular datasets if not carefully tuned, robustness sweeps across seeds needed before claiming superiority over logistic regression or SVC. Causal signal type: mixed. |
| run_005 | random_forest | yes | yes | Selected random_forest because score=79.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; new model family increases search coverage; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=small-dataset variance may cause a single split to overstate a model win, tree ensembles can overfit on small tabular datasets if depth is unbounded, robustness sweeps needed before strong claims when a high single-run score exists. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=152.0; autoresearch agent preferred this catalog action; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.97-1.0 for both logistic_regression_scaled_C=0.1 and random_forest_n=100 across seeds, runtime under 10, recommendation=run, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, Top models are tied on single-split evidence; robust best-model claim requires multi-seed verification.. Causal signal type: mixed. |
| run_007 | verification_sweep | yes | yes | Selected verification_sweep because score=90.0; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-0.99 across C values, with C=0.1 and C=1.0 likely near 1.0 but showing seed variance, runtime under 10, recommendation=run, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, label-noise sweeps test claim robustness, not leaderboard performance. Causal signal type: mixed. |