# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 42.86%
- Runtime interval coverage: 85.71%
- Mean metric absolute error outside interval: 0.0175
- Mean runtime relative error above bound: 42.59%
- Prediction miss count: 5
- Risk recall approximation: 14.29%
- Useful decision signals: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.85-0.90 | 0.9778 | no | under 5 | 0.47s | yes | accuracy 0.9778 outside predicted range 0.85-0.90 |
| run_002 | logistic_regression | accuracy around 0.97-0.98 | 1.0000 | no | under 5 | 0.02s | yes | accuracy 1.0000 outside predicted range 0.97-0.98 |
| run_003 | svc | accuracy around 0.96-0.98 | 0.9778 | yes | under 5 | 0.01s | yes |  |
| run_004 | random_forest | accuracy around 0.96-0.98 | 1.0000 | no | under 5 | 0.37s | yes | accuracy 1.0000 outside predicted range 0.96-0.98 |
| run_005 | gradient_boosting | accuracy around 0.96-0.98 | 0.9556 | no | under 5 | 0.43s | yes | accuracy 0.9556 outside predicted range 0.96-0.98 |
| run_006 | top_model_verification | accuracy around 0.97-1.0 for both logistic_regression_scaled_C=0.1 and random_forest_n=100 across seeds | 0.9956 | yes | under 10 | 39.82s | no | runtime 39.82s exceeded predicted 10s |
| run_007 | logistic_regression | accuracy around 0.97-1.0 | 1.0000 | yes | under 5 | 0.02s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | feature scaling is relevant for logistic regression and not applied; single-run best score may not justify a scientific claim | accuracy 0.9778 outside predicted range 0.85-0.90 | no |
| run_002 | feature scaling matters for logistic regression; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win | accuracy 1.0000 outside predicted range 0.97-0.98 | no |
| run_003 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; feature scaling is required for SVC and is included, but small-dataset variance could still overstate a model win | none | no |
| run_004 | small-dataset variance may overstate a model win on a single split; tree ensembles can overfit or show variance on small tabular datasets | accuracy 1.0000 outside predicted range 0.96-0.98 | no |
| run_005 | small-dataset variance may cause single-run results to overstate model win; gradient boosting may overfit on small tabular datasets if not carefully tuned; robustness sweeps needed before strong claims when high single-run score exists | accuracy 0.9556 outside predicted range 0.96-0.98 | no |
| run_006 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; Top models are tied on single-split evidence; robust best-model claim requires multi-seed verification. | runtime 39.82s exceeded predicted 10s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; feature scaling is relevant for logistic regression | none | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=162.0; autoresearch agent preferred this catalog action; world model recommended modification, so the action remains informative but lower priority; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple baseline should anchor the research loop before interventions.. World model predicted accuracy around 0.85-0.90, runtime under 5, recommendation=modify, risks=feature scaling is relevant for logistic regression and not applied, single-run best score may not justify a scientific claim. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize a scale-sensitive linear model.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=feature scaling matters for logistic regression, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=82.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.96-0.98, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, feature scaling is required for SVC and is included, but small-dataset variance could still overstate a model win. Causal signal type: mixed. |
| run_004 | random_forest | yes | yes | Selected random_forest because score=113.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.96-0.98, runtime under 5, recommendation=run, risks=small-dataset variance may overstate a model win on a single split, tree ensembles can overfit or show variance on small tabular datasets. Causal signal type: mixed. |
| run_005 | gradient_boosting | yes | yes | Selected gradient_boosting because score=83.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.96-0.98, runtime under 5, recommendation=run, risks=small-dataset variance may cause single-run results to overstate model win, gradient boosting may overfit on small tabular datasets if not carefully tuned, robustness sweeps needed before strong claims when high single-run score exists. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=152.0; autoresearch agent preferred this catalog action; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best observed single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.97-1.0 for both logistic_regression_scaled_C=0.1 and random_forest_n=100 across seeds, runtime under 10, recommendation=run, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, Top models are tied on single-split evidence; robust best-model claim requires multi-seed verification.. Causal signal type: mixed. |
| run_007 | logistic_regression | yes | yes | Selected logistic_regression because score=97.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A new model family tests whether the current evidence depends on inductive bias.. World model predicted accuracy around 0.97-1.0, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, feature scaling is relevant for logistic regression. Causal signal type: mixed. |