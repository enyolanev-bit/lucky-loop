# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 50.00%
- Runtime interval coverage: 85.71%
- Mean metric absolute error outside interval: 0.0104
- Mean runtime relative error above bound: 89.38%
- Prediction miss count: 4
- Risk recall approximation: 14.29%
- Useful decision signals: 7/7
- High claim-impact verification/stop decisions: 1
- Skip/stop recommendations: 1
- Memory-augmented predictions: 6/7
- Few-shot-augmented predictions: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.93-0.95 | 0.9778 | no | under 5 | 0.57s | yes | accuracy 0.9778 outside predicted range 0.93-0.95 |
| run_002 | logistic_regression | accuracy around 0.97-0.98 | 1.0000 | no | under 5 | 0.03s | yes | accuracy 1.0000 outside predicted range 0.97-0.98 |
| run_003 | logistic_regression | accuracy around 0.97-1.0 | 1.0000 | yes | under 1 | 0.01s | yes |  |
| run_004 | svc | accuracy around 0.96-0.99 | 0.9778 | yes | under 5 | 0.01s | yes |  |
| run_005 | gradient_boosting | accuracy around 0.97-0.99 | 0.9556 | no | under 5 | 0.39s | yes | accuracy 0.9556 outside predicted range 0.97-0.99 |
| run_006 | top_model_verification | accuracy around 0.97-1.0 across seeds | 0.9956 | yes | under 5 | 36.28s | no | runtime 36.28s exceeded predicted 5s |
| run_007 | stop_and_report | no further metric expected; current verified mean accuracy ~0.9956 with top models tied across seeds |  | no | under 1 | 0.02s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | feature scaling not applied for logistic regression; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win | accuracy 0.9778 outside predicted range 0.93-0.95 | no |
| run_002 | feature scaling not applied in run_001 may have suppressed logistic regression performance; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win | accuracy 1.0000 outside predicted range 0.97-0.98 | no |
| run_003 | feature scaling applied for logistic regression; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win; seed variance may exceed apparent model or hyperparameter effects | none | no |
| run_004 | feature scaling applied for SVC; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win; seed variance may exceed apparent model or hyperparameter effects | none | no |
| run_005 | small-dataset variance could overstate a model win; seed variance may exceed apparent model or hyperparameter effects; single-run best score may not justify a scientific claim | accuracy 0.9556 outside predicted range 0.97-0.99 | no |
| run_006 | seed variance may exceed apparent model or hyperparameter effects; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win | runtime 36.28s exceeded predicted 5s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; small-dataset variance could overstate a model win | none | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=146.0; autoresearch agent preferred this catalog action; world model recommended modification, so the action remains informative but lower priority; world model predicted low compute value; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple unscaled linear baseline should anchor the search before interventions.. World model predicted accuracy around 0.93-0.95, runtime under 5, recommendation=modify, risks=feature scaling not applied for logistic regression, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=89.0; autoresearch agent preferred this catalog action; world model recommended run; world model predicted low compute value; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize scale-sensitive linear classification.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=feature scaling not applied in run_001 may have suppressed logistic regression performance, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win. Causal signal type: mixed. |
| run_003 | logistic_regression | yes | yes | Selected logistic_regression because score=81.0; autoresearch agent included this action in its candidate shortlist; world model recommended verification; world model predicted high claim impact; world model predicted low compute value; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-1.0, runtime under 1, recommendation=verify, risks=feature scaling applied for logistic regression, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win, seed variance may exceed apparent model or hyperparameter effects. Causal signal type: mixed. |
| run_004 | svc | yes | yes | Selected svc because score=96.0; autoresearch agent preferred this catalog action; world model recommended verification; world model predicted high claim impact; world model predicted low compute value; new model family increases search coverage; world model predicted overfitting risk; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.96-0.99, runtime under 5, recommendation=verify, risks=feature scaling applied for SVC, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win, seed variance may exceed apparent model or hyperparameter effects. Causal signal type: mixed. |
| run_005 | gradient_boosting | yes | yes | Selected gradient_boosting because score=105.0; autoresearch agent preferred this catalog action; world model recommended verification; world model predicted high claim impact; world model predicted low compute value; new model family increases search coverage; ensemble baseline is useful as a late comparison; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-0.99, runtime under 5, recommendation=verify, risks=small-dataset variance could overstate a model win, seed variance may exceed apparent model or hyperparameter effects, single-run best score may not justify a scientific claim. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=170.0; autoresearch agent preferred this catalog action; world model recommended verification; world model predicted high claim impact; world model predicted low compute value; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.97-1.0 across seeds, runtime under 5, recommendation=verify, risks=seed variance may exceed apparent model or hyperparameter effects, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win. Causal signal type: mixed. |
| run_007 | stop_and_report | yes | yes | Selected stop_and_report because score=88.0; autoresearch agent included this action in its candidate shortlist; world model recommended stop_and_report; world model predicted low claim impact; world model predicted low compute value; new model family increases search coverage; verifier already blocked the robust claim; stop instead of score-chasing; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted no further metric expected; current verified mean accuracy ~0.9956 with top models tied across seeds, runtime under 1, recommendation=stop_and_report, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, small-dataset variance could overstate a model win. Causal signal type: mixed. |

## Prompt Context

| Run | Prompt version | Schema version | Few-shot examples | Retrieved memory examples | Claim impact | Compute value | Recommendation |
|---|---|---|---:|---:|---|---|---|
| run_001 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 0 | medium | low | modify |
| run_002 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 1 | medium | low | run |
| run_003 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 2 | high | low | verify |
| run_004 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | high | low | verify |
| run_005 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | high | low | verify |
| run_006 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | high | low | verify |
| run_007 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | low | low | stop_and_report |