# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 100.00%
- Runtime interval coverage: 85.71%
- Mean metric absolute error outside interval: 0.0000
- Mean runtime relative error above bound: 83.09%
- Prediction miss count: 1
- Risk recall approximation: 14.29%
- Useful decision signals: 7/7
- High claim-impact verification/stop decisions: 2
- Skip/stop recommendations: 1
- Memory-augmented predictions: 6/7
- Few-shot-augmented predictions: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.951 | 0.9510 | yes | under 5 | 0.57s | yes |  |
| run_002 | logistic_regression | accuracy around 0.97-0.99 | 0.9790 | yes | under 5 | 0.05s | yes |  |
| run_003 | svc | accuracy around 0.97-0.99 | 0.9720 | yes | under 5 | 0.03s | yes |  |
| run_004 | logistic_regression | accuracy around 0.97-0.99 | 0.9860 | yes | under 5 | 0.09s | yes |  |
| run_005 | logistic_regression | accuracy around 0.97-0.99 | 0.9720 | yes | under 1 | 0.03s | yes |  |
| run_006 | top_model_verification | accuracy around 0.97-0.99 with seed variance | 0.9706 | yes | under 5 | 34.08s | no | runtime 34.08s exceeded predicted 5s |
| run_007 | stop_and_report | accuracy around 0.97-0.99 with seed variance |  | no | under 5 | 0.04s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | single-run best score may not justify a scientific claim; scaling_sensitivity | none | no |
| run_002 | single-run best score may not justify a scientific claim; scaling_sensitivity | none | no |
| run_003 | single-run best score may not justify a scientific claim; scaling_sensitivity | none | no |
| run_004 | single-run best score may not justify a scientific claim; scaling_sensitivity | none | no |
| run_005 | single-run best score may not justify a scientific claim; scaling_sensitivity | none | no |
| run_006 | seed variance may exceed apparent model or hyperparameter effects; single-run best score may not justify a scientific claim; Top-model gap 0.0070 is within margin 0.0100; verify before claiming a winner. | runtime 34.08s exceeded predicted 5s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; Top-model gap 0.0070 is within margin 0.0100; verify before claiming a winner. | none | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=146.0; autoresearch agent preferred this catalog action; world model recommended modification, so the action remains informative but lower priority; world model predicted low compute value; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple unscaled linear baseline should anchor the search before interventions.. World model predicted accuracy around 0.951, runtime under 5, recommendation=modify, risks=single-run best score may not justify a scientific claim, scaling_sensitivity. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=89.0; autoresearch agent preferred this catalog action; world model recommended run; world model predicted low compute value; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize scale-sensitive linear classification.. World model predicted accuracy around 0.97-0.99, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, scaling_sensitivity. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=74.0; autoresearch agent preferred this catalog action; world model recommended run; world model predicted low compute value; new model family increases search coverage. Agent hypothesis: Testing a new model family can reveal whether the current best score depends on inductive bias.. World model predicted accuracy around 0.97-0.99, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, scaling_sensitivity. Causal signal type: mixed. |
| run_004 | logistic_regression | yes | yes | Selected logistic_regression because score=55.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; world model predicted low compute value; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Testing a new model family can reveal whether the current best score depends on inductive bias.. World model predicted accuracy around 0.97-0.99, runtime under 5, recommendation=run, risks=single-run best score may not justify a scientific claim, scaling_sensitivity. Causal signal type: mixed. |
| run_005 | logistic_regression | yes | yes | Selected logistic_regression because score=47.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; world model predicted low compute value; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: Testing a new model family can reveal whether the current best score depends on inductive bias.. World model predicted accuracy around 0.97-0.99, runtime under 1, recommendation=run, risks=single-run best score may not justify a scientific claim, scaling_sensitivity. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=170.0; autoresearch agent preferred this catalog action; world model recommended verification; world model predicted high claim impact; world model predicted low compute value; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.97-0.99 with seed variance, runtime under 5, recommendation=verify, risks=seed variance may exceed apparent model or hyperparameter effects, single-run best score may not justify a scientific claim, Top-model gap 0.0070 is within margin 0.0100; verify before claiming a winner.. Causal signal type: mixed. |
| run_007 | stop_and_report | yes | yes | Selected stop_and_report because score=122.0; autoresearch agent included this action in its candidate shortlist; world model recommended stop_and_report; world model predicted high claim impact; world model predicted low compute value; new model family increases search coverage; verifier already blocked the robust claim; stop instead of score-chasing; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-0.99 with seed variance, runtime under 5, recommendation=stop_and_report, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, Top-model gap 0.0070 is within margin 0.0100; verify before claiming a winner.. Causal signal type: mixed. |

## Prompt Context

| Run | Prompt version | Schema version | Few-shot examples | Retrieved memory examples | Claim impact | Compute value | Recommendation |
|---|---|---|---:|---:|---|---|---|
| run_001 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 0 | medium | low | modify |
| run_002 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 1 | medium | low | run |
| run_003 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 2 | medium | low | run |
| run_004 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | medium | low | run |
| run_005 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | medium | low | run |
| run_006 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | high | low | verify |
| run_007 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | high | low | stop_and_report |