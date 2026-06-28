# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: 100.00%
- Runtime interval coverage: 71.43%
- Mean metric absolute error outside interval: 0.0000
- Mean runtime relative error above bound: 8.21%
- Prediction miss count: 2
- Risk recall approximation: 14.29%
- Useful decision signals: 7/7
- High claim-impact verification/stop decisions: 1
- Skip/stop recommendations: 1
- Memory-augmented predictions: 6/7
- Few-shot-augmented predictions: 7/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | accuracy around 0.94-0.97 | 0.9644 | yes | under 5 | 6.39s | no | runtime 6.39s exceeded predicted 5s |
| run_002 | logistic_regression | accuracy around 0.97-0.98 | 0.9711 | yes | under 5 | 0.40s | yes |  |
| run_003 | svc | accuracy around 0.97-0.98 | 0.9778 | yes | under 5 | 0.10s | yes |  |
| run_004 | svc | accuracy around 0.97-0.98 | 0.9800 | yes | under 1 | 0.11s | yes |  |
| run_005 | logistic_regression | accuracy around 0.97-0.98 | 0.9778 | yes | under 5 | 1.00s | yes |  |
| run_006 | top_model_verification | accuracy around 0.977-0.982 across seeds | 0.9818 | yes | under 30 | 38.89s | no | runtime 38.89s exceeded predicted 30s |
| run_007 | stop_and_report | accuracy around 0.977-0.982 across seeds |  | no | under 5 | 0.03s | yes |  |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | logistic regression without scaling may underperform relative to scaled variants; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win on a single split | runtime 6.39s exceeded predicted 5s | no |
| run_002 | logistic regression without scaling may underperform relative to scaled variants; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win on a single split | none | no |
| run_003 | SVC with RBF kernel is sensitive to feature scaling; scaling is applied so performance should be stable; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win on a single split | none | no |
| run_004 | SVC with RBF kernel is sensitive to feature scaling; scaling is applied so performance should be stable; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win on a single split | none | no |
| run_005 | logistic regression with scaling is competitive but may not surpass best SVC single-run score; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win on a single split | none | no |
| run_006 | seed variance may exceed apparent model or hyperparameter effects; single-run best score may not justify a scientific claim; small-dataset variance could overstate a model win on a single split | runtime 38.89s exceeded predicted 30s; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | yes |
| run_007 | single-run best score may not justify a scientific claim; seed variance may exceed apparent model or hyperparameter effects; small-dataset variance could overstate a model win on a single split | none | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | yes | yes | Selected logistic_regression because score=140.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; world model predicted high claim impact; world model predicted low compute value; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple unscaled linear baseline should anchor the search before interventions.. World model predicted accuracy around 0.94-0.97, runtime under 5, recommendation=run, risks=logistic regression without scaling may underperform relative to scaled variants, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win on a single split. Causal signal type: mixed. |
| run_002 | logistic_regression | yes | yes | Selected logistic_regression because score=89.0; autoresearch agent preferred this catalog action; world model recommended run; world model predicted low compute value; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize scale-sensitive linear classification.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=logistic regression without scaling may underperform relative to scaled variants, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win on a single split. Causal signal type: mixed. |
| run_003 | svc | yes | yes | Selected svc because score=94.0; autoresearch agent preferred this catalog action; world model recommended run; world model predicted high claim impact; world model predicted low compute value; new model family increases search coverage. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=SVC with RBF kernel is sensitive to feature scaling; scaling is applied so performance should be stable, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win on a single split. Causal signal type: mixed. |
| run_004 | svc | yes | yes | Selected svc because score=64.0; autoresearch agent preferred this catalog action; world model recommended run; world model predicted high claim impact; world model predicted low compute value; candidate is a variant of an already tested family. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 1, recommendation=run, risks=SVC with RBF kernel is sensitive to feature scaling; scaling is applied so performance should be stable, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win on a single split. Causal signal type: mixed. |
| run_005 | logistic_regression | yes | yes | Selected logistic_regression because score=47.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; world model predicted low compute value; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.97-0.98, runtime under 5, recommendation=run, risks=logistic regression with scaling is competitive but may not surpass best SVC single-run score, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win on a single split. Causal signal type: mixed. |
| run_006 | top_model_verification | yes | yes | Selected top_model_verification because score=186.0; autoresearch agent preferred this catalog action; world model recommended verification; world model predicted high claim impact; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.977-0.982 across seeds, runtime under 30, recommendation=verify, risks=seed variance may exceed apparent model or hyperparameter effects, single-run best score may not justify a scientific claim, small-dataset variance could overstate a model win on a single split. Causal signal type: mixed. |
| run_007 | stop_and_report | yes | yes | Selected stop_and_report because score=88.0; autoresearch agent included this action in its candidate shortlist; world model recommended stop_and_report; world model predicted low claim impact; world model predicted low compute value; new model family increases search coverage; verifier already blocked the robust claim; stop instead of score-chasing; high best score increases claim risk for additional single-run score chasing. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.977-0.982 across seeds, runtime under 5, recommendation=stop_and_report, risks=single-run best score may not justify a scientific claim, seed variance may exceed apparent model or hyperparameter effects, small-dataset variance could overstate a model win on a single split. Causal signal type: mixed. |

## Prompt Context

| Run | Prompt version | Schema version | Few-shot examples | Retrieved memory examples | Claim impact | Compute value | Recommendation |
|---|---|---|---:|---:|---|---|---|
| run_001 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 0 | high | low | run |
| run_002 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 1 | medium | low | run |
| run_003 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 2 | high | low | run |
| run_004 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | high | low | run |
| run_005 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | medium | low | run |
| run_006 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | high | medium | verify |
| run_007 | world_model_prompt_v2 | prediction_schema_v2 | 3 | 3 | low | low | stop_and_report |