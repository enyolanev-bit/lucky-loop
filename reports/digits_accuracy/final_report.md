# Lucky Loop Research Report

Goal: Maximize validation accuracy on sklearn digits under a small compute budget, while avoiding unsupported claims.

## Thesis

Predict before you compute, then verify before you claim: each experiment is simulated before real execution, compared against actual metrics, and any sweep claim is gated by a deterministic effect-vs-noise verifier.

## Experiment timeline

| Run | World model said | Agent did | Reality showed | Claim verdict |
|---|---|---|---|---|
| run_001 | Establish a baseline before testing scaling, nonlinear models, or sweeps. | ran logistic_regression; signal=mixed | accuracy 0.9622 | observation only; no robust claim |
| run_002 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9711 | observation only; no robust claim |
| run_003 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_004 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9644 | observation only; no robust claim |
| run_005 | SVC tests a margin-based nonlinear hypothesis after linear/tree baselines. | ran svc; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_006 | Use random forest to test whether nonlinear interactions improve over the linear baseline. | ran random_forest; signal=mixed | accuracy 0.9600 | observation only; no robust claim |
| run_007 | Verify the top observed models on matched seeds before allowing a best-model claim. | verified top models: logistic_regression_scaled_C=1.0, svc_scaled_C=0.5_kernel=rbf, logistic_regression_scaled_C=0.1; signal=mixed | best mean accuracy 0.9764 | blocked: svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |

## Best result

Best single run: run_003, model=logistic_regression, accuracy=0.9778, f1=0.9778.

## Top model robustness

- run_007: verified logistic_regression_scaled_C=1.0, svc_scaled_C=0.5_kernel=rbf, logistic_regression_scaled_C=0.1; verdict=inconclusive; effect/noise=0.639996; svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.

## World model calibration

- Metric interval coverage: 100.00%
- Runtime interval coverage: 85.71%
- Prediction miss count: 1
- Useful decision signals: 7/7
- Full calibration table: `reports/world_model_calibration.md`

## Supported claims

- No claim reached supported or strongly_supported yet.

## Weakly supported claims

- No weakly supported claim was recorded.

## Blocked / inconclusive claims

- Blocked: svc_scaled_C=0.5_kernel=rbf is robustly better than logistic_regression_scaled_C=1.0.
  Allowed rewrite: svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.

## Claim ledger

- Full ledger: `reports/claim_ledger.json`

## Prediction misses

- run_007: runtime 45.90s exceeded predicted 45s

## Evidence notes

### run_001
- Prediction rationale: A simple linear baseline is cheap and informative for sklearn tabular classification.
- Risks: unscaled features can slow convergence or underperform when feature scales differ
- State before: state_001; budget_remaining=7; known_results=0
- Candidates considered: action_001:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_logistic_regression_C-0p1_scale-True:logistic_regression, action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest
- Planner decision: Selected logistic_regression because score=128.0; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.90-0.97, runtime under 5, recommendation=run, risks=unscaled features can slow convergence or underperform when feature scales differ. Causal signal type: mixed.
- Rejected / deferred: logistic_regression: score=128.0; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions; logistic_regression: score=128.0; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions; logistic_regression: score=128.0; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; logistic_regression: score=8.0; world model recommended run; new model family increases search coverage; scaled baseline is deferred until after an unscaled control; logistic_regression: score=8.0; world model recommended run; new model family increases search coverage; scaled baseline is deferred until after an unscaled control; logistic_regression: score=8.0; world model recommended run; new model family increases search coverage; scaled baseline is deferred until after an unscaled control
- Actual status: success, runtime: 3.4108s
- Lesson: Prediction was broadly consistent with the real run.

### run_002
- Prediction rationale: Scaling usually helps logistic regression on tabular numeric features.
- Risks: minor convergence warning possible
- State before: state_002; budget_remaining=6; known_results=1
- Candidates considered: action_logistic_regression_C-0p1_scale-True:logistic_regression, action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest
- Planner decision: Selected logistic_regression because score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed.
- Rejected / deferred: logistic_regression: score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; logistic_regression: score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 1.0745s
- Lesson: Prediction was broadly consistent with the real run.

### run_003
- Prediction rationale: Scaling usually helps logistic regression on tabular numeric features.
- Risks: minor convergence warning possible
- State before: state_003; budget_remaining=5; known_results=2
- Candidates considered: action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest
- Planner decision: Selected logistic_regression because score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed.
- Rejected / deferred: logistic_regression: score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 0.814s
- Lesson: Prediction was broadly consistent with the real run.

### run_004
- Prediction rationale: Scaling usually helps logistic regression on tabular numeric features.
- Risks: minor convergence warning possible
- State before: state_004; budget_remaining=4; known_results=3
- Candidates considered: action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest
- Planner decision: Selected logistic_regression because score=63.0; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed.
- Rejected / deferred: svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 0.239s
- Lesson: Prediction was broadly consistent with the real run.

### run_005
- Prediction rationale: Scaled SVC can be strong on small numeric classification tasks but needs careful comparison.
- Risks: sensitive to scaling and C
- State before: state_005; budget_remaining=3; known_results=4
- Candidates considered: action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest, action_random_forest_max_depth-8_n_estimators-300:random_forest
- Planner decision: Selected svc because score=48.0; world model recommended run; new model family increases search coverage. World model predicted accuracy around 0.90-0.99, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Causal signal type: mixed.
- Rejected / deferred: svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 0.1271s
- Lesson: Prediction was broadly consistent with the real run.

### run_006
- Prediction rationale: Tree ensembles test a different inductive bias, but may not beat a strong scaled linear model on small tabular datasets.
- Risks: overfitting or split variance if depth is unconstrained
- State before: state_006; budget_remaining=2; known_results=5
- Candidates considered: action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest, action_random_forest_max_depth-8_n_estimators-300:random_forest, action_hist_gradient_boosting_learning_rate-0p05_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p05_max_depth-8_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_n_estimators-100:hist_gradient_boosting
- Planner decision: Selected random_forest because score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk. World model predicted accuracy around 0.90-0.98, runtime under 10, recommendation=run, risks=overfitting or split variance if depth is unconstrained. Causal signal type: mixed.
- Rejected / deferred: random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 0.4276s
- Lesson: Prediction was broadly consistent with the real run.

### run_007
- Prediction rationale: A multi-seed top-model comparison is required before reporting a robust best model.
- Risks: top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap
- State before: state_007; budget_remaining=1; known_results=6
- Candidates considered: action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_hist_gradient_boosting_learning_rate-0p05_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p05_max_depth-8_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_max_depth-8_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p05_n_estimators-150:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p05_max_depth-8_n_estimators-150:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_n_estimators-150:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_max_depth-8_n_estimators-150:hist_gradient_boosting, action_verify_top_models:top_model_verification
- Planner decision: Selected top_model_verification because score=110.0; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. World model predicted accuracy around 0.90-0.99, runtime under 45, recommendation=run, risks=top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap. Causal signal type: mixed.
- Rejected / deferred: hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=18.0; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 45.897s
- Unexpected: runtime 45.90s exceeded predicted 45s
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.007111; seed_noise=0.011111; effect_to_noise_ratio=0.639996
- Blocked claim: svc_scaled_C=0.5_kernel=rbf is robustly better than logistic_regression_scaled_C=1.0.
- Allowed claim: svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.
- Verifier rationale: Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model.
