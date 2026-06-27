# Lucky Loop Research Report

Goal: Maximize validation accuracy on sklearn wine under a small compute budget, while avoiding unsupported claims.

## Thesis

Predict before you compute, then verify before you claim: each experiment is simulated before real execution, compared against actual metrics, and any sweep claim is gated by a deterministic effect-vs-noise verifier.

## Experiment timeline

| Run | World model said | Agent did | Reality showed | Claim verdict |
|---|---|---|---|---|
| run_001 | Establish a baseline before testing scaling, nonlinear models, or sweeps. | ran logistic_regression; signal=mixed | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_002 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 1.0000 | prediction miss logged; no robust claim |
| run_003 | Use random forest to test whether nonlinear interactions improve over the linear baseline. | ran random_forest; signal=mixed | accuracy 1.0000 | prediction miss logged; no robust claim |
| run_004 | SVC tests a margin-based nonlinear hypothesis after linear/tree baselines. | ran svc; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_005 | Use boosting to test a staged-tree alternative under the remaining budget. | ran gradient_boosting; signal=mixed | accuracy 0.9556 | observation only; no robust claim |
| run_006 | Verify the top observed models on matched seeds before allowing a best-model claim. | verified top models: logistic_regression_scaled, random_forest_n=300, logistic_regression; signal=mixed | best mean accuracy 0.9956 | blocked: random_forest_n=300 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |
| run_007 | Run a robustness sweep before allowing a best-hyperparameter claim. | ran multi-seed logistic_regression C sweep; signal=mixed | best mean accuracy 0.9667 | blocked: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise. |

## Best result

Best single run: run_002, model=logistic_regression, accuracy=1.0000, f1=1.0000.

## Top model robustness

- run_006: verified logistic_regression_scaled, random_forest_n=300, logistic_regression; verdict=inconclusive; effect/noise=0.400009; random_forest_n=300 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.

## World model calibration

- Metric interval coverage: 42.86%
- Runtime interval coverage: 100.00%
- Prediction miss count: 4
- Useful decision signals: 7/7
- Full calibration table: `reports/world_model_calibration.md`

## Supported claims

- No claim reached supported or strongly_supported yet.

## Weakly supported claims

- No weakly supported claim was recorded.

## Blocked / inconclusive claims

- Blocked: random_forest_n=300 is robustly better than logistic_regression_scaled.
  Allowed rewrite: random_forest_n=300 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.
- Blocked: C=0.1 is robustly better than C=10.0.
  Allowed rewrite: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise.

## Claim ledger

- Full ledger: `reports/claim_ledger.json`

## Prediction misses

- run_001: accuracy 0.9778 outside predicted range 0.90-0.97
- run_002: accuracy 1.0000 outside predicted range 0.95-0.98
- run_003: accuracy 1.0000 outside predicted range 0.90-0.98
- run_006: accuracy 0.9956 outside predicted range 0.90-0.99

## Evidence notes

### run_001
- Prediction rationale: A simple linear baseline is cheap and informative for sklearn tabular classification.
- Risks: unscaled features can slow convergence or underperform when feature scales differ
- State before: state_001; budget_remaining=7; known_results=0
- Candidates considered: action_001:logistic_regression, action_logistic_regression:logistic_regression, action_random_forest:random_forest, action_svc:svc, action_gradient_boosting:gradient_boosting
- Planner decision: Selected logistic_regression because score=110; world model recommended run; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.90-0.97, runtime under 5, recommendation=run, risks=unscaled features can slow convergence or underperform when feature scales differ. Causal signal type: mixed.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; random_forest: score=26; world model recommended run; world model predicted overfitting risk; logistic_regression: score=-10; world model recommended run; scaled baseline is deferred until after an unscaled control
- Actual status: success, runtime: 0.4795s
- Unexpected: accuracy 0.9778 outside predicted range 0.90-0.97
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_002
- Prediction rationale: Scaling usually helps logistic regression on tabular numeric features.
- Risks: minor convergence warning possible
- State before: state_002; budget_remaining=6; known_results=1
- Candidates considered: action_logistic_regression:logistic_regression, action_random_forest:random_forest, action_svc:svc, action_gradient_boosting:gradient_boosting
- Planner decision: Selected logistic_regression because score=75; world model recommended run; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; random_forest: score=26; world model recommended run; world model predicted overfitting risk
- Actual status: success, runtime: 0.0138s
- Unexpected: accuracy 1.0000 outside predicted range 0.95-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_003
- Prediction rationale: Tree ensembles test a different inductive bias, but may not beat a strong scaled linear model on small tabular datasets.
- Risks: overfitting or split variance if depth is unconstrained
- State before: state_003; budget_remaining=5; known_results=2
- Candidates considered: action_random_forest:random_forest, action_svc:svc, action_gradient_boosting:gradient_boosting
- Planner decision: Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.90-0.98, runtime under 10, recommendation=run, risks=overfitting or split variance if depth is unconstrained. Causal signal type: mixed.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run
- Actual status: success, runtime: 0.7322s
- Unexpected: accuracy 1.0000 outside predicted range 0.90-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_004
- Prediction rationale: Scaled SVC can be strong on small numeric classification tasks but needs careful comparison.
- Risks: sensitive to scaling and C
- State before: state_004; budget_remaining=4; known_results=3
- Candidates considered: action_svc:svc, action_gradient_boosting:gradient_boosting
- Planner decision: Selected svc because score=60; world model recommended run; tree prediction missed, so a scaled margin model tests a different hypothesis. World model predicted accuracy around 0.90-0.99, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Causal signal type: mixed.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk
- Actual status: success, runtime: 0.0076s
- Lesson: Prediction was broadly consistent with the real run.

### run_005
- Prediction rationale: Boosting is a useful late comparison when linear and bagged-tree baselines are known.
- Risks: can overfit with too many estimators
- State before: state_005; budget_remaining=3; known_results=4
- Candidates considered: action_gradient_boosting:gradient_boosting
- Planner decision: Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.90-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. Causal signal type: mixed.
- Actual status: success, runtime: 0.5058s
- Lesson: Prediction was broadly consistent with the real run.

### run_006
- Prediction rationale: A multi-seed top-model comparison is required before reporting a robust best model.
- Risks: top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap
- State before: state_006; budget_remaining=2; known_results=5
- Candidates considered: action_verify_top_models:top_model_verification, action_sweep_1_logistic_regression_C:verification_sweep
- Planner decision: Selected top_model_verification because score=110; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. World model predicted accuracy around 0.90-0.99, runtime under 45, recommendation=run, risks=top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap. Causal signal type: mixed.
- Rejected / deferred: verification_sweep: score=90; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk
- Actual status: success, runtime: 31.6609s
- Unexpected: accuracy 0.9956 outside predicted range 0.90-0.99
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.008889; seed_noise=0.022222; effect_to_noise_ratio=0.400009
- Blocked claim: random_forest_n=300 is robustly better than logistic_regression_scaled.
- Allowed claim: random_forest_n=300 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.
- Verifier rationale: Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model.

### run_007
- Prediction rationale: A multi-seed sweep is useful for the deterministic verifier: it tests whether an apparent gain survives effect-vs-noise scrutiny.
- Risks: label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains
- State before: state_007; budget_remaining=1; known_results=6
- Candidates considered: action_sweep_1_logistic_regression_C:verification_sweep
- Planner decision: Selected verification_sweep because score=90; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk. World model predicted accuracy around 0.85-0.98, runtime under 35, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. Causal signal type: mixed.
- Actual status: success, runtime: 21.2372s
- Lesson: Prediction was broadly consistent with the real run.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.044444; seed_noise=0.088889; effect_to_noise_ratio=0.499994
- Blocked claim: C=0.1 is robustly better than C=10.0.
- Allowed claim: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise.
- Verifier rationale: Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery.
