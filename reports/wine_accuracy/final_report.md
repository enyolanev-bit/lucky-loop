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
| run_005 | Run a robustness sweep before allowing a best-hyperparameter claim. | ran multi-seed logistic_regression C sweep; signal=mixed | best mean accuracy 0.9667 | blocked: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise. |
| run_006 | Use boosting to test a staged-tree alternative under the remaining budget. | ran gradient_boosting; signal=mixed | accuracy 0.9556 | observation only; no robust claim |

## Best result

Best single run: run_002, model=logistic_regression, accuracy=1.0000, f1=1.0000.

## World model calibration

- Metric interval coverage: 50.00%
- Runtime interval coverage: 100.00%
- Prediction miss count: 3
- Useful decision signals: 6/6
- Full calibration table: `reports/world_model_calibration.md`

## Supported claims

- No claim reached supported or strongly_supported yet.

## Weakly supported claims

- No weakly supported claim was recorded.

## Blocked / inconclusive claims

- Blocked: C=0.1 is robustly better than C=10.0.
  Allowed rewrite: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise.

## Claim ledger

- Full ledger: `reports/claim_ledger.json`

## Prediction misses

- run_001: accuracy 0.9778 outside predicted range 0.90-0.97
- run_002: accuracy 1.0000 outside predicted range 0.95-0.98
- run_003: accuracy 1.0000 outside predicted range 0.90-0.98

## Evidence notes

### run_001
- Prediction rationale: A simple linear baseline is cheap and informative for sklearn tabular classification.
- Risks: unscaled features can slow convergence or underperform when feature scales differ
- State before: state_001; budget_remaining=6; known_results=0
- Candidates considered: action_001:logistic_regression, action_logistic_regression:logistic_regression, action_random_forest:random_forest, action_svc:svc, action_gradient_boosting:gradient_boosting, action_sweep_1_logistic_regression_C:verification_sweep
- Planner decision: Selected logistic_regression because score=110; world model recommended run; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.90-0.97, runtime under 5, recommendation=run, risks=unscaled features can slow convergence or underperform when feature scales differ. Causal signal type: mixed.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; random_forest: score=26; world model recommended run; world model predicted overfitting risk; verification_sweep: score=25; world model recommended run; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist; logistic_regression: score=-10; world model recommended run; scaled baseline is deferred until after an unscaled control
- Actual status: success, runtime: 0.6096s
- Unexpected: accuracy 0.9778 outside predicted range 0.90-0.97
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_002
- Prediction rationale: Scaling usually helps logistic regression on tabular numeric features.
- Risks: minor convergence warning possible
- State before: state_002; budget_remaining=5; known_results=1
- Candidates considered: action_logistic_regression:logistic_regression, action_random_forest:random_forest, action_svc:svc, action_gradient_boosting:gradient_boosting, action_sweep_1_logistic_regression_C:verification_sweep
- Planner decision: Selected logistic_regression because score=75; world model recommended run; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; random_forest: score=26; world model recommended run; world model predicted overfitting risk; verification_sweep: score=25; world model recommended run; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist
- Actual status: success, runtime: 0.0151s
- Unexpected: accuracy 1.0000 outside predicted range 0.95-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_003
- Prediction rationale: Tree ensembles test a different inductive bias, but may not beat a strong scaled linear model on small tabular datasets.
- Risks: overfitting or split variance if depth is unconstrained
- State before: state_003; budget_remaining=4; known_results=2
- Candidates considered: action_random_forest:random_forest, action_svc:svc, action_gradient_boosting:gradient_boosting, action_sweep_1_logistic_regression_C:verification_sweep
- Planner decision: Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.90-0.98, runtime under 10, recommendation=run, risks=overfitting or split variance if depth is unconstrained. Causal signal type: mixed.
- Rejected / deferred: verification_sweep: score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist; gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run
- Actual status: success, runtime: 0.8772s
- Unexpected: accuracy 1.0000 outside predicted range 0.90-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_004
- Prediction rationale: Scaled SVC can be strong on small numeric classification tasks but needs careful comparison.
- Risks: sensitive to scaling and C
- State before: state_004; budget_remaining=3; known_results=3
- Candidates considered: action_svc:svc, action_gradient_boosting:gradient_boosting, action_sweep_1_logistic_regression_C:verification_sweep
- Planner decision: Selected svc because score=60; world model recommended run; tree prediction missed, so a scaled margin model tests a different hypothesis. World model predicted accuracy around 0.90-0.99, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Causal signal type: mixed.
- Rejected / deferred: verification_sweep: score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist; gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk
- Actual status: success, runtime: 0.0102s
- Lesson: Prediction was broadly consistent with the real run.

### run_005
- Prediction rationale: A multi-seed sweep is useful for the deterministic verifier: it tests whether an apparent gain survives effect-vs-noise scrutiny.
- Risks: label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains
- State before: state_005; budget_remaining=2; known_results=4
- Candidates considered: action_gradient_boosting:gradient_boosting, action_sweep_1_logistic_regression_C:verification_sweep
- Planner decision: Selected verification_sweep because score=90; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk. World model predicted accuracy around 0.85-0.98, runtime under 35, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. Causal signal type: mixed.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk
- Actual status: success, runtime: 28.4141s
- Lesson: Prediction was broadly consistent with the real run.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.044444; seed_noise=0.088889; effect_to_noise_ratio=0.499994
- Blocked claim: C=0.1 is robustly better than C=10.0.
- Allowed claim: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise.
- Verifier rationale: Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery.

### run_006
- Prediction rationale: Boosting is a useful late comparison when linear and bagged-tree baselines are known.
- Risks: can overfit with too many estimators
- State before: state_006; budget_remaining=1; known_results=5
- Candidates considered: action_gradient_boosting:gradient_boosting
- Planner decision: Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.90-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. Causal signal type: mixed.
- Actual status: success, runtime: 0.6349s
- Lesson: Prediction was broadly consistent with the real run.
