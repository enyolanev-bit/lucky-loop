# Lucky Loop Research Report

Goal: Maximize validation accuracy on sklearn breast cancer dataset in five experiments.

## Thesis

Predict before you compute, then verify before you claim: each experiment is simulated before real execution, compared against actual metrics, and any sweep claim is gated by a deterministic effect-vs-noise verifier.

## Experiment timeline

| Run | Hypothesis | Model | Prediction | Actual accuracy | Match | Verifier | Decision |
|---|---|---|---|---:|---|---|---|
| run_001 | Establish a simple linear baseline before spending search budget. | logistic_regression | accuracy around 0.92-0.96 | 0.9510 | yes |  | Selected logistic_regression because score=110; world model recommended run; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.92-0.96, runtime under 5, recommendation=run, risks=unscaled features can slow convergence. |
| run_002 | Selected logistic_regression because score=75; world model recommended run; previous world-model risk flagged unscaled features. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. | logistic_regression | accuracy around 0.95-0.98 | 0.9860 | partial/no |  | Selected logistic_regression because score=75; world model recommended run; previous world-model risk flagged unscaled features. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. |
| run_003 | Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=overfitting if max_depth is unconstrained. | random_forest | accuracy around 0.94-0.98 | 0.9580 | yes |  | Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=overfitting if max_depth is unconstrained. |
| run_004 | Selected verification_sweep because score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist. World model predicted accuracy around 0.94-0.98, runtime under 25, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. | verification_sweep | accuracy around 0.94-0.98 | best mean 0.9615 | partial/no | inconclusive; effect=0.020979; noise=0.027972 | Selected verification_sweep because score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist. World model predicted accuracy around 0.94-0.98, runtime under 25, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. |
| run_005 | Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. | gradient_boosting | accuracy around 0.94-0.98 | 0.9510 | yes |  | Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. |
| run_006 | Selected svc because score=30; world model recommended run. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=sensitive to scaling and C. | svc | accuracy around 0.94-0.98 | 0.9860 | partial/no |  | Selected svc because score=30; world model recommended run. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=sensitive to scaling and C. |

## Best result

Best single run: run_002, model=logistic_regression, accuracy=0.9860, f1=0.9860.

## Supported claims

- No sweep-level claim cleared the effect-vs-noise verifier yet.

## Weak / inconclusive findings

- Best config 'C=0.1' improves over 'C=10.0' by 0.0210 accuracy, but best-config seed noise is 0.0280.

## Prediction misses

- run_002: accuracy 0.9860 outside predicted range 0.95-0.98
- run_004: runtime 31.28s exceeded predicted 25s
- run_006: accuracy 0.9860 outside predicted range 0.94-0.98

## Evidence notes

### run_001
- Prediction rationale: Good baseline for breast cancer dataset.
- Risks: unscaled features can slow convergence
- State before: state_001; budget_remaining=6; known_results=0
- Candidates considered: action_001:logistic_regression, action_scaled_logreg:logistic_regression, action_random_forest:random_forest, action_svc_scaled:svc, action_noisy_sweep:verification_sweep
- Planner decision: Selected logistic_regression because score=110; world model recommended run; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.92-0.96, runtime under 5, recommendation=run, risks=unscaled features can slow convergence.
- Rejected / deferred: svc: score=30; world model recommended run; random_forest: score=26; world model recommended run; world model predicted overfitting risk; verification_sweep: score=25; world model recommended run; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist; logistic_regression: score=-10; world model recommended run; scaled baseline is deferred until after an unscaled control
- Actual status: success, runtime: 0.4496s
- Lesson: Prediction was broadly consistent with the real run.

### run_002
- Prediction rationale: Scaling usually helps logistic regression on tabular medical data.
- Risks: minor convergence warning possible
- State before: state_002; budget_remaining=5; known_results=1
- Candidates considered: action_scaled_logreg:logistic_regression, action_random_forest:random_forest, action_svc_scaled:svc, action_noisy_sweep:verification_sweep, action_gradient_boosting:gradient_boosting
- Planner decision: Selected logistic_regression because score=75; world model recommended run; previous world-model risk flagged unscaled features. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; random_forest: score=26; world model recommended run; world model predicted overfitting risk; verification_sweep: score=25; world model recommended run; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist
- Actual status: success, runtime: 0.0414s
- Unexpected: accuracy 0.9860 outside predicted range 0.95-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_003
- Prediction rationale: Tree ensembles are strong baselines and robust to scaling.
- Risks: overfitting if max_depth is unconstrained
- State before: state_003; budget_remaining=4; known_results=2
- Candidates considered: action_random_forest:random_forest, action_svc_scaled:svc, action_noisy_sweep:verification_sweep, action_gradient_boosting:gradient_boosting
- Planner decision: Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=overfitting if max_depth is unconstrained.
- Rejected / deferred: verification_sweep: score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist; gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run
- Actual status: success, runtime: 0.8987s
- Lesson: Prediction was broadly consistent with the real run.

### run_004
- Prediction rationale: A multi-seed sweep is useful for the deterministic verifier: it tests whether an apparent gain survives effect-vs-noise scrutiny.
- Risks: label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains
- State before: state_004; budget_remaining=3; known_results=3
- Candidates considered: action_svc_scaled:svc, action_noisy_sweep:verification_sweep, action_gradient_boosting:gradient_boosting
- Planner decision: Selected verification_sweep because score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist. World model predicted accuracy around 0.94-0.98, runtime under 25, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run
- Actual status: success, runtime: 31.2782s
- Unexpected: runtime 31.28s exceeded predicted 25s
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.020979; seed_noise=0.027972
- Verifier rationale: Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery.

### run_005
- Prediction rationale: Boosting often performs well but needs tuning.
- Risks: can overfit with too many estimators
- State before: state_005; budget_remaining=2; known_results=4
- Candidates considered: action_svc_scaled:svc, action_gradient_boosting:gradient_boosting
- Planner decision: Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators.
- Rejected / deferred: svc: score=30; world model recommended run
- Actual status: success, runtime: 0.7815s
- Lesson: Prediction was broadly consistent with the real run.

### run_006
- Prediction rationale: Scaled RBF SVC is strong on small tabular datasets.
- Risks: sensitive to scaling and C
- State before: state_006; budget_remaining=1; known_results=5
- Candidates considered: action_svc_scaled:svc
- Planner decision: Selected svc because score=30; world model recommended run. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=sensitive to scaling and C.
- Actual status: success, runtime: 0.0189s
- Unexpected: accuracy 0.9860 outside predicted range 0.94-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
