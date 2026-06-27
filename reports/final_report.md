# Lucky Loop Research Report

Goal: Maximize validation accuracy on sklearn breast cancer dataset in five experiments.

## Thesis

Predict before you compute, then verify before you claim: each experiment is simulated before real execution, compared against actual metrics, and any sweep claim is gated by a deterministic effect-vs-noise verifier.

## Experiment timeline

| Run | Hypothesis | Model | Prediction | Actual metric | Match | Verifier | Decision |
|---|---|---|---|---:|---|---|---|
| run_001 | Establish a simple linear baseline before spending search budget. | logistic_regression | accuracy around 0.92-0.96 | 0.9510 | yes |  | Selected logistic_regression because score=110; world model recommended run; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.92-0.96, runtime under 5, recommendation=run, risks=unscaled features can slow convergence. |
| run_002 | Selected logistic_regression because score=75; world model recommended run; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. | logistic_regression | accuracy around 0.95-0.98 | 0.9860 | partial/no |  | Selected logistic_regression because score=75; world model recommended run; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. |
| run_003 | Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=overfitting if max_depth is unconstrained. | random_forest | accuracy around 0.94-0.98 | 0.9580 | yes |  | Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=overfitting if max_depth is unconstrained. |
| run_004 | Selected verification_sweep because score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist. World model predicted accuracy around 0.94-0.98, runtime under 25, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. | verification_sweep | accuracy around 0.94-0.98 | best mean 0.9615 | partial/no | inconclusive; effect=0.020979; noise=0.027972 | Selected verification_sweep because score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist. World model predicted accuracy around 0.94-0.98, runtime under 25, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. |
| run_005 | Selected real_effect because score=54; world model recommended run; real-effect scenario should produce a supported trust-ladder case. World model predicted accuracy around 0.76-0.88, runtime under 5, recommendation=run, risks=large effect should be checked against seed noise. | real_effect | accuracy around 0.76-0.88 | best mean 0.8675 | yes | strongly_supported; effect=0.1; noise=0.015 | Selected real_effect because score=54; world model recommended run; real-effect scenario should produce a supported trust-ladder case. World model predicted accuracy around 0.76-0.88, runtime under 5, recommendation=run, risks=large effect should be checked against seed noise. |
| run_006 | Selected weak_effect because score=48; world model recommended run; weak-effect scenario demonstrates the lower rung of the trust ladder. World model predicted accuracy around 0.94-0.99, runtime under 5, recommendation=run, risks=effect may be smaller than seed noise, claim should likely be weak or blocked. | weak_effect | accuracy around 0.94-0.99 | best mean 0.9750 | yes | inconclusive; effect=0.012; noise=0.03 | Selected weak_effect because score=48; world model recommended run; weak-effect scenario demonstrates the lower rung of the trust ladder. World model predicted accuracy around 0.94-0.99, runtime under 5, recommendation=run, risks=effect may be smaller than seed noise, claim should likely be weak or blocked. |
| run_007 | Selected data_leakage_trap because score=52; world model recommended run; data leakage trap tests whether protocol warnings block suspicious wins. World model predicted accuracy around 0.95-1.00, runtime under 5, recommendation=run, risks=suspiciously high accuracy may indicate data leakage, protocol warning should block a strong claim. | data_leakage_trap | accuracy around 0.95-1.00 | best mean 0.9992 | yes | inconclusive; effect=0.04525; noise=0.002 | Selected data_leakage_trap because score=52; world model recommended run; data leakage trap tests whether protocol warnings block suspicious wins. World model predicted accuracy around 0.95-1.00, runtime under 5, recommendation=run, risks=suspiciously high accuracy may indicate data leakage, protocol warning should block a strong claim. |
| run_008 | Selected metric_misuse because score=50; world model recommended run; metric misuse scenario tests whether the system avoids accuracy-only claims. World model predicted balanced_accuracy around 0.50-0.73, runtime under 5, recommendation=run, risks=accuracy may be misleading under class imbalance, balanced accuracy or F1 should be preferred. | metric_misuse | balanced_accuracy around 0.50-0.73 | best mean 0.7175 | yes | inconclusive; effect=0.21; noise=0.015 | Selected metric_misuse because score=50; world model recommended run; metric misuse scenario tests whether the system avoids accuracy-only claims. World model predicted balanced_accuracy around 0.50-0.73, runtime under 5, recommendation=run, risks=accuracy may be misleading under class imbalance, balanced accuracy or F1 should be preferred. |
| run_009 | Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. | gradient_boosting | accuracy around 0.94-0.98 | 0.9510 | yes |  | Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. |
| run_010 | Selected svc because score=30; world model recommended run. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=sensitive to scaling and C. | svc | accuracy around 0.94-0.98 | 0.9860 | partial/no |  | Selected svc because score=30; world model recommended run. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=sensitive to scaling and C. |

## Best result

Best single run: run_002, model=logistic_regression, accuracy=0.9860, f1=0.9860.

## World model calibration

- Metric interval coverage: 80.00%
- Runtime interval coverage: 90.00%
- Prediction miss count: 3
- Useful decision signals: 10/10
- Full calibration table: `reports/world_model_calibration.md`

## Supported claims

- candidate beat baseline by 0.1000 accuracy; effect/noise ratio=6.67, status=strongly supported.

## Weakly supported claims

- No weakly supported claim was recorded.

## Blocked / inconclusive claims

- Blocked: C=0.1 is robustly better than C=10.0.
  Allowed rewrite: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise.
- Blocked: candidate is robustly better than baseline.
  Allowed rewrite: candidate had the best mean accuracy, but the effect was smaller than seed noise.
- Blocked: leaky_protocol is a valid scientific winner.
  Allowed rewrite: leaky_protocol produced the best observed accuracy, but the protocol warning blocks a strong claim: suspiciously high score; label-derived feature was included before the split.
- Blocked: balanced_objective is a valid scientific winner.
  Allowed rewrite: balanced_objective produced the best observed balanced_accuracy, but the protocol warning blocks a strong claim: accuracy is misleading on an imbalanced dataset; balanced_accuracy is the verifier metric.

## Claim ledger

- Full ledger: `reports/claim_ledger.json`

## Prediction misses

- run_002: accuracy 0.9860 outside predicted range 0.95-0.98
- run_004: runtime 28.40s exceeded predicted 25s
- run_010: accuracy 0.9860 outside predicted range 0.94-0.98

## Evidence notes

### run_001
- Prediction rationale: Good baseline for breast cancer dataset.
- Risks: unscaled features can slow convergence
- State before: state_001; budget_remaining=10; known_results=0
- Candidates considered: action_001:logistic_regression, action_scaled_logreg:logistic_regression, action_random_forest:random_forest, action_svc_scaled:svc, action_noisy_sweep:verification_sweep, action_gradient_boosting:gradient_boosting, action_weak_effect:weak_effect, action_real_effect:real_effect
- Planner decision: Selected logistic_regression because score=110; world model recommended run; first run should establish the unscaled baseline before interventions. World model predicted accuracy around 0.92-0.96, runtime under 5, recommendation=run, risks=unscaled features can slow convergence.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; weak_effect: score=30; world model recommended run; real_effect: score=30; world model recommended run; random_forest: score=26; world model recommended run; world model predicted overfitting risk; verification_sweep: score=25; world model recommended run; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist; logistic_regression: score=-10; world model recommended run; scaled baseline is deferred until after an unscaled control
- Actual status: success, runtime: 0.5161s
- Lesson: Prediction was broadly consistent with the real run.

### run_002
- Prediction rationale: Scaling usually helps logistic regression on tabular medical data.
- Risks: minor convergence warning possible
- State before: state_002; budget_remaining=9; known_results=1
- Candidates considered: action_scaled_logreg:logistic_regression, action_random_forest:random_forest, action_svc_scaled:svc, action_noisy_sweep:verification_sweep, action_gradient_boosting:gradient_boosting, action_weak_effect:weak_effect, action_real_effect:real_effect, action_leakage_trap:data_leakage_trap
- Planner decision: Selected logistic_regression because score=75; world model recommended run; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; weak_effect: score=30; world model recommended run; real_effect: score=30; world model recommended run; data_leakage_trap: score=30; world model recommended run; random_forest: score=26; world model recommended run; world model predicted overfitting risk; verification_sweep: score=25; world model recommended run; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist
- Actual status: success, runtime: 0.0373s
- Unexpected: accuracy 0.9860 outside predicted range 0.95-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_003
- Prediction rationale: Tree ensembles are strong baselines and robust to scaling.
- Risks: overfitting if max_depth is unconstrained
- State before: state_003; budget_remaining=8; known_results=2
- Candidates considered: action_random_forest:random_forest, action_svc_scaled:svc, action_noisy_sweep:verification_sweep, action_gradient_boosting:gradient_boosting, action_weak_effect:weak_effect, action_real_effect:real_effect, action_leakage_trap:data_leakage_trap, action_metric_misuse:metric_misuse
- Planner decision: Selected random_forest because score=61; world model recommended run; strong linear baseline makes a different inductive bias useful; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=overfitting if max_depth is unconstrained.
- Rejected / deferred: verification_sweep: score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist; gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; weak_effect: score=30; world model recommended run; real_effect: score=30; world model recommended run; data_leakage_trap: score=30; world model recommended run; metric_misuse: score=30; world model recommended run
- Actual status: success, runtime: 0.8226s
- Lesson: Prediction was broadly consistent with the real run.

### run_004
- Prediction rationale: A multi-seed sweep is useful for the deterministic verifier: it tests whether an apparent gain survives effect-vs-noise scrutiny.
- Risks: label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains
- State before: state_004; budget_remaining=7; known_results=3
- Candidates considered: action_svc_scaled:svc, action_noisy_sweep:verification_sweep, action_gradient_boosting:gradient_boosting, action_weak_effect:weak_effect, action_real_effect:real_effect, action_leakage_trap:data_leakage_trap, action_metric_misuse:metric_misuse
- Planner decision: Selected verification_sweep because score=60; world model recommended run; high single-run score needs robustness verification before a strong claim; world model predicted robustness or seed-variance risk; defer verifier sweep until several single-model baselines exist. World model predicted accuracy around 0.94-0.98, runtime under 25, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; weak_effect: score=30; world model recommended run; real_effect: score=30; world model recommended run; data_leakage_trap: score=30; world model recommended run; metric_misuse: score=30; world model recommended run
- Actual status: success, runtime: 28.4049s
- Unexpected: runtime 28.40s exceeded predicted 25s
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.020979; seed_noise=0.027972; effect_to_noise_ratio=0.75
- Blocked claim: C=0.1 is robustly better than C=10.0.
- Allowed claim: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise.
- Verifier rationale: Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery.

### run_005
- Prediction rationale: A real-effect scenario should produce a supported claim when the effect/noise ratio is high.
- Risks: large effect should be checked against seed noise
- State before: state_005; budget_remaining=6; known_results=4
- Candidates considered: action_svc_scaled:svc, action_gradient_boosting:gradient_boosting, action_weak_effect:weak_effect, action_real_effect:real_effect, action_leakage_trap:data_leakage_trap, action_metric_misuse:metric_misuse
- Planner decision: Selected real_effect because score=54; world model recommended run; real-effect scenario should produce a supported trust-ladder case. World model predicted accuracy around 0.76-0.88, runtime under 5, recommendation=run, risks=large effect should be checked against seed noise.
- Rejected / deferred: weak_effect: score=48; world model recommended run; weak-effect scenario demonstrates the lower rung of the trust ladder; gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; data_leakage_trap: score=30; world model recommended run; metric_misuse: score=30; world model recommended run
- Actual status: success, runtime: 0.0s
- Lesson: Prediction was broadly consistent with the real run.
- Verifier verdict: strongly_supported; trustworthy=True; effect_size=0.1; seed_noise=0.015; effect_to_noise_ratio=6.666667
- Allowed claim: candidate beat baseline by 0.1000 accuracy; effect/noise ratio=6.67, status=strongly supported.
- Verifier rationale: Measured effect is larger than inter-seed noise for the best config. This is not a full statistical significance engine; it is a conservative claim gate.

### run_006
- Prediction rationale: A weak-effect scenario tests whether the verifier refuses to overclaim a small apparent improvement.
- Risks: effect may be smaller than seed noise, claim should likely be weak or blocked
- State before: state_006; budget_remaining=5; known_results=5
- Candidates considered: action_svc_scaled:svc, action_gradient_boosting:gradient_boosting, action_weak_effect:weak_effect, action_leakage_trap:data_leakage_trap, action_metric_misuse:metric_misuse
- Planner decision: Selected weak_effect because score=48; world model recommended run; weak-effect scenario demonstrates the lower rung of the trust ladder. World model predicted accuracy around 0.94-0.99, runtime under 5, recommendation=run, risks=effect may be smaller than seed noise, claim should likely be weak or blocked.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run; data_leakage_trap: score=30; world model recommended run; metric_misuse: score=30; world model recommended run
- Actual status: success, runtime: 0.0s
- Lesson: Prediction was broadly consistent with the real run.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.012; seed_noise=0.03; effect_to_noise_ratio=0.4
- Blocked claim: candidate is robustly better than baseline.
- Allowed claim: candidate had the best mean accuracy, but the effect was smaller than seed noise.
- Verifier rationale: Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery.

### run_007
- Prediction rationale: A leakage trap tests whether the system blocks claims from invalid protocols even when metrics look excellent.
- Risks: suspiciously high accuracy may indicate data leakage, protocol warning should block a strong claim
- State before: state_007; budget_remaining=4; known_results=6
- Candidates considered: action_svc_scaled:svc, action_gradient_boosting:gradient_boosting, action_leakage_trap:data_leakage_trap, action_metric_misuse:metric_misuse
- Planner decision: Selected data_leakage_trap because score=52; world model recommended run; data leakage trap tests whether protocol warnings block suspicious wins. World model predicted accuracy around 0.95-1.00, runtime under 5, recommendation=run, risks=suspiciously high accuracy may indicate data leakage, protocol warning should block a strong claim.
- Rejected / deferred: metric_misuse: score=50; world model recommended run; metric misuse scenario tests whether the system avoids accuracy-only claims; gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run
- Actual status: success, runtime: 0.0s
- Lesson: Prediction was broadly consistent with the real run.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.04525; seed_noise=0.002; effect_to_noise_ratio=22.625
- Blocked claim: leaky_protocol is a valid scientific winner.
- Allowed claim: leaky_protocol produced the best observed accuracy, but the protocol warning blocks a strong claim: suspiciously high score; label-derived feature was included before the split.
- Verifier rationale: Protocol warning blocks the claim even though the metric effect may look large: suspiciously high score; label-derived feature was included before the split

### run_008
- Prediction rationale: Metric misuse tests whether the report avoids accuracy-only claims on an imbalanced setting.
- Risks: accuracy may be misleading under class imbalance, balanced accuracy or F1 should be preferred
- State before: state_008; budget_remaining=3; known_results=7
- Candidates considered: action_svc_scaled:svc, action_gradient_boosting:gradient_boosting, action_metric_misuse:metric_misuse
- Planner decision: Selected metric_misuse because score=50; world model recommended run; metric misuse scenario tests whether the system avoids accuracy-only claims. World model predicted balanced_accuracy around 0.50-0.73, runtime under 5, recommendation=run, risks=accuracy may be misleading under class imbalance, balanced accuracy or F1 should be preferred.
- Rejected / deferred: gradient_boosting: score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk; svc: score=30; world model recommended run
- Actual status: success, runtime: 0.0s
- Lesson: Prediction was broadly consistent with the real run.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.21; seed_noise=0.015; effect_to_noise_ratio=14.0
- Blocked claim: balanced_objective is a valid scientific winner.
- Allowed claim: balanced_objective produced the best observed balanced_accuracy, but the protocol warning blocks a strong claim: accuracy is misleading on an imbalanced dataset; balanced_accuracy is the verifier metric.
- Verifier rationale: Protocol warning blocks the claim even though the metric effect may look large: accuracy is misleading on an imbalanced dataset; balanced_accuracy is the verifier metric

### run_009
- Prediction rationale: Boosting often performs well but needs tuning.
- Risks: can overfit with too many estimators
- State before: state_009; budget_remaining=2; known_results=8
- Candidates considered: action_svc_scaled:svc, action_gradient_boosting:gradient_boosting
- Planner decision: Selected gradient_boosting because score=31; world model recommended run; ensemble baseline is useful as a late comparison; world model predicted overfitting risk. World model predicted accuracy around 0.94-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators.
- Rejected / deferred: svc: score=30; world model recommended run
- Actual status: success, runtime: 0.6433s
- Lesson: Prediction was broadly consistent with the real run.

### run_010
- Prediction rationale: Scaled RBF SVC is strong on small tabular datasets.
- Risks: sensitive to scaling and C
- State before: state_010; budget_remaining=1; known_results=9
- Candidates considered: action_svc_scaled:svc
- Planner decision: Selected svc because score=30; world model recommended run. World model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=sensitive to scaling and C.
- Actual status: success, runtime: 0.0159s
- Unexpected: accuracy 0.9860 outside predicted range 0.94-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
