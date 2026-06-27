# Lucky Loop Research Report

Goal: Maximize validation accuracy on sklearn breast cancer dataset in five experiments.

## Thesis

Predict before you compute, then verify before you claim: each experiment is simulated before real execution, compared against actual metrics, and any sweep claim is gated by a deterministic effect-vs-noise verifier.

## Experiment timeline

| Run | Hypothesis | Model | Prediction | Actual accuracy | Match | Verifier | Decision |
|---|---|---|---|---:|---|---|---|
| run_001 | Establish a simple linear baseline before spending search budget. | logistic_regression | accuracy around 0.92-0.96 | 0.9510 | yes |  | Executed the selected action after the world model predicted accuracy around 0.92-0.96, runtime under 5, recommendation=run, risks=unscaled features can slow convergence. Next decision: Next selected by world-model-guided policy: World model warned that unscaled logistic regression may underperform; rerun the same baseline with feature scaling. |
| run_002 | World model warned that unscaled logistic regression may underperform; rerun the same baseline with feature scaling. | logistic_regression | accuracy around 0.95-0.98 | 0.9860 | partial/no |  | Executed the selected action after the world model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Next decision: Next selected by world-model-guided policy: The current linear baseline is already strong; test a different inductive bias for robustness rather than only chasing accuracy. |
| run_003 | The current linear baseline is already strong; test a different inductive bias for robustness rather than only chasing accuracy. | random_forest | accuracy around 0.94-0.98 | 0.9580 | yes |  | Executed the selected action after the world model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=overfitting if max_depth is unconstrained. Next decision: Next selected by world-model-guided policy: Try a scaled RBF SVC because Qwen-AgentWorld expects scaling-sensitive models to be competitive on small tabular data. |
| run_004 | Try a scaled RBF SVC because Qwen-AgentWorld expects scaling-sensitive models to be competitive on small tabular data. | svc | accuracy around 0.94-0.98 | 0.9860 | partial/no |  | Executed the selected action after the world model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Next decision: Next selected by world-model-guided policy: Run a controlled noisy-label multi-seed sweep so the deterministic Verifier can decide whether an apparent improvement is larger than seed noise. |
| run_005 | Run a controlled noisy-label multi-seed sweep so the deterministic Verifier can decide whether an apparent improvement is larger than seed noise. | verification_sweep | accuracy around 0.94-0.98 | best mean 0.9615 | partial/no | inconclusive; effect=0.020979; noise=0.027972 | Executed the selected action after the world model predicted accuracy around 0.94-0.98, runtime under 25, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. Next decision: Next selected by world-model-guided policy: Try conservative gradient boosting as a final ensemble baseline and compare prediction accuracy against actual metrics. |
| run_006 | Try conservative gradient boosting as a final ensemble baseline and compare prediction accuracy against actual metrics. | gradient_boosting | accuracy around 0.94-0.98 | 0.9510 | yes |  | Executed the selected action after the world model predicted accuracy around 0.94-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. Next decision: Stop and report the best observed model. |

## Best result

Best single run: run_002, model=logistic_regression, accuracy=0.9860, f1=0.9860.

## Supported claims

- No sweep-level claim cleared the effect-vs-noise verifier yet.

## Weak / inconclusive findings

- Best config 'C=0.1' improves over 'C=10.0' by 0.0210 accuracy, but best-config seed noise is 0.0280.

## Prediction misses

- run_002: accuracy 0.9860 outside predicted range 0.95-0.98
- run_004: accuracy 0.9860 outside predicted range 0.94-0.98
- run_005: runtime 48.65s exceeded predicted 25s

## Evidence notes

### run_001
- Prediction rationale: Good baseline for breast cancer dataset.
- Risks: unscaled features can slow convergence
- State before: state_001; budget_remaining=6; known_results=0
- Candidates considered: action_selected:logistic_regression, action_next_candidate:logistic_regression
- Planner decision: Executed the selected action after the world model predicted accuracy around 0.92-0.96, runtime under 5, recommendation=run, risks=unscaled features can slow convergence. Next decision: Next selected by world-model-guided policy: World model warned that unscaled logistic regression may underperform; rerun the same baseline with feature scaling.
- Rejected / deferred: logistic_regression: Not executed in this step; it is the queued candidate for the next research iteration.
- Actual status: success, runtime: 1.006s
- Lesson: Prediction was broadly consistent with the real run.

### run_002
- Prediction rationale: Scaling usually helps logistic regression on tabular medical data.
- Risks: minor convergence warning possible
- State before: state_002; budget_remaining=5; known_results=1
- Candidates considered: action_selected:logistic_regression, action_next_candidate:random_forest
- Planner decision: Executed the selected action after the world model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Next decision: Next selected by world-model-guided policy: The current linear baseline is already strong; test a different inductive bias for robustness rather than only chasing accuracy.
- Rejected / deferred: random_forest: Not executed in this step; it is the queued candidate for the next research iteration.
- Actual status: success, runtime: 0.0342s
- Unexpected: accuracy 0.9860 outside predicted range 0.95-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_003
- Prediction rationale: Tree ensembles are strong baselines and robust to scaling.
- Risks: overfitting if max_depth is unconstrained
- State before: state_003; budget_remaining=4; known_results=2
- Candidates considered: action_selected:random_forest, action_next_candidate:svc
- Planner decision: Executed the selected action after the world model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=overfitting if max_depth is unconstrained. Next decision: Next selected by world-model-guided policy: Try a scaled RBF SVC because Qwen-AgentWorld expects scaling-sensitive models to be competitive on small tabular data.
- Rejected / deferred: svc: Not executed in this step; it is the queued candidate for the next research iteration.
- Actual status: success, runtime: 1.0894s
- Lesson: Prediction was broadly consistent with the real run.

### run_004
- Prediction rationale: Scaled RBF SVC is strong on small tabular datasets.
- Risks: sensitive to scaling and C
- State before: state_004; budget_remaining=3; known_results=3
- Candidates considered: action_selected:svc, action_next_candidate:verification_sweep
- Planner decision: Executed the selected action after the world model predicted accuracy around 0.94-0.98, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Next decision: Next selected by world-model-guided policy: Run a controlled noisy-label multi-seed sweep so the deterministic Verifier can decide whether an apparent improvement is larger than seed noise.
- Rejected / deferred: verification_sweep: Not executed in this step; it is the queued candidate for the next research iteration.
- Actual status: success, runtime: 0.0226s
- Unexpected: accuracy 0.9860 outside predicted range 0.94-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_005
- Prediction rationale: A multi-seed sweep is useful for the deterministic verifier: it tests whether an apparent gain survives effect-vs-noise scrutiny.
- Risks: label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains
- State before: state_005; budget_remaining=2; known_results=4
- Candidates considered: action_selected:verification_sweep, action_next_candidate:gradient_boosting
- Planner decision: Executed the selected action after the world model predicted accuracy around 0.94-0.98, runtime under 25, recommendation=run, risks=label noise can make small hyperparameter differences non-robust, seed variance may exceed apparent gains. Next decision: Next selected by world-model-guided policy: Try conservative gradient boosting as a final ensemble baseline and compare prediction accuracy against actual metrics.
- Rejected / deferred: gradient_boosting: Not executed in this step; it is the queued candidate for the next research iteration.
- Actual status: success, runtime: 48.6546s
- Unexpected: runtime 48.65s exceeded predicted 25s
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.020979; seed_noise=0.027972
- Verifier rationale: Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery.

### run_006
- Prediction rationale: Boosting often performs well but needs tuning.
- Risks: can overfit with too many estimators
- State before: state_006; budget_remaining=1; known_results=5
- Candidates considered: action_selected:gradient_boosting
- Planner decision: Executed the selected action after the world model predicted accuracy around 0.94-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. Next decision: Stop and report the best observed model.
- Actual status: success, runtime: 0.943s
- Lesson: Prediction was broadly consistent with the real run.
