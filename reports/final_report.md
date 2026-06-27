# Lucky Loop Research Report

Goal: Maximize validation accuracy on sklearn breast cancer dataset in five experiments.

## Thesis

Predict before you compute, then verify before you claim: each experiment is simulated before real execution, compared against actual metrics, and any sweep claim is gated by a deterministic effect-vs-noise verifier.

## Experiment timeline

| Run | Hypothesis | Model | Prediction | Actual accuracy | Match | Verifier | Decision |
|---|---|---|---|---:|---|---|---|
| run_001 | Establish a simple linear baseline before spending search budget. | logistic_regression | accuracy around 0.95-0.97, weighted F1 around 0.95-0.97 | 0.9510 | yes |  | Next selected by world-model-guided policy: World model warned that unscaled logistic regression may underperform; rerun the same baseline with feature scaling. |
| run_002 | World model warned that unscaled logistic regression may underperform; rerun the same baseline with feature scaling. | logistic_regression | accuracy around 0.95-0.97, weighted F1 around 0.95-0.97 | 0.9860 | partial/no |  | Next selected by world-model-guided policy: The current linear baseline is already strong; test a different inductive bias for robustness rather than only chasing accuracy. |
| run_003 | The current linear baseline is already strong; test a different inductive bias for robustness rather than only chasing accuracy. | random_forest | accuracy around 0.96-0.98, weighted F1 around 0.96-0.98 | 0.9580 | partial/no |  | Next selected by world-model-guided policy: The world model overestimated the tree ensemble; try a scaled margin-based model as a different hypothesis. |
| run_004 | The world model overestimated the tree ensemble; try a scaled margin-based model as a different hypothesis. | svc | accuracy around 0.97-0.99, weighted F1 around 0.97-0.99 | 0.9860 | yes |  | Next selected by world-model-guided policy: Run a controlled noisy-label multi-seed sweep so the deterministic Verifier can decide whether an apparent improvement is larger than seed noise. |
| run_005 | Run a controlled noisy-label multi-seed sweep so the deterministic Verifier can decide whether an apparent improvement is larger than seed noise. | verification_sweep | accuracy around 0.95-0.97, weighted F1 around 0.95-0.97 | best mean 0.9615 | partial/no | inconclusive; effect=0.020979; noise=0.027972 | Next selected by world-model-guided policy: Try conservative gradient boosting as a final ensemble baseline and compare prediction accuracy against actual metrics. |
| run_006 | Try conservative gradient boosting as a final ensemble baseline and compare prediction accuracy against actual metrics. | gradient_boosting | accuracy around 0.97-0.99, weighted F1 around 0.97-0.99 | 0.9510 | partial/no |  | Stop and report the best observed model. |

## Best result

Best single run: run_002, model=logistic_regression, accuracy=0.9860, f1=0.9860.

## Supported claims

- No sweep-level claim cleared the effect-vs-noise verifier yet.

## Weak / inconclusive findings

- Best config 'C=0.1' improves over 'C=10.0' by 0.0210 accuracy, but best-config seed noise is 0.0280.

## Prediction misses

- run_002: accuracy 0.9860 outside predicted range 0.95-0.97
- run_003: accuracy 0.9580 outside predicted range 0.96-0.98
- run_005: runtime 20.35s exceeded predicted 10s
- run_006: accuracy 0.9510 outside predicted range 0.97-0.99

## Evidence notes

### run_001
- Prediction rationale: Logistic regression on the breast cancer dataset typically achieves high accuracy even without scaling, though scaling might improve it slightly. The experiment is quick to run and provides a baseline for comparison.
- Risks: Model may not be fully optimized without scaling, Potential for slightly lower accuracy compared to scaled models
- Actual status: success, runtime: 0.291s
- Lesson: Prediction was broadly consistent with the real run.

### run_002
- Prediction rationale: The proposed experiment applies feature scaling to logistic regression on the breast cancer dataset, which aligns with the lesson from run_001 that unscaled logistic regression may underperform. Given the dataset's characteristics and the model's typical performance with scaling, accuracy and weighted F1 are expected to remain within the 0.95-0.97 range. The runtime is expected to be under 5 seconds, and there are minimal risks of failure.
- Risks: minimal risk of failure due to known dataset and model compatibility
- Actual status: success, runtime: 0.0192s
- Unexpected: accuracy 0.9860 outside predicted range 0.95-0.97
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_003
- Prediction rationale: Random forest is a robust non-linear model that should perform well on the breast cancer dataset. Given the success of scaled logistic regression with high accuracy, a random forest with 300 estimators is likely to achieve similar or slightly better performance. The main risks are overfitting and higher variance, but these are mitigated by the relatively small size of the dataset and the use of an ensemble method.
- Risks: Overfitting on small dataset, Higher variance in performance compared to linear models
- Actual status: success, runtime: 0.6423s
- Unexpected: accuracy 0.9580 outside predicted range 0.96-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_004
- Prediction rationale: The proposed SVC model with scaling and RBF kernel follows the world model's suggestion to test a scaled margin-based model. Previous linear and tree models showed strong performance, and an RBF SVM could potentially capture non-linear patterns while maintaining good generalization. The scaling parameter addresses previous concerns about unscaled features.
- Risks: SVC with RBF kernel may overfit if not properly tuned, Scaling is applied but C=2.0 might not be optimal for this dataset
- Actual status: success, runtime: 0.0212s
- Lesson: Prediction was broadly consistent with the real run.

### run_005
- Prediction rationale: The proposed experiment introduces label noise (0.08) which is not typical for the sklearn breast_cancer dataset and may artificially degrade performance. Additionally, the logistic regression with scaling has already achieved high accuracy (0.986) in run_002, so a sweep over C values with noise is unlikely to yield meaningful improvements. Consider removing the label noise parameter or testing on a different model variant.
- Risks: Label noise may reduce validation accuracy and F1 scores, Hyperparameter sweep may not significantly improve over best baseline, Seed variance could lead to inconsistent results across runs
- Actual status: success, runtime: 20.3479s
- Unexpected: runtime 20.35s exceeded predicted 10s
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.020979; seed_noise=0.027972
- Verifier rationale: Measured effect is not larger than inter-seed noise. The report must not oversell this as a robust discovery.

### run_006
- Prediction rationale: The gradient boosting model with conservative parameters (n_estimators=150, learning_rate=0.05) aligns with the world-model-guided policy to try a conservative gradient boosting ensemble baseline. Previous models achieved 0.986 accuracy with SVC and logistic regression with scaling, so gradient boosting should perform similarly or slightly better. The small dataset size and conservative hyperparameters reduce overfitting risks while maintaining reasonable runtime expectations.
- Risks: Gradient boosting may overfit on small datasets like breast_cancer if not properly regularized, Prediction accuracy might still miss quantitative details as seen in previous runs, Performance may not significantly exceed existing SVC and logistic regression baselines
- Actual status: success, runtime: 0.5161s
- Unexpected: accuracy 0.9510 outside predicted range 0.97-0.99
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
