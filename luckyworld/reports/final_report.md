# LuckyWorld Research Report

Goal: Maximize validation accuracy on sklearn breast cancer dataset in five experiments.

## Thesis

Predict before you compute: each experiment was simulated before real execution, then compared against actual logs and metrics.

## Experiment timeline

| Run | Hypothesis | Model | Prediction | Actual accuracy | Match | Decision |
|---|---|---|---|---:|---|---|
| run_001 | Establish a simple linear baseline before spending search budget. | logistic_regression | accuracy around 0.93-0.95, weighted F1 around 0.93-0.95 | 0.9510 | partial/no | Next selected by world-model-guided policy: World model warned that unscaled logistic regression may underperform; rerun the same baseline with feature scaling. |
| run_002 | World model warned that unscaled logistic regression may underperform; rerun the same baseline with feature scaling. | logistic_regression | accuracy around 0.95-0.97, weighted F1 around 0.95-0.97 | 0.9860 | partial/no | Next selected by world-model-guided policy: The current linear baseline is already strong; test a different inductive bias for robustness rather than only chasing accuracy. |
| run_003 | The current linear baseline is already strong; test a different inductive bias for robustness rather than only chasing accuracy. | random_forest | accuracy around 0.96-0.98, weighted F1 around 0.96-0.98 | 0.9580 | partial/no | Next selected by world-model-guided policy: The world model overestimated the tree ensemble; try a scaled margin-based model as a different hypothesis. |
| run_004 | The world model overestimated the tree ensemble; try a scaled margin-based model as a different hypothesis. | svc | accuracy around 0.97-0.99, weighted F1 around 0.97-0.99 | 0.9860 | yes | Next selected by world-model-guided policy: Try conservative gradient boosting as a final ensemble baseline and compare prediction accuracy against actual metrics. |
| run_005 | Try conservative gradient boosting as a final ensemble baseline and compare prediction accuracy against actual metrics. | gradient_boosting | accuracy around 0.97-0.99, weighted F1 around 0.97-0.99 | 0.9510 | partial/no | Stop and report the best observed model. |

## Best result

Best run: run_002, model=logistic_regression, accuracy=0.9860, f1=0.9860.

## Evidence notes

### run_001
- Prediction rationale: Logistic regression on the breast cancer dataset typically benefits from feature scaling. The current proposal has 'scale': false, which may lead to suboptimal performance. Modifying the experiment to include scaling or trying a model like random forest or SVM could yield higher accuracy.
- Risks: without scaling, logistic regression may perform suboptimally on unscaled features, potential for slightly lower accuracy compared to scaled or more complex models
- Actual status: success, runtime: 0.2846s
- Unexpected: accuracy 0.9510 outside predicted range 0.93-0.95
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_002
- Prediction rationale: The previous logistic regression model without scaling achieved an accuracy of 0.951, but the world model suggested that feature scaling could improve performance. Running the logistic regression with scaling is expected to maintain or slightly improve the accuracy and F1 scores, while keeping the runtime minimal.
- Risks: Model may still underperform if scaling parameters are not optimally set, Potential overfitting if scaling is not properly validated
- Actual status: success, runtime: 0.0305s
- Unexpected: accuracy 0.9860 outside predicted range 0.95-0.97
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_003
- Prediction rationale: Random forest is a robust non-linear model that should perform well on the breast cancer dataset. Given the success of scaled logistic regression with high accuracy, a random forest with 300 estimators is likely to achieve similar or slightly better performance while providing a different inductive bias for robustness testing.
- Risks: Overfitting on small dataset, Higher variance in performance compared to linear models
- Actual status: success, runtime: 0.658s
- Unexpected: accuracy 0.9580 outside predicted range 0.96-0.98
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.

### run_004
- Prediction rationale: The proposed SVC model with scaling and C=2.0 follows the world model's suggestion to test a scaled margin-based model. Previous linear and tree models achieved high accuracy, and an RBF-SVC is a logical next step to explore non-linear decision boundaries while maintaining robustness through scaling.
- Risks: SVC with RBF kernel may overfit if C is too high, Scaling is applied but kernel choice may limit generalization compared to linear models
- Actual status: success, runtime: 0.0146s
- Lesson: Prediction was broadly consistent with the real run.

### run_005
- Prediction rationale: The gradient boosting model is a conservative ensemble method that should perform well on the breast cancer dataset, likely achieving accuracy and weighted F1 in the 0.97-0.99 range. Previous runs show that ensemble methods like random forest and SVC perform strongly, and gradient boosting is a logical next step to compare against these baselines. The runtime is expected to be under 10 seconds given the dataset size and model complexity.
- Risks: Prediction may miss exact quantitative details like previous runs, Gradient boosting might not significantly improve over SVC or scaled logistic regression
- Actual status: success, runtime: 0.4891s
- Unexpected: accuracy 0.9510 outside predicted range 0.97-0.99
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
