# Lucky Loop Demo Summary

Goal: Maximize validation accuracy on sklearn breast_cancer under a small compute budget, while avoiding unsupported claims.

All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: predict, run, compare, verify.

| Run | World model said | Agent did | Reality showed | Claim verdict |
|---|---|---|---|---|
| run_001 | Establish a baseline before testing scaling, nonlinear models, or sweeps. | ran logistic_regression; signal=mixed | accuracy 0.9510 | observation only; no robust claim |
| run_002 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9790 | observation only; no robust claim |
| run_003 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9860 | prediction miss logged; no robust claim |
| run_004 | Use random forest to test whether nonlinear interactions improve over the linear baseline. | ran random_forest; signal=mixed | accuracy 0.9580 | observation only; no robust claim |
| run_005 | Verify the top observed models on matched seeds before allowing a best-model claim. | verified top models: logistic_regression_scaled_C=1.0, logistic_regression_scaled_C=0.1, random_forest_n=100; signal=mixed | best mean accuracy 0.9706 | blocked: logistic_regression_scaled_C=1.0 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |
| run_006 | Run a robustness sweep before allowing a best-hyperparameter claim. | ran multi-seed logistic_regression C sweep; signal=mixed | best mean accuracy 0.9615 | blocked: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise. |
| run_007 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9720 | observation only; no robust claim |
