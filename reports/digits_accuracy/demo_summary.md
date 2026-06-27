# Lucky Loop Demo Summary

Goal: Maximize validation accuracy on sklearn digits under a small compute budget, while avoiding unsupported claims.

All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: predict, run, compare, verify.

| Run | World model said | Agent did | Reality showed | Claim verdict |
|---|---|---|---|---|
| run_001 | Establish a baseline before testing scaling, nonlinear models, or sweeps. | ran logistic_regression; signal=mixed | accuracy 0.9622 | observation only; no robust claim |
| run_002 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9711 | observation only; no robust claim |
| run_003 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_004 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9644 | observation only; no robust claim |
| run_005 | SVC tests a margin-based nonlinear hypothesis after linear/tree baselines. | ran svc; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_006 | Use random forest to test whether nonlinear interactions improve over the linear baseline. | ran random_forest; signal=mixed | accuracy 0.9600 | observation only; no robust claim |
| run_007 | Verify the top observed models on matched seeds before allowing a best-model claim. | verified top models: logistic_regression_scaled_C=1.0, svc_scaled_C=0.5_kernel=rbf, logistic_regression_scaled_C=0.1; signal=mixed | best mean accuracy 0.9764 | blocked: svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |
