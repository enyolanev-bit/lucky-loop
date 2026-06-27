# Lucky Loop Demo Summary

Goal: Maximize validation accuracy on sklearn wine under a small compute budget, while avoiding unsupported claims.

All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: predict, run, compare, verify.

| Run | World model said | Agent did | Reality showed | Claim verdict |
|---|---|---|---|---|
| run_001 | Establish a baseline before testing scaling, nonlinear models, or sweeps. | ran logistic_regression; signal=mixed | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_002 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 1.0000 | prediction miss logged; no robust claim |
| run_003 | Use random forest to test whether nonlinear interactions improve over the linear baseline. | ran random_forest; signal=mixed | accuracy 1.0000 | prediction miss logged; no robust claim |
| run_004 | SVC tests a margin-based nonlinear hypothesis after linear/tree baselines. | ran svc; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_005 | Use boosting to test a staged-tree alternative under the remaining budget. | ran gradient_boosting; signal=mixed | accuracy 0.9556 | observation only; no robust claim |
| run_006 | Verify the top observed models on matched seeds before allowing a best-model claim. | verified top models: logistic_regression_scaled, random_forest_n=300, logistic_regression; signal=mixed | best mean accuracy 0.9956 | blocked: random_forest_n=300 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |
| run_007 | Run a robustness sweep before allowing a best-hyperparameter claim. | ran multi-seed logistic_regression C sweep; signal=mixed | best mean accuracy 0.9667 | blocked: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise. |
