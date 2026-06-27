# Lucky Loop Demo Summary

Goal: Maximize validation accuracy on sklearn breast_cancer under a small compute budget, while avoiding unsupported claims.

All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: agent proposes, world model predicts, Lucky Loop runs, verifier gates claims.

| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |
|---|---|---|---|---|---|
| run_001 | A simple unscaled linear baseline should anchor the search before interventions. | Establish a baseline before testing scaling, nonlinear models, or sweeps. | ran logistic_regression; signal=mixed | accuracy 0.9510 | observation only; no robust claim |
| run_002 | Feature scaling should improve or stabilize scale-sensitive linear classification. | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9790 | observation only; no robust claim |
| run_003 | Testing a new model family can reveal whether the current best score depends on inductive bias. | SVC tests a margin-based nonlinear hypothesis after linear/tree baselines. | ran svc; signal=mixed | accuracy 0.9720 | observation only; no robust claim |
| run_004 | Testing a new model family can reveal whether the current best score depends on inductive bias. | Use random forest to test whether nonlinear interactions improve over the linear baseline. | ran random_forest; signal=mixed | accuracy 0.9580 | observation only; no robust claim |
| run_005 | Testing a new model family can reveal whether the current best score depends on inductive bias. | Use boosting to test a staged-tree alternative under the remaining budget. | ran gradient_boosting; signal=mixed | accuracy 0.9441 | observation only; no robust claim |
| run_006 | The best single-run model may not be robust across matched seeds. | Verify the top observed models on matched seeds before allowing a best-model claim. | verified top models: logistic_regression_scaled_C=0.1, svc_scaled_C=0.5_kernel=rbf, random_forest_n=100; signal=mixed | best mean accuracy 0.9636 | blocked: logistic_regression_scaled_C=0.1 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |
| run_007 | Remaining budget should consolidate evidence rather than chase another single split. | Run a robustness sweep before allowing a best-hyperparameter claim. | ran multi-seed logistic_regression C sweep; signal=mixed | best mean accuracy 0.9615 | blocked: C=0.1 had the best mean accuracy, but the effect was smaller than seed noise. |
