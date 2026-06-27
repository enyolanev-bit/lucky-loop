# Lucky Loop Demo Summary

Goal: Maximize validation accuracy on sklearn digits under a small compute budget, while avoiding unsupported claims.

All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: predict, run, compare, verify.

| Run | World model said | Agent did | Reality showed | Claim verdict |
|---|---|---|---|---|
| run_001 | Establish a baseline before testing scaling, nonlinear models, or sweeps. | ran logistic_regression; signal=mixed | accuracy 0.9622 | observation only; no robust claim |
| run_002 | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_003 | SVC tests a margin-based nonlinear hypothesis after linear/tree baselines. | ran svc; signal=world_model_prediction | accuracy 0.9800 | observation only; no robust claim |
| run_004 | Use random forest to test whether nonlinear interactions improve over the linear baseline. | ran random_forest; signal=mixed | accuracy 0.9689 | observation only; no robust claim |
| run_005 | Run a robustness sweep before allowing a best-hyperparameter claim. | ran multi-seed svc C sweep; signal=mixed | best mean accuracy 0.9806 | blocked: C=2.0 had the best mean accuracy, but the effect was smaller than seed noise. |
| run_006 | Use boosting to test a staged-tree alternative under the remaining budget. | ran hist_gradient_boosting; signal=world_model_prediction | accuracy 0.9600 | observation only; no robust claim |
