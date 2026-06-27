# Lucky Loop Demo Summary

Goal: Maximize validation accuracy on sklearn digits under a small compute budget, while avoiding unsupported claims.

All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: agent proposes, world model predicts, Lucky Loop runs, verifier gates claims.

| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |
|---|---|---|---|---|---|
| run_001 | A simple unscaled linear baseline should anchor the search before interventions. | logistic regression on unscaled 64-dimensional pixel features typically yields lower accuracy compared to scaled variants; consider setting scale=true | ran logistic_regression; signal=mixed | accuracy 0.9622 | prediction miss logged; no robust claim |
| run_002 | Feature scaling should improve or stabilize scale-sensitive linear classification. | feature scaling is relevant for logistic regression and may yield a small accuracy gain over unscaled features | ran logistic_regression; signal=mixed | accuracy 0.9711 | observation only; no robust claim |
| run_003 | A prediction miss should trigger exploration of a different inductive bias. | SVC with RBF kernel and scaling is competitive on digits; expect accuracy near 0.97-0.98 but verify across seeds to confirm robustness. | ran svc; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_004 | A prediction miss should trigger exploration of a different inductive bias. | logistic regression with scaling and C=1.0 is expected to perform similarly to or slightly better than C=0.1, given the digits dataset's well-behaved pixel features | ran logistic_regression; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_005 | A prediction miss should trigger exploration of a different inductive bias. | feature scaling is relevant for logistic regression and this action includes it; previous scaled logistic regression runs achieved 0.9777 accuracy with C=1.0 | ran logistic_regression; signal=mixed | accuracy 0.9644 | prediction miss logged; no robust claim |
| run_006 | The best single-run model may not be robust across matched seeds. | top_model_verification: expecting tied or near-tied performance across seeds for svc and logistic_regression, with no clear robust separation | verified top models: svc_scaled_C=0.5_kernel=rbf, logistic_regression_scaled_C=1.0, logistic_regression_scaled_C=0.1; signal=mixed | best mean accuracy 0.9764 | blocked: svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |
| run_007 | A prediction miss should trigger exploration of a different inductive bias. | SVC with RBF kernel and feature scaling is competitive on digits; C=2.0 is within the tested range and should yield similar or slightly better accuracy than C=0.5 | ran svc; signal=mixed | accuracy 0.9800 | observation only; no robust claim |
