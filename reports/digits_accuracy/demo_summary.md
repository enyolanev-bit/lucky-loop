# Lucky Loop Demo Summary

Goal: Maximize validation accuracy on sklearn digits under a small compute budget, while avoiding unsupported claims.

All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: agent proposes, world model predicts, Lucky Loop runs, verifier gates claims.

| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |
|---|---|---|---|---|---|
| run_001 | A simple unscaled linear baseline should anchor the search before interventions. | logistic regression without scaling on digits typically yields lower accuracy than scaled variants or tree-based models | ran logistic_regression; signal=mixed | accuracy 0.9622 | prediction miss logged; no robust claim |
| run_002 | Feature scaling should improve or stabilize scale-sensitive linear classification. | feature scaling is relevant for logistic regression and should be verified across seeds | ran logistic_regression; signal=mixed | accuracy 0.9711 | prediction miss logged; no robust claim |
| run_003 | A prediction miss should trigger exploration of a different inductive bias. | SVC with rbf kernel and scaling is expected to achieve competitive accuracy on digits, but robustness sweeps across seeds are needed to confirm performance stability. | ran svc; signal=mixed | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_004 | A prediction miss should trigger exploration of a different inductive bias. | logistic regression with scaling and C=1.0 is expected to be competitive with the current top SVC model, but robustness verification across seeds is needed to confirm any improvement. | ran logistic_regression; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_005 | A prediction miss should trigger exploration of a different inductive bias. | logistic regression with scaling and C=10.0 is expected to perform comparably to C=1.0, with accuracy likely in the 0.975-0.98 range; feature scaling is appropriately applied. | ran logistic_regression; signal=mixed | accuracy 0.9644 | prediction miss logged; no robust claim |
| run_006 | The best single-run model may not be robust across matched seeds. | robustness sweeps before strong claims when a high single-run score exists | verified top models: svc_scaled_C=0.5_kernel=rbf, logistic_regression_scaled_C=1.0, logistic_regression_scaled_C=0.1; signal=mixed | best mean accuracy 0.9764 | blocked: svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |
| run_007 | A prediction miss should trigger exploration of a different inductive bias. | SVC with RBF kernel and feature scaling is appropriate for the digits dataset; C=2.0 is within the tested range and should be evaluated for robustness. | ran svc; signal=mixed | accuracy 0.9800 | observation only; no robust claim |
