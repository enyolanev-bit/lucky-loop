# Lucky Loop Demo Summary

Goal: Maximize validation accuracy on sklearn breast_cancer under a small compute budget, while avoiding unsupported claims.

All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: agent proposes, world model predicts, Lucky Loop runs, verifier gates claims.

| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |
|---|---|---|---|---|---|
| run_001 | Classic autoresearch should verify the top observed models before a robust winner claim. | none (claim_impact=medium, compute_value=medium, rec=run) | ran logistic_regression; signal=selector_policy | accuracy 0.9510 | prediction miss logged; no robust claim |
| run_002 | Classic autoresearch should verify the top observed models before a robust winner claim. | none (claim_impact=medium, compute_value=medium, rec=run) | ran logistic_regression; signal=selector_policy | accuracy 0.9790 | prediction miss logged; no robust claim |
| run_003 | Classic autoresearch should verify the top observed models before a robust winner claim. | none (claim_impact=medium, compute_value=medium, rec=run) | ran logistic_regression; signal=selector_policy | accuracy 0.9860 | prediction miss logged; no robust claim |
| run_004 | Classic autoresearch should verify the top observed models before a robust winner claim. | none (claim_impact=medium, compute_value=medium, rec=run) | ran logistic_regression; signal=selector_policy | accuracy 0.9720 | prediction miss logged; no robust claim |
| run_005 | Classic autoresearch should verify the top observed models before a robust winner claim. | none (claim_impact=medium, compute_value=medium, rec=run) | ran svc; signal=selector_policy | accuracy 0.9720 | prediction miss logged; no robust claim |
| run_006 | Classic autoresearch should verify the top observed models before a robust winner claim. | none (claim_impact=medium, compute_value=medium, rec=run) | verified top models: logistic_regression_scaled_C=1.0, logistic_regression_scaled_C=0.1, logistic_regression_scaled_C=10.0; signal=selector_policy | best mean accuracy 0.9706 | blocked: logistic_regression_scaled_C=1.0 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |
| run_007 | Classic autoresearch should verify the top observed models before a robust winner claim. | none (claim_impact=medium, compute_value=medium, rec=run) | ran random_forest; signal=selector_policy | accuracy 0.9580 | prediction miss logged; no robust claim |
