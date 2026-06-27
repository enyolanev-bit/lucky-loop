# Lucky Loop Demo Summary

Goal: Maximize validation accuracy on sklearn breast_cancer under a small compute budget, while avoiding unsupported claims.

All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: agent proposes, world model predicts, Lucky Loop runs, verifier gates claims.

| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |
|---|---|---|---|---|---|
| run_001 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9510 | prediction miss logged; no robust claim |
| run_002 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9790 | prediction miss logged; no robust claim |
| run_003 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9860 | prediction miss logged; no robust claim |
| run_004 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9720 | prediction miss logged; no robust claim |
| run_005 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran svc; signal=selector_policy | accuracy 0.9720 | prediction miss logged; no robust claim |
| run_006 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran random_forest; signal=selector_policy | accuracy 0.9580 | prediction miss logged; no robust claim |
| run_007 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran gradient_boosting; signal=selector_policy | accuracy 0.9441 | prediction miss logged; no robust claim |
