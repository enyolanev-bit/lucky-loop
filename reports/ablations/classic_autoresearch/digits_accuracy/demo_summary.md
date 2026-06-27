# Lucky Loop Demo Summary

Goal: Maximize validation accuracy on sklearn digits under a small compute budget, while avoiding unsupported claims.

All rows are real sklearn executions or real multi-seed sweeps. The table summarizes the auditable loop: agent proposes, world model predicts, Lucky Loop runs, verifier gates claims.

| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |
|---|---|---|---|---|---|
| run_001 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9622 | prediction miss logged; no robust claim |
| run_002 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9711 | prediction miss logged; no robust claim |
| run_003 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_004 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9644 | prediction miss logged; no robust claim |
| run_005 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran svc; signal=selector_policy | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_006 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran random_forest; signal=selector_policy | accuracy 0.9600 | prediction miss logged; no robust claim |
| run_007 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran hist_gradient_boosting; signal=selector_policy | accuracy 0.9578 | prediction miss logged; no robust claim |
