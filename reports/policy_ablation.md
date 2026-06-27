# Policy Ablation

This ablation is computed from the real benchmark traces. It contrasts what the current adaptive loop logs against simpler policies over the same observed evidence.

| Task | Policy | Best single-run | Top-model verification | Unsupported best-model claims | Claims blocked | Supported claims | Prediction misses |
|---|---|---:|---|---:|---:|---:|---:|
| breast_cancer_accuracy | fixed_order | 0.9860 | yes | 0 | 2 | 0 | 1 |
| breast_cancer_accuracy | score_chaser | 0.9860 | no | 1 | 0 | 0 | 1 |
| breast_cancer_accuracy | lucky_loop_adaptive | 0.9860 | yes | 0 | 2 | 0 | 1 |
| wine_accuracy | fixed_order | 1.0000 | yes | 0 | 2 | 0 | 5 |
| wine_accuracy | score_chaser | 1.0000 | no | 1 | 0 | 0 | 5 |
| wine_accuracy | lucky_loop_adaptive | 1.0000 | yes | 0 | 2 | 0 | 5 |
| digits_accuracy | fixed_order | 0.9778 | yes | 0 | 1 | 0 | 1 |
| digits_accuracy | score_chaser | 0.9778 | no | 1 | 0 | 0 | 1 |
| digits_accuracy | lucky_loop_adaptive | 0.9778 | yes | 0 | 1 | 0 | 1 |
