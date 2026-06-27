# World Model Ablation

This suite runs real sklearn experiments under three backend policies.

- `classic_autoresearch`: agent policy runs experiments and can report a best single-run score without prospective simulation.
- `classic_verified`: same no-world-model planner, but with deterministic top-model verification.
- `lucky_loop_full`: agent-in-repo planner plus Qwen-AgentWorld predictions before compute and deterministic claim verification.

| Task | Policy | Runs | Best single-run | Top-model verification | Unsupported best-model claims | Claims blocked | Supported claims | Qwen predictions | Useful WM decisions |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| breast_cancer_accuracy | classic_autoresearch | 7 | 0.9860 | no | 1 | 0 | 0 | 0 | 0 |
| breast_cancer_accuracy | classic_verified | 7 | 0.9860 | yes | 0 | 1 | 0 | 0 | 0 |
| breast_cancer_accuracy | lucky_loop_full | 7 | 0.9860 | yes | 0 | 1 | 0 | 84 | 7 |
| wine_accuracy | classic_autoresearch | 7 | 1.0000 | no | 1 | 0 | 0 | 0 | 0 |
| wine_accuracy | classic_verified | 7 | 1.0000 | yes | 0 | 1 | 0 | 0 | 0 |
| wine_accuracy | lucky_loop_full | 7 | 1.0000 | yes | 0 | 2 | 0 | 83 | 7 |
| digits_accuracy | classic_autoresearch | 7 | 0.9778 | no | 1 | 0 | 0 | 0 | 0 |
| digits_accuracy | classic_verified | 7 | 0.9778 | yes | 0 | 1 | 0 | 0 | 0 |
| digits_accuracy | lucky_loop_full | 7 | 0.9800 | yes | 0 | 1 | 0 | 84 | 7 |
