# World Model Ablation

This suite runs real sklearn experiments under three backend policies.

- `classic_autoresearch`: agent policy runs experiments and can report a best single-run score without prospective simulation.
- `classic_verified`: same no-world-model planner, but with deterministic top-model verification.
- `lucky_loop_full`: agent-in-repo planner plus Qwen-AgentWorld predictions before compute and deterministic claim verification.

| Task | Policy | Runs | Best single-run | Best verified mean | Best claimable | Runs to verification | Compute / claimable claim | Wasted score-chasing | Unsupported claims | Claims blocked | Qwen predictions | Qwen triggered verification | Qwen choice usefulness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| breast_cancer_accuracy | classic_autoresearch | 7 | 0.9860 |  |  | ∞ | ∞ | 2 | 1 | 0 | 0 | no |  |
| breast_cancer_accuracy | classic_verified | 7 | 0.9860 | 0.9706 |  | 6 | ∞ | 0 | 0 | 1 | 0 | no |  |
| breast_cancer_accuracy | lucky_loop_full | 7 | 0.9860 | 0.9706 |  | 6 | ∞ | 0 | 0 | 1 | 84 | yes | 100.00% |
| wine_accuracy | classic_autoresearch | 7 | 1.0000 |  |  | ∞ | ∞ | 2 | 1 | 0 | 0 | no |  |
| wine_accuracy | classic_verified | 7 | 1.0000 | 0.9956 |  | 6 | ∞ | 0 | 0 | 1 | 0 | no |  |
| wine_accuracy | lucky_loop_full | 7 | 1.0000 | 0.9956 |  | 6 | ∞ | 0 | 0 | 1 | 83 | yes | 100.00% |
| digits_accuracy | classic_autoresearch | 7 | 0.9778 |  |  | ∞ | ∞ | 2 | 1 | 0 | 0 | no |  |
| digits_accuracy | classic_verified | 7 | 0.9778 | 0.9764 |  | 6 | ∞ | 0 | 0 | 1 | 0 | no |  |
| digits_accuracy | lucky_loop_full | 7 | 0.9800 | 0.9818 |  | 6 | ∞ | 0 | 0 | 1 | 84 | yes | 100.00% |

## Claimable Evidence Per Compute

`best_claimable_score` is strict: inconclusive verifier outcomes do not count. `∞` means no claim reached the trust ladder threshold.

| Task | Policy | Best claimable | Runtime | Compute / claimable claim |
|---|---|---:|---:|---:|
| breast_cancer_accuracy | classic_autoresearch |  | 1.41s | ∞ |
| breast_cancer_accuracy | classic_verified |  | 34.86s | ∞ |
| breast_cancer_accuracy | lucky_loop_full |  | 34.88s | ∞ |
| wine_accuracy | classic_autoresearch |  | 1.47s | ∞ |
| wine_accuracy | classic_verified |  | 39.60s | ∞ |
| wine_accuracy | lucky_loop_full |  | 37.31s | ∞ |
| digits_accuracy | classic_autoresearch |  | 9.79s | ∞ |
| digits_accuracy | classic_verified |  | 35.73s | ∞ |
| digits_accuracy | lucky_loop_full |  | 46.93s | ∞ |
