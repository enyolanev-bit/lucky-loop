# Budgeted Compute Evaluation

This report holds the run budget constant and asks whether Lucky Loop spends less compute on non-claimable score chasing. It does not claim lower total runtime when Lucky Loop chooses multi-seed verification.

## Summary

- Tasks: 3
- Tasks with saved score-chasing runs: 3
- Total saved score-chasing runs: 6
- Total saved score-chasing runtime: 5.4781s
- Tasks where Qwen would skip/stop after verifier: 3
- Strict stop policy saved runs after verification: 3
- Strict stop policy saved runtime after verification: 28.1262s

## Paired Budget Results

| Task | Budget | Classic wasted runs | Lucky wasted runs | Saved score-chasing runs | Saved score-chasing runtime | Qwen verification | Qwen stop/skip | Stop saved runs | Stop saved runtime |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| breast_cancer_accuracy | 7 | 2 | 0 | 2 | 0.7771s | yes | stop_and_report | 1 | 0.2607s |
| wine_accuracy | 7 | 2 | 0 | 2 | 0.7853s | yes | stop_and_report | 1 | 27.7334s |
| digits_accuracy | 7 | 2 | 0 | 2 | 3.9157s | yes | stop_and_report | 1 | 0.1321s |

## Policy Rows

| Task | Policy | Runs | Non-claimable runs | Non-claimable runtime | Runs after verification needed | Wasted score-chasing runs | Wasted score-chasing runtime | Compute / claimable claim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| breast_cancer_accuracy | classic_autoresearch | 7 | 7 | 1.3669s | 2 | 2 | 0.7771s | ∞ |
| breast_cancer_accuracy | lucky_loop_full | 7 | 7 | 40.6264s | 1 | 0 | 0.0000s | ∞ |
| wine_accuracy | classic_autoresearch | 7 | 7 | 3.4770s | 2 | 2 | 0.7853s | ∞ |
| wine_accuracy | lucky_loop_full | 7 | 7 | 70.0864s | 1 | 0 | 0.0000s | ∞ |
| digits_accuracy | classic_autoresearch | 7 | 7 | 9.5469s | 2 | 2 | 3.9157s | ∞ |
| digits_accuracy | lucky_loop_full | 7 | 7 | 49.7456s | 1 | 0 | 0.0000s | ∞ |

## Interpretation

A saved score-chasing run is a model run taken after the evidence state already required top-model verification. Lucky Loop's advantage here is compute allocation: it spends budget on claim risk instead of continuing non-claimable leaderboard search.

`stop_and_report` is evaluated as a strict best-model-claim mode: after an inconclusive verifier result, the system should stop making robust winner claims and skip remaining score-chasing budget unless the research objective changes.
