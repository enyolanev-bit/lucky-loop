# Classic Autoresearch vs Lucky Loop

Lucky Loop is not evaluated only by final score. The comparison asks whether the system predicted before compute, verified top models before claims, and avoided unsupported best-model reporting.

## breast_cancer_accuracy
- Classic autoresearch best single-run=0.9860, best claimable=none, unsupported best-model claims=1, wasted score-chasing runs=2, Qwen predictions=0.
- Lucky Loop full best single-run=0.9860, best verified mean=0.9706, best claimable=none, unsupported best-model claims=0, Qwen predictions=84, claims blocked=1, supported claims=0, Qwen triggered verification=yes, Qwen choice usefulness=100.00%.
- Classic verified isolates the verifier contribution: top-model verification=yes, Qwen predictions=0.

## digits_accuracy
- Classic autoresearch best single-run=0.9778, best claimable=none, unsupported best-model claims=1, wasted score-chasing runs=2, Qwen predictions=0.
- Lucky Loop full best single-run=0.9800, best verified mean=0.9764, best claimable=none, unsupported best-model claims=0, Qwen predictions=84, claims blocked=1, supported claims=0, Qwen triggered verification=yes, Qwen choice usefulness=100.00%.
- Classic verified isolates the verifier contribution: top-model verification=yes, Qwen predictions=0.

## wine_accuracy
- Classic autoresearch best single-run=1.0000, best claimable=none, unsupported best-model claims=1, wasted score-chasing runs=2, Qwen predictions=0.
- Lucky Loop full best single-run=1.0000, best verified mean=0.9956, best claimable=none, unsupported best-model claims=0, Qwen predictions=83, claims blocked=2, supported claims=0, Qwen triggered verification=yes, Qwen choice usefulness=100.00%.
- Classic verified isolates the verifier contribution: top-model verification=yes, Qwen predictions=0.
