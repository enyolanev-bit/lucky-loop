# Classic Autoresearch vs Lucky Loop

Lucky Loop is not evaluated only by final score. The comparison asks whether the system predicted before compute, verified top models before claims, and avoided unsupported best-model reporting.

## breast_cancer_accuracy
- Classic autoresearch best single-run=0.9860, unsupported best-model claims=1, Qwen predictions=0.
- Lucky Loop full best single-run=0.9860, unsupported best-model claims=0, Qwen predictions=84, claims blocked=1, supported claims=0.
- Classic verified isolates the verifier contribution: top-model verification=yes, Qwen predictions=0.

## digits_accuracy
- Classic autoresearch best single-run=0.9778, unsupported best-model claims=1, Qwen predictions=0.
- Lucky Loop full best single-run=0.9800, unsupported best-model claims=0, Qwen predictions=84, claims blocked=1, supported claims=0.
- Classic verified isolates the verifier contribution: top-model verification=yes, Qwen predictions=0.

## wine_accuracy
- Classic autoresearch best single-run=1.0000, unsupported best-model claims=1, Qwen predictions=0.
- Lucky Loop full best single-run=1.0000, unsupported best-model claims=0, Qwen predictions=83, claims blocked=2, supported claims=0.
- Classic verified isolates the verifier contribution: top-model verification=yes, Qwen predictions=0.
