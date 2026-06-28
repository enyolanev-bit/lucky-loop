# Counterfactual Evaluation

This report executes paired choices from the same reconstructed state: Lucky Loop's Qwen-guided action versus the action a classic score-chasing policy would have taken.

- Cases: 3
- Lucky wins: 3
- Qwen choice usefulness: 100.00%

| Task | Case | Lucky action | Classic action | Lucky metric | Classic metric | Score verdict | Claim-safety verdict | Overall | Reason |
|---|---|---|---|---:|---:|---|---|---|---|
| breast_cancer_accuracy | case_001 | top_model_verification | random_forest | 0.9706 | 0.9580 | lucky_win | lucky_win | lucky_win | Lucky choice ran verification and prevented a robust claim that classic score-chasing would leave unsupported. |
| wine_accuracy | case_001 | top_model_verification | logistic_regression | 0.9956 | 0.9778 | lucky_win | lucky_win | lucky_win | Lucky choice ran verification and prevented a robust claim that classic score-chasing would leave unsupported. |
| digits_accuracy | case_001 | top_model_verification | logistic_regression | 0.9818 | 0.9644 | lucky_win | lucky_win | lucky_win | Lucky choice ran verification and prevented a robust claim that classic score-chasing would leave unsupported. |

## Interpretation

A Lucky win means the Qwen-guided choice produced trusted evidence or prevented an unsupported claim when the classic policy would have continued score chasing. It does not necessarily mean Lucky won the immediate raw-score comparison.
