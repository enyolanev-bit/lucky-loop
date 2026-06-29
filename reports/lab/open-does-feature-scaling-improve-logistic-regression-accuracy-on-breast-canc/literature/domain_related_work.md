# Related Work Context

Research question: Does feature scaling improve logistic regression accuracy on breast_cancer?

## Search Protocol

- API: official arXiv Atom API (`https://export.arxiv.org/api/query`).
- Rate limit: one request every 3.1s.
- Curated fallback: known core papers are included and deduplicated against arXiv metadata.
- Citation stability: arXiv IDs and versions are preserved when available.

## Search Queries

- Does feature scaling improve logistic regression accuracy on breast_cancer?
- feature scaling improve logistic regression accuracy breast cancer machine learning classification
- feature scaling improve logistic regression accuracy breast cancer logistic regression random forest svm gradient boosting
- feature scaling improve logistic regression accuracy breast cancer robustness repeated seeds cross validation

## Source -> Gap -> Metric -> Experiment

| Gap | Sources | Metric | Experiment |
|---|---|---|---|
| The literature context must establish whether nonlinear models are actually expected to improve the selected ML task over simple baselines. | [no_source] | `domain_source_coverage` | Compare baseline and nonlinear model families on a dataset selected for the user question. |
| Claims about the selected ML task performance require robustness checks across splits or seeds, not a single score. | [no_source] | `effect_to_noise_ratio` | Run matched repeated-seed comparisons and verify effect size against seed noise. |

## Included Sources

## Excluded / Low-Relevance Sources

- None.

## Metrics Suggested By Literature

- `balanced_accuracy`
- `accuracy`
- `f1_macro`
- `precision_macro`
- `recall_macro`

## Baselines

- `literature_baseline`
- `simple_model_baseline`
