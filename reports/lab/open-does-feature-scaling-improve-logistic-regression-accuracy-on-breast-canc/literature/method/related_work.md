# Related Work Context

Research question: Does feature scaling improve logistic regression accuracy on breast_cancer? auto-research method safeguards

## Search Protocol

- API: official arXiv Atom API (`https://export.arxiv.org/api/query`).
- Rate limit: one request every 3.1s.
- Curated fallback: known core papers are included and deduplicated against arXiv metadata.
- Citation stability: arXiv IDs and versions are preserved when available.

## Search Queries

- ti:"AI Scientist" OR abs:"autonomous scientific discovery"
- all:"Agent Laboratory" "research assistants"
- all:"MLE-bench" "machine learning agents"
- all:"Qwen-AgentWorld" OR all:"language world models"
- all:"claim verification" scientific claims

## Source -> Gap -> Metric -> Experiment

| Gap | Sources | Metric | Experiment |
|---|---|---|---|
| Autonomous research agents can automate idea-to-report workflows, but final reports still need explicit claim calibration. | [arxiv_2408_06292_2024], [arxiv_2501_04227_2025], [arxiv_2506_01372_2025] | `unsupported_best_model_claims` | Compare classic_autoresearch against lucky_loop_full claim ledger outcomes. |
| ML-agent benchmarks and search policies emphasize final performance more than prospective prediction before compute. | [arxiv_2410_07095_2024], [arxiv_2507_02554_2025], [arxiv_2606_24597_2026] | `prediction_interval_coverage` | Measure Qwen-AgentWorld prediction-vs-reality for every Lucky Loop candidate decision. |
| Single-run leaderboard wins can overstate robustness when top models are close across seeds. | [arxiv_2410_07095_2024], [arxiv_2506_01372_2025], [arxiv_2507_02554_2025] | `best_claimable_score` | Detect top observed models, rerun matched seeds, and compare effect size against seed noise. |
| End-to-end autoresearch needs literature context, execution, analysis, and report generation tied to auditable evidence. | [arxiv_2408_06292_2024], [arxiv_2501_04227_2025], [arxiv_2506_01372_2025] | `evidence_manifest_completeness` | Generate reports/autoresearch/<slug>/ with sources.json, research_context.json, experiment_plan.json, and evidence_manifest.json. |

## Included Sources

### [arxiv_2501_04227_2025] Agent Laboratory: Using LLM Agents as Research Assistants

- Authors: Agent Laboratory authors
- Year: 2025
- URL: https://arxiv.org/abs/2501.04227
- arXiv: 2501.04227
- Categories: n/a
- Source: curated
- Used for: autonomous_research_baseline, end_to_end_loop, literature_to_report_loop
- Relevance score: 6.0

A multi-agent research assistant workflow covering literature review, experiments, and report writing.

### [arxiv_2506_01372_2025] AI Scientists Fail Without Strong Implementation Capability

- Authors: AI scientist critique authors
- Year: 2025
- URL: https://arxiv.org/abs/2506.01372
- arXiv: 2506.01372
- Categories: n/a
- Source: curated
- Used for: autonomous_research_baseline, claim_risk, implementation_validity
- Relevance score: 6.0

Argues that autonomous scientist systems can fail when implementation quality, experimental validity, and claim discipline are weak.

### [arxiv_2606_24597_2026] Qwen-AgentWorld: Language World Models for General Agents

- Authors: Qwen Team
- Year: 2026
- URL: https://arxiv.org/abs/2606.24597
- arXiv: 2606.24597
- Categories: n/a
- Source: curated
- Used for: state_action_prediction, world_model_framing
- Relevance score: 5.0

Frames language world models as predictors of environment dynamics from state and action. Lucky Loop applies this idea to experiment outcomes.

### [arxiv_2507_02554_2025] AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench

- Authors: MLE-bench research agents authors
- Year: 2025
- URL: https://arxiv.org/abs/2507.02554
- arXiv: 2507.02554
- Categories: n/a
- Source: curated
- Used for: ml_agent_baseline, search_policy
- Relevance score: 5.0

Studies search and exploration behavior of AI research agents for machine learning tasks.

### [arxiv_2408_06292_2024] The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery

- Authors: Sakana AI
- Year: 2024
- URL: https://arxiv.org/abs/2408.06292
- arXiv: 2408.06292
- Categories: n/a
- Source: curated
- Used for: autonomous_research_baseline, end_to_end_loop
- Relevance score: 5.0

An autonomous scientific discovery system that generates ideas, writes code, runs experiments, and drafts papers. Useful as a baseline for end-to-end AI scientist workflows.

### [arxiv_2410_07095_2024] MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering

- Authors: OpenAI
- Year: 2024
- URL: https://arxiv.org/abs/2410.07095
- arXiv: 2410.07095
- Categories: n/a
- Source: curated
- Used for: ml_agent_baseline, score_based_evaluation
- Relevance score: 5.0

A benchmark for ML engineering agents where agents search over real machine learning experiment spaces and are evaluated by task performance.

## Excluded / Low-Relevance Sources

- None.

## Metrics Suggested By Literature

- `unsupported_claim_rate`
- `prediction_interval_coverage`
- `prediction_miss_count`
- `compute_per_claimable_claim`

## Baselines

- `classic_autoresearch`
- `classic_verified`
- `lucky_loop_full`
