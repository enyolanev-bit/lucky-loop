# Related Work Context

Research question: UI smoke test autoresearch venv launch auto-research method safeguards

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
| Autonomous research agents can automate idea-to-report workflows, but final reports still need explicit claim calibration. | [arxiv_2408_06292_2024], [arxiv_2501_04227_2025], [arxiv_2506_01372_2025], [arxiv_2605_08956_2026] | `unsupported_best_model_claims` | Compare classic_autoresearch against lucky_loop_full claim ledger outcomes. |
| ML-agent benchmarks and search policies emphasize final performance more than prospective prediction before compute. | [arxiv_2410_07095_2024], [arxiv_2507_02554_2025], [arxiv_2605_08956_2026], [arxiv_2606_24597_2026] | `prediction_interval_coverage` | Measure Qwen-AgentWorld prediction-vs-reality for every Lucky Loop candidate decision. |
| Single-run leaderboard wins can overstate robustness when top models are close across seeds. | [arxiv_2410_07095_2024], [arxiv_2506_01372_2025], [arxiv_2507_02554_2025], [arxiv_2605_08956_2026] | `best_claimable_score` | Detect top observed models, rerun matched seeds, and compare effect size against seed noise. |
| End-to-end autoresearch needs literature context, execution, analysis, and report generation tied to auditable evidence. | [arxiv_2408_06292_2024], [arxiv_2501_04227_2025], [arxiv_2506_01372_2025], [arxiv_2605_08956_2026] | `evidence_manifest_completeness` | Generate reports/autoresearch/<slug>/ with sources.json, research_context.json, experiment_plan.json, and evidence_manifest.json. |

## Included Sources

### [arxiv_2605_08956_2026] Agentic AI Scientists Are Not Built For Autonomous Scientific Discovery

- Authors: Harshit Bisht, Vinay Kumar, Kevin Maik Jablonka, Mausam
- Year: 2026
- URL: https://arxiv.org/abs/2605.08956v1
- arXiv: 2605.08956v1
- Categories: cs.AI
- Source: arxiv
- Used for: autonomous_research_baseline, claim_risk, world_model_framing
- Relevance score: 6.0

A growing body of work pursues AI scientists capable of end-to-end autonomous scientific discovery. This position paper argues that although they already function as co-scientists, agentic AI scientists are not built for autonomous scientific discovery. We identify the following challenges in building and deploying autonomous AI scientists: (1) Problem selection is influenced by the McNamara fallacy; (2) Agents are built on large language models (LLMs) whose training corpora omit tacit procedural and failure knowledge of laboratory practice; (3) Preference optimisation during post-training compresses output diversity toward consensus; and (4) Most scientific benchmarks measure single-turn prediction accuracy and lack feedback from physical experiments back to the computational model. These challenges are not just questions of scale and scaffolding; they require revisiting fundamental design choices. To build truly autonomous AI scientists, we recommend the use of scientific simulations as verifiers for training, the design of persistent world models that represent the shifting objectives governing real investigations, the establishment of a centralized preregistration repository for all AI-generated hypotheses, and application driven by scientific need rather than tool affordance.

### [arxiv_2501_04227_2025] Agent Laboratory: Using LLM Agents as Research Assistants

- Authors: Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun
- Year: 2025
- URL: https://arxiv.org/abs/2501.04227v2
- arXiv: 2501.04227v2
- Categories: cs.AI, cs.CL, cs.HC, cs.LG
- Source: arxiv+curated
- Used for: autonomous_research_baseline, end_to_end_loop, literature_to_report_loop
- Relevance score: 6.0

Historically, scientific discovery has been a lengthy and costly process, demanding substantial time and resources from initial conception to final results. To accelerate scientific discovery, reduce research costs, and improve research quality, we introduce Agent Laboratory, an autonomous LLM-based framework capable of completing the entire research process. This framework accepts a human-provided research idea and progresses through three stages--literature review, experimentation, and report writing to produce comprehensive research outputs, including a code repository and a research report, while enabling users to provide feedback and guidance at each stage. We deploy Agent Laboratory with various state-of-the-art LLMs and invite multiple researchers to assess its quality by participating in a survey, providing human feedback to guide the research process, and then evaluate the final paper. We found that: (1) Agent Laboratory driven by o1-preview generates the best research outcomes; (2) The generated machine learning code is able to achieve state-of-the-art performance compared to existing methods; (3) Human involvement, providing feedback at each stage, significantly improves the overall quality of research; (4) Agent Laboratory significantly reduces research expenses, achieving an 84% decrease compared to previous autonomous research methods. We hope Agent Laboratory enables researchers to allocate more effort toward creative ideation rather than low-level coding and writing, ultimately accelerating scientific discovery.

### [arxiv_2506_01372_2025] AI Scientists Fail Without Strong Implementation Capability

- Authors: Minjun Zhu, Qiujie Xie, Yixuan Weng, Jian Wu
- Year: 2025
- URL: https://arxiv.org/abs/2506.01372v2
- arXiv: 2506.01372v2
- Categories: cs.AI, cs.CL, cs.LG
- Source: arxiv+curated
- Used for: autonomous_research_baseline, claim_risk, end_to_end_loop, implementation_validity
- Relevance score: 6.0

The emergence of Artificial Intelligence (AI) Scientist represents a paradigm shift in scientific discovery, with large language models (LLMs) taking the lead as the primary executor in the entire scientific workflow from idea generation to experiment implementation. Recent AI Scientist studies demonstrate sufficient capabilities for independent scientific discovery, with the generated research reports gaining acceptance at the ICLR 2025 workshop and ACL 2025, arguing that a human-level AI Scientist, capable of uncovering phenomena previously unknown to humans, may be imminent. Despite this substantial progress, AI Scientist has yet to produce a groundbreaking achievement in the domain of computer science on par with automated scientific tools. Based on extensive quantitative evidence from existing benchmarks in complex engineering tasks and a systematic evaluation assess 28 research papers generated by five advanced AI Scientist systems, we argue that \textbf{the fundamental bottleneck for AI Scientists lies in their capability to execute the requisite verification procedures.} Current AI Scientist systems lack the execution capabilities needed to execute rigorous experiments and produce high-quality scientific papers. To better illustrate the root cause of this \textbf{implementation gap}, we provide an in-depth discussion on the fundamental limitations of AI Scientist. This position paper aims to call for the participants in the community to bridge the implementation gap.

### [arxiv_2410_07095_2024] MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering

- Authors: Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung
- Year: 2024
- URL: https://arxiv.org/abs/2410.07095v6
- arXiv: 2410.07095v6
- Categories: cs.CL
- Source: arxiv+curated
- Used for: ml_agent_baseline, score_based_evaluation
- Relevance score: 6.0

We introduce MLE-bench, a benchmark for measuring how well AI agents perform at machine learning engineering. To this end, we curate 75 ML engineering-related competitions from Kaggle, creating a diverse set of challenging tasks that test real-world ML engineering skills such as training models, preparing datasets, and running experiments. We establish human baselines for each competition using Kaggle's publicly available leaderboards. We use open-source agent scaffolds to evaluate several frontier language models on our benchmark, finding that the best-performing setup--OpenAI's o1-preview with AIDE scaffolding--achieves at least the level of a Kaggle bronze medal in 16.9% of competitions. In addition to our main results, we investigate various forms of resource scaling for AI agents and the impact of contamination from pre-training. We open-source our benchmark code (github.com/openai/mle-bench/) to facilitate future research in understanding the ML engineering capabilities of AI agents.

### [arxiv_2606_24597_2026] Qwen-AgentWorld: Language World Models for General Agents

- Authors: Yuxin Zuo, Zikai Xiao, Li Sheng, Fei Huang
- Year: 2026
- URL: https://arxiv.org/abs/2606.24597v1
- arXiv: 2606.24597v1
- Categories: cs.CL
- Source: arxiv+curated
- Used for: state_action_prediction, world_model_framing
- Relevance score: 5.0

A world model predicts environment dynamics based on current observations and actions, serving as a core cognitive mechanism for reasoning and planning. In this work, we investigate how world modeling based on language models can further push the boundaries of general agents. (i) We first focus on building foundation models for agentic environment simulation. We introduce Qwen-AgentWorld-35B-A3B and Qwen-AgentWorld-397B-A17B, the first language world models capable of simulating agentic environments covering 7 domains via long chain-of-thought reasoning. Leveraging more than 10M environment interaction trajectories of 7 domains in real-world environments, we develop Qwen-AgentWorld through a three-stage training pipeline: CPT injects general-purpose world modeling capabilities from the state transition dynamics and augmented professional corpora, SFT activates next-state-prediction reasoning, and RL sharpens simulation fidelity through a tailored framework with hybrid rubric-and-rule rewards. To evaluate language world models, we present AgentWorldBench, a comprehensive benchmark constructed from real-world interactions of 5 frontier models on 9 established benchmarks. Empirical results demonstrate that Qwen-AgentWorld significantly outperforms existing frontier models. (ii) Beyond foundation models, we further investigate two complementary paradigms through which world modeling enhances general agents. First, as a decoupled environment simulator, Qwen-AgentWorld supports scalable and controllable simulation of thousands of real-world environments for agentic RL, yielding gains that surpass real-environment training alone. Second, as a unified agent foundation model, world-model training acts as a highly effective warm-up that improves downstream performance across 7 agentic benchmarks. Code: https://github.com/QwenLM/Qwen-AgentWorld

### [arxiv_2507_02554_2025] AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench

- Authors: Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Rishi Hazra
- Year: 2025
- URL: https://arxiv.org/abs/2507.02554v2
- arXiv: 2507.02554v2
- Categories: cs.AI, cs.LG
- Source: arxiv+curated
- Used for: ml_agent_baseline, search_policy
- Relevance score: 5.0

AI research agents are demonstrating great potential to accelerate scientific progress by automating the design, implementation, and training of machine learning models. We focus on methods for improving agents' performance on MLE-bench, a challenging benchmark where agents compete in Kaggle competitions to solve real-world machine learning problems. We formalize AI research agents as search policies that navigate a space of candidate solutions, iteratively modifying them using operators. By designing and systematically varying different operator sets and search policies (Greedy, MCTS, Evolutionary), we show that their interplay is critical for achieving high performance. Our best pairing of search strategy and operator set achieves a state-of-the-art result on MLE-bench lite, increasing the success rate of achieving a Kaggle medal from 39.6% to 47.7%. Our investigation underscores the importance of jointly considering the search strategy, operator design, and evaluation methodology in advancing automated machine learning.

### [arxiv_2408_06292_2024] The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery

- Authors: Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster
- Year: 2024
- URL: https://arxiv.org/abs/2408.06292v3
- arXiv: 2408.06292v3
- Categories: cs.AI, cs.CL, cs.LG
- Source: arxiv+curated
- Used for: autonomous_research_baseline, end_to_end_loop
- Relevance score: 5.0

One of the grand challenges of artificial general intelligence is developing agents capable of conducting scientific research and discovering new knowledge. While frontier models have already been used as aides to human scientists, e.g. for brainstorming ideas, writing code, or prediction tasks, they still conduct only a small part of the scientific process. This paper presents the first comprehensive framework for fully automatic scientific discovery, enabling frontier large language models to perform research independently and communicate their findings. We introduce The AI Scientist, which generates novel research ideas, writes code, executes experiments, visualizes results, describes its findings by writing a full scientific paper, and then runs a simulated review process for evaluation. In principle, this process can be repeated to iteratively develop ideas in an open-ended fashion, acting like the human scientific community. We demonstrate its versatility by applying it to three distinct subfields of machine learning: diffusion modeling, transformer-based language modeling, and learning dynamics. Each idea is implemented and developed into a full paper at a cost of less than $15 per paper. To evaluate the generated papers, we design and validate an automated reviewer, which we show achieves near-human performance in evaluating paper scores. The AI Scientist can produce papers that exceed the acceptance threshold at a top machine learning conference as judged by our automated reviewer. This approach signifies the beginning of a new era in scientific discovery in machine learning: bringing the transformative benefits of AI agents to the entire research process of AI itself, and taking us closer to a world where endless affordable creativity and innovation can be unleashed on the world's most challenging problems. Our code is open-sourced at https://github.com/SakanaAI/AI-Scientist

### [arxiv_2405_13352_2024] "Turing Tests" For An AI Scientist

- Authors: Xiaoxin Yin
- Year: 2024
- URL: https://arxiv.org/abs/2405.13352v1
- arXiv: 2405.13352v1
- Categories: cs.AI
- Source: arxiv
- Used for: autonomous_research_baseline
- Relevance score: 5.0

While LLMs have shown impressive capabilities in solving math or coding problems, the ability to make scientific discoveries remains a distinct challenge. This paper proposes a "Turing test for an AI scientist" to assess whether an AI agent can conduct scientific research independently, without relying on human-generated knowledge. Drawing inspiration from the historical development of science, we propose seven benchmark tests that evaluate an AI agent's ability to make groundbreaking discoveries in various scientific domains. These tests include inferring the heliocentric model from celestial observations, discovering the laws of motion in a simulated environment, deriving the differential equation governing vibrating strings, inferring Maxwell's equations from electrodynamics simulations, inventing numerical methods for initial value problems, discovering Huffman coding for data compression, and developing efficient sorting algorithms. To ensure the validity of these tests, the AI agent is provided with interactive libraries or datasets specific to each problem, without access to human knowledge that could potentially contain information about the target discoveries. The ultimate goal is to create an AI scientist capable of making novel and impactful scientific discoveries, surpassing the best human experts in their respective fields. These "Turing tests" serve as intermediate milestones, assessing the AI agent's ability to make discoveries that were groundbreaking in their time. If an AI agent can pass the majority of these seven tests, it would indicate significant progress towards building an AI scientist, paving the way for future advancements in autonomous scientific discovery. This paper aims to establish a benchmark for the capabilities of AI in scientific research and to stimulate further research in this exciting field.

## Excluded / Low-Relevance Sources

- Planning with Reasoning using Vision Language World Model (https://arxiv.org/abs/2509.02722v2): low relevance to autonomous research, ML agents, world models, or claim verification; score=1.0
- RerrFact: Reduced Evidence Retrieval Representations for Scientific Claim Verification (https://arxiv.org/abs/2202.02646v2): low relevance to autonomous research, ML agents, world models, or claim verification; score=1.0
- Emergent Communication with World Models (https://arxiv.org/abs/2002.09604v1): low relevance to autonomous research, ML agents, world models, or claim verification; score=1.0
- Scientific Claim Verification with VERT5ERINI (https://arxiv.org/abs/2010.11930v1): low relevance to autonomous research, ML agents, world models, or claim verification; score=1.0
- EAIRA: Establishing a Methodology for Evaluating AI Models as Scientific Research Assistants (https://arxiv.org/abs/2502.20309v1): low relevance to autonomous research, ML agents, world models, or claim verification; score=0.0
- Reconciling Methodological Paradigms: Employing Large Language Models as Novice Qualitative Research Assistants in Talent Management Research (https://arxiv.org/abs/2408.11043v1): low relevance to autonomous research, ML agents, world models, or claim verification; score=0.0

## Metrics Suggested By Literature

- `unsupported_claim_rate`
- `prediction_interval_coverage`
- `prediction_miss_count`
- `compute_per_claimable_claim`

## Baselines

- `classic_autoresearch`
- `classic_verified`
- `lucky_loop_full`
