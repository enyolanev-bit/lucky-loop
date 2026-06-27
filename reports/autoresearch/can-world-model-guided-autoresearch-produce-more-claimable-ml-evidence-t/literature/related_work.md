# Related Work Context

Research question: Can world-model-guided autoresearch produce more claimable ML evidence than classic autoresearch?

## Search Queries

- Can world-model-guided autoresearch produce more claimable ML evidence than classic autoresearch?
- autonomous research agents AI Scientist claim verification
- MLE-bench machine learning agents experiment search
- language world models for agents Qwen-AgentWorld
- AI scientist hallucinated findings reproducibility verification
- automated machine learning agents claim calibration

## Sources

### Qwen-AgentWorld: Language World Models for General Agents

- Authors: Yuxin Zuo, Zikai Xiao, Li Sheng, Fei Huang
- Year: 2026
- URL: http://arxiv.org/abs/2606.24597v1
- Source: arxiv
- Used for: world_model_framing
- Relevance score: 6.0

A world model predicts environment dynamics based on current observations and actions, serving as a core cognitive mechanism for reasoning and planning. In this work, we investigate how world modeling based on language models can further push the boundaries of general agents. (i) We first focus on building foundation models for agentic environment simulation. We introduce Qwen-AgentWorld-35B-A3B and Qwen-AgentWorld-397B-A17B, the first language world models capable of simulating agentic environments covering 7 domains via long chain-of-thought reasoning. Leveraging more than 10M environment interaction trajectories of 7 domains in real-world environments, we develop Qwen-AgentWorld through a three-stage training pipeline: CPT injects general-purpose world modeling capabilities from the state transition dynamics and augmented professional corpora, SFT activates next-state-prediction reasoning, and RL sharpens simulation fidelity through a tailored framework with hybrid rubric-and-rule rewards. To evaluate language world models, we present AgentWorldBench, a comprehensive benchmark constructed from real-world interactions of 5 frontier models on 9 established benchmarks. Empirical results demonstrate that Qwen-AgentWorld significantly outperforms existing frontier models. (ii) Beyond foundation models, we further investigate two complementary paradigms through which world modeling enhances general agents. First, as a decoupled environment simulator, Qwen-AgentWorld supports scalable and controllable simulation of thousands of real-world environments for agentic RL, yielding gains that surpass real-environment training alone. Second, as a unified agent foundation model, world-model training acts as a highly effective warm-up that improves downstream performance across 7 agentic benchmarks. Code: https://github.com/QwenLM/Qwen-AgentWorld

### Qwen-AgentWorld: Language World Models for General Agents

- Authors: Qwen Team
- Year: 2026
- URL: https://arxiv.org/abs/2606.24597
- Source: curated
- Used for: world_model_framing
- Relevance score: 5.0

Frames language world models as predictors of environment dynamics from state and action. Lucky Loop applies this idea to experiment outcomes.

### AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench

- Authors: Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Rishi Hazra
- Year: 2025
- URL: http://arxiv.org/abs/2507.02554v2
- Source: arxiv
- Used for: ml_agent_baseline
- Relevance score: 5.0

AI research agents are demonstrating great potential to accelerate scientific progress by automating the design, implementation, and training of machine learning models. We focus on methods for improving agents' performance on MLE-bench, a challenging benchmark where agents compete in Kaggle competitions to solve real-world machine learning problems. We formalize AI research agents as search policies that navigate a space of candidate solutions, iteratively modifying them using operators. By designing and systematically varying different operator sets and search policies (Greedy, MCTS, Evolutionary), we show that their interplay is critical for achieving high performance. Our best pairing of search strategy and operator set achieves a state-of-the-art result on MLE-bench lite, increasing the success rate of achieving a Kaggle medal from 39.6% to 47.7%. Our investigation underscores the importance of jointly considering the search strategy, operator design, and evaluation methodology in advancing automated machine learning.

### AI Scientists Fail Without Strong Implementation Capability

- Authors: AI scientist critique authors
- Year: 2025
- URL: https://arxiv.org/abs/2506.01372
- Source: curated
- Used for: autonomous_research_baseline, claim_risk
- Relevance score: 4.0

Argues that autonomous scientist systems can fail when implementation quality, experimental validity, and claim discipline are weak.

### Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper

- Authors: Atsuyuki Miyai, Mashiro Toyooka, Takashi Otonari, Zaiying Zhao
- Year: 2025
- URL: http://arxiv.org/abs/2511.04583v4
- Source: arxiv
- Used for: autonomous_research_baseline
- Relevance score: 4.0

Understanding the current capabilities and risks of AI Scientist systems (autoresearch) is essential for ensuring trustworthy and sustainable AI-driven scientific progress while preserving the integrity of the academic ecosystem. To this end, we develop Jr. AI Scientist, a state-of-the-art autonomous AI scientist system that mimics the core research workflow of a novice student researcher: Given the baseline paper from the human mentor, it analyzes its limitations, formulates novel hypotheses for improvement, iteratively experiments until improvements are achieved, and writes a paper with the results. Unlike previous approaches that assume full automation or operate on small-scale code, Jr. AI Scientist follows a well-defined research workflow and leverages modern coding agents to handle complex, multi-file implementations, leading to scientifically valuable contributions. Through our experiments, the Jr. AI Scientist successfully generated new research papers that build upon real NeurIPS, IJCV, and ICLR works by proposing and implementing novel methods. For evaluation, we conducted automated assessments using AI Reviewers, author-led evaluations, and submissions to Agents4Science, a venue dedicated to AI-driven contributions. The findings demonstrate that Jr. AI Scientist generates papers receiving higher review scores by DeepReviewer than existing fully automated systems. Nevertheless, we identify important limitations from the author evaluation and the Agents4Science reviews, indicating the potential risks of directly applying current AI Scientist systems and key challenges for future research. Finally, we comprehensively report various risks identified during development. We believe this study clarifies the current role and limitations of AI Scientist systems, offering insights into the areas that still require human expertise and the risks that may emerge as these systems evolve.

### Agent Laboratory: Using LLM Agents as Research Assistants

- Authors: Agent Laboratory authors
- Year: 2025
- URL: https://arxiv.org/abs/2501.04227
- Source: curated
- Used for: autonomous_research_baseline
- Relevance score: 3.0

A multi-agent research assistant workflow covering literature review, experiments, and report writing.

### AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench

- Authors: MLE-bench research agents authors
- Year: 2025
- URL: https://arxiv.org/abs/2507.02554
- Source: curated
- Used for: ml_agent_baseline
- Relevance score: 3.0

Studies search and exploration behavior of AI research agents for machine learning tasks.

### Language-conditioned world model improves policy generalization by reading environmental descriptions

- Authors: Anh Nguyen, Stefan Lee
- Year: 2025
- URL: http://arxiv.org/abs/2511.22904v1
- Source: arxiv
- Used for: world_model_framing
- Relevance score: 3.0

To interact effectively with humans in the real world, it is important for agents to understand language that describes the dynamics of the environment--that is, how the environment behaves--rather than just task instructions specifying "what to do". Understanding this dynamics-descriptive language is important for human-agent interaction and agent behavior. Recent work address this problem using a model-based approach: language is incorporated into a world model, which is then used to learn a behavior policy. However, these existing methods either do not demonstrate policy generalization to unseen games or rely on limiting assumptions. For instance, assuming that the latency induced by inference-time planning is tolerable for the target task or expert demonstrations are available. Expanding on this line of research, we focus on improving policy generalization from a language-conditioned world model while dropping these assumptions. We propose a model-based reinforcement learning approach, where a language-conditioned world model is trained through interaction with the environment, and a policy is learned from this model--without planning or expert demonstrations. Our method proposes Language-aware Encoder for Dreamer World Model (LED-WM) built on top of DreamerV3. LED-WM features an observation encoder that uses an attention mechanism to explicitly ground language descriptions to entities in the observation. We show that policies trained with LED-WM generalize more effectively to unseen games described by novel dynamics and language compared to other baselines in several settings in two environments: MESSENGER and MESSENGER-WM.To highlight how the policy can leverage the trained world model before real-world deployment, we demonstrate the policy can be improved through fine-tuning on synthetic test trajectories generated by the world model.

### The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery

- Authors: Sakana AI
- Year: 2024
- URL: https://arxiv.org/abs/2408.06292
- Source: curated
- Used for: autonomous_research_baseline
- Relevance score: 3.0

An autonomous scientific discovery system that generates ideas, writes code, runs experiments, and drafts papers. Useful as a baseline for end-to-end AI scientist workflows.

### MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering

- Authors: OpenAI
- Year: 2024
- URL: https://arxiv.org/abs/2410.07095
- Source: curated
- Used for: ml_agent_baseline
- Relevance score: 3.0

A benchmark for ML engineering agents where agents search over real machine learning experiment spaces and are evaluated by task performance.

### Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch

- Authors: Fabio Ferreira, Lucca Wobbe, Arjun Krishnakumar, Frank Hutter
- Year: 2026
- URL: http://arxiv.org/abs/2603.24647v5
- Source: arxiv
- Used for: background
- Relevance score: 2.0

The autoresearch repository enables an LLM agent to optimize hyperparameters by editing training code directly. We use it as a testbed to compare classical HPO algorithms against LLM-based methods on tuning the hyperparameters of a small language model under a fixed compute budget. When defining a fixed search space over autoresearch, classical methods such as CMA-ES and TPE consistently outperform LLM-based agents, where avoiding out-of-memory failures matters more than search diversity. Allowing the LLM to directly edit source code narrows the gap to the classical methods but does not close it, even with frontier models available at the time of writing such as Claude Opus 4.6 and Gemini 3.1 Pro Preview. We observe that LLMs struggle to track optimization state across trials. In contrast, classical methods lack the domain knowledge of LLMs. To combine the strengths of both, we introduce Centaur, a hybrid that shares CMA-ES's interpretable internal state, including mean vector, step-size, and covariance matrix, with an LLM. Centaur achieves the best result in our experiments, and a 0.8B LLM already suffices to outperform all classical and pure LLM methods. Unconstrained code editing requires larger models to be competitive with classical methods. We further analyze search diversity, model scaling from 0.8B to frontier models, and ablate the fraction of LLM-proposed trials in Centaur. All in all, our results suggest that LLMs are most effective as a complement to classical optimizers, not as a replacement. Code is available at https://github.com/ferreirafabio/autoresearch-automl & interactive demo at https://ferreirafabio.github.io/autoresearch-automl.

### How do Humans Process AI-generated Hallucination Contents: a Neuroimaging Study

- Authors: Shuqi Zhu, Yi Zhong, Ziyi Ye, Bangde Du
- Year: 2026
- URL: http://arxiv.org/abs/2605.16953v2
- Source: arxiv
- Used for: background
- Relevance score: 2.0

While AI-generated hallucinations pose considerable risks, the underlying cognitive mechanisms by which humans can successfully recognize or be misled by these hallucinations remain unclear. To address this problem, this paper explores humans' neural dynamics to characterize how the brain processes hallucinated content. We record EEG signals from 27 participants while they are performing a verification task to judge the correctness of image descriptions generated by a multi-modal large language model (MLLM). Based on an averaged event-related potential (ERP) study, we reveal that multiple cognitive processes, e.g., semantic integration, inferential processing, memory retrieval, and cognitive load, exhibit distinct patterns when humans process hallucinated versus non-hallucinated content. Notably, neural responses to hallucinations that were misjudged versus correctly judged by human participants showed significant differences. This indicates that misjudged AI-generated hallucinations failed to trigger the standard neurocognitive fact verification pathway.

## Gaps Lucky Loop Tests

- Autonomous research agents are often evaluated by final score or report plausibility rather than claim calibration.
- Most ML-agent benchmarks do not measure whether the agent predicted experiment outcomes before spending compute.
- A strong single-run result can become an unsupported claim when seed variance or matched multi-seed checks are missing.
- World-model predictions are usually not logged as auditable prediction-vs-reality evidence in research-agent loops.

## Metrics Suggested By Literature

- best_single_run_score
- best_verified_mean_score
- best_claimable_score
- unsupported_best_model_claims
- prediction_interval_coverage
- prediction_miss_count
- runs_to_first_verification
- compute_per_claimable_claim

## Baselines

- classic_autoresearch
- classic_verified
- lucky_loop_full
