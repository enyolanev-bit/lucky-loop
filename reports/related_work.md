# Related Work — drop-in for paper.md

> Positioning from a verified GitHub/landscape survey (June 2026). Every repo below is public and
> URL-checked. The point: existing systems verify **post-hoc**; none gate experiment selection on a
> *learned world model* AND gate claims on measured effect-vs-noise. That combination is the open lane.

## Autonomous research agents

Systems that automate the research pipeline (idea → code → experiment → paper).

- **SakanaAI/AI-Scientist** and **AI-Scientist-v2** (github.com/SakanaAI/AI-Scientist, /AI-Scientist-v2):
  end-to-end discovery; v2 uses agentic tree search over experiments. The strongest baseline. It runs
  experiments and then writes up; verification is an LLM reviewer reviewing its own work.
- **SamuelSchmidgall/AgentLaboratory**, **HKUDS/AI-Researcher**, **microsoft/RD-Agent**
  (top MLE-bench performer): multi-stage research/R&D loops that rank ideas from prior eval feedback.
- **Just-Curieous/Curie**: framework whose explicit pitch is rigor/reproducibility — the closest prior
  work to "verify before claim", but it does not predict outcomes before compute.

**How Lucky Loop differs:** these select experiments without a predictive model of the outcome (blind
or LLM-judged), and their claim check is self-review or re-execution. Lucky Loop inserts a *world
model* between planning and execution (predict before compute) and a *deterministic* verifier between
result and claim (effect-vs-noise, not LLM self-judgment).

## World models for agents

Models that predict outcomes from state + action.

- **QwenLM/Qwen-AgentWorld** (github.com/QwenLM/Qwen-AgentWorld; arXiv:2606.24597; weights
  `Qwen/Qwen-AgentWorld-35B-A3B`): native language world model predicting next agent-environment state.
  The model we use.
- **kyle8581/WMA-Agents** (ICLR 2025, "Web Agents with World Models"), **Ber666/llm-reasoners**
  (RAP, EMNLP 2023, LLM-as-world-model + MCTS), **danijar/dreamerv3** (Nature 2025, imagination
  rollouts), **facebookresearch/vjepa2** (V-JEPA 2-AC, action-conditioned world model for planning).

**How Lucky Loop differs:** prior LLM/RL world models predict *environment dynamics* (web pages,
games, robot states) to pick actions. Lucky Loop applies a world model to *experimental research*:
the state is the evidence base, the action is a candidate ML experiment, the prediction is a
metric/runtime/risk forecast — and crucially we **measure when that prediction can be trusted**
(calibration is regime-bound, R2/R6) rather than assuming it.

## ML experiment agents and benchmarks

- **openai/mle-bench** (75 Kaggle tasks), **METR/RE-Bench** (open-ended ML R&D vs human experts),
  **snap-stanford/MLAgentBench**: eval harnesses with human/medal baselines. These grade agents but
  do not model outcomes in advance.

**How Lucky Loop differs / future work:** these are the objective functions to *prove* the thesis at
scale — measure compute saved and score delta with vs without world-model guidance. Lucky Loop's
present evidence is on small tabular benchmarks; MLE-bench/RE-Bench are the next target.

## The gap Lucky Loop fills

Across all surveyed public systems (June 2026), none combines:

1. a **learned world model** that predicts experiment outcomes *before* compute is spent, to guide
   which experiment runs next, and
2. a **calibrated claim gate** that admits a finding only on measured effect-vs-noise, independent of
   the model's (overconfident) self-assessment.

Lucky Loop is, to our knowledge, the first to do both, and to **measure the boundary**: the world
model helps in proportion to its calibration (+54% compute saved on familiar regimes, −45% on novel),
which is exactly why the verifier is a necessary safety layer rather than an optional extra.

## References (verified)

- The AI Scientist / v2 — github.com/SakanaAI/AI-Scientist, /AI-Scientist-v2
- Agent Laboratory — github.com/SamuelSchmidgall/AgentLaboratory
- RD-Agent — github.com/microsoft/RD-Agent
- Curie — github.com/Just-Curieous/Curie
- Qwen-AgentWorld — github.com/QwenLM/Qwen-AgentWorld, arXiv:2606.24597
- WMA-Agents (ICLR 2025) — github.com/kyle8581/WMA-Agents
- LLM-Reasoners / RAP (EMNLP 2023) — github.com/Ber666/llm-reasoners
- DreamerV3 (Nature 2025) — github.com/danijar/dreamerv3
- V-JEPA 2 — github.com/facebookresearch/vjepa2
- MLE-bench — github.com/openai/mle-bench
- RE-Bench (METR) — github.com/METR/RE-Bench
- MLAgentBench — github.com/snap-stanford/MLAgentBench
