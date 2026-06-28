# Lucky Loop: Complete ML Research Validity Lab with Claim-Calibrated Reporting

**Team Pegasus** — Paris Research Hackathon (TUM.ai x Iterate), Track 3 "Lucky Loop", June 2026.

> Short workshop-style draft. Keep the claim narrow: this is an auditable research loop, not proof of general scientific discovery.

## Abstract

Autonomous research agents can generate plausible experiments and convincing reports, but they often lack three safeguards: prospective prediction before compute, protocol awareness during action selection, and conservative claim calibration after execution. We present **Lucky Loop**, a complete computational ML research validity lab. Starting from a research question, Lucky Loop builds literature context, maps gaps to hypotheses, compiles safe experimental protocols, asks Qwen-AgentWorld to predict the next computer-lab observation before action, runs real Python ML experiments, analyzes effects, and gates every claim through a deterministic verifier and claim ledger. The contribution is not another AutoML score chaser. It is an auditable research loop that measures whether world-model foresight changes compute allocation, protocol-risk handling, and claim quality.

## 1. Introduction

Autonomous research agents promise to accelerate the loop from idea to experiment to report. Existing systems can draft hypotheses, modify code, run experiments, and summarize results. The critical risk is not only that they fail to improve a metric; it is that they produce a convincing report from weak evidence.

Lucky Loop targets that failure mode by adding foresight before compute and discipline before claim. The core idea is simple:

```text
Predict before compute. Verify before claim.
```

An autoresearch agent proposes candidate lab actions from a safe protocol catalog. Qwen-AgentWorld is used in its natural setting as a language world model for computer-lab transitions: given the current notebook state and a candidate action, it predicts the likely terminal/output/artifact observation, failure risk, protocol risk, and claim impact. Lucky Loop then runs the real ML experiment, compares prediction against reality, and only allows claims that survive a deterministic evidence gate.

Contributions:

1. A complete ML research lab pipeline from literature review to final claim ledger.
2. A Qwen-AgentWorld integration aligned with computer-use world modeling rather than scientific oracle claims.
3. Real ML validity protocols for split validity, leakage, metric misuse, seed variance, and small-data complexity.
4. A deterministic verifier and notebook trace for claim-calibrated reporting.

## 2. Related Work

**Autonomous research agents.** Systems such as The AI Scientist and Agent Laboratory automate parts of the scientific workflow: idea generation, literature review, code writing, experiments, and report drafting. Lucky Loop does not try to replace that stack. It adds a missing layer: explicit world-model prediction before compute and evidence-gated claims after execution.

**ML experiment agents.** MLE-bench and related work frame ML agents as search policies over experiment spaces. These systems are often judged by final score. Lucky Loop also asks whether the agent could predict what would happen before acting, and whether its final claims are supported by the evidence it collected.

**World models for agents.** Qwen-AgentWorld frames a language world model as a predictor of environment dynamics from state and action across computer-use domains such as terminal, software engineering, tools, web, and OS. Lucky Loop applies that idea to an ML research lab: the state is the current notebook and evidence base, the action is a protocol-compiled lab command, and the predicted observation is the expected computer-lab output, artifact pattern, failure risk, protocol risk, and claim impact.

**Verification and reproducibility.** Critiques of AI scientist systems highlight hallucinated findings, weak implementations, and benchmark misuse. Lucky Loop addresses this with logs, prediction-vs-actual comparisons, a deterministic verifier, and a claim ledger.

## 3. Method

### World Model as Computer-Lab Simulator

At step `t`, Lucky Loop maintains an explicit lab state `s_t`: research question, literature gaps, hypotheses, protocols, notebook entries, known results, budget, and unresolved risks. The autoresearch agent proposes candidate actions `a_t`, such as running a split-validity protocol, launching a leakage probe, auditing metrics under imbalance, or replicating a single-run winner.

For each candidate action, Qwen-AgentWorld predicts a next observation:

```text
o_hat_t+1 = world_model(s_t, a_t)
```

The predicted observation includes:

- terminal/output pattern
- expected artifacts
- runtime or failure risk
- protocol risk
- expected claim delta
- recommendation: run / skip / modify / verify / stop

The selector chooses an action using the world-model signal. The executor then runs real code and returns the actual observation:

```text
o_t+1 = executor(a_t)
```

The comparator measures prediction-vs-reality. The verifier gates any scientific claim derived from the result. The loop updates state:

```text
s_t+1 = update(s_t, a_t, o_hat_t+1, o_t+1, verifier_result)
```

This makes Qwen-AgentWorld a simulator of the research computer environment, not an oracle for scientific truth. Scientific evidence comes only from executed Python experiments and deterministic analysis.

### Claim-Calibrated Reporting

Lucky Loop separates experiment outcomes from scientific claims. A single high score is an observation, not a claim. For sweep results, the verifier compares measured effect against seed noise:

```text
effect_size = best_mean_metric - worst_mean_metric
seed_noise = max(best_config_seeds) - min(best_config_seeds)
effect_to_noise_ratio = effect_size / seed_noise
```

The current verifier is deliberately conservative. It is not a full statistical significance engine; it is a claim gate that prevents obvious overclaiming.

## 4. System

Lucky Loop is implemented as a small auditable pipeline:

1. **Autoresearch agent / planner API** proposes hypotheses, candidate action IDs, and the preferred next experiment.
2. **Qwen-AgentWorld simulator** predicts each candidate's likely outcome before compute.
3. **Executor** runs real sklearn experiments and returns measured metrics.
4. **Comparator** logs prediction hits, misses, and lessons.
5. **Verifier** labels claims as missing, inconclusive, weakly supported, supported, or strongly supported.
6. **Claim ledger** records blocked and allowed claims with evidence run IDs.
7. **Reporter/UI** show the timeline and only write evidence-backed claims.

Lucky Loop is agent-agnostic. The production planner mode can call an OpenAI-compatible autoresearch agent API that returns a strict `AgentDecision` over a safe action catalog. The hackathon backend also supports an agent-in-repo mode for Codex, Claude Code, OpenClaw, Hermes, or similar coding agents operating from `program.md`. For smoke tests without credentials or an attached agent, replay mode tests the same schema and validation path. The rest of the system remains unchanged: Qwen-AgentWorld is the world model, the safety selector validates catalog choices, the executor tests reality, the comparator measures prediction-vs-reality, and the verifier gates claims.

This separation is central to the design. Qwen-AgentWorld is not the research agent and not the verifier. It is the simulator used by the agent before compute.

## 5. Experiments

Current experiments use sklearn tabular benchmarks declared as task specs: `breast_cancer`, `wine`, and `digits`. Each task specifies its dataset, metric, candidate models, budget, and robustness sweep. The loop contains real runs with:

- unscaled and scaled logistic regression
- random forest
- scaled SVC
- gradient boosting
- matched multi-seed top-model verification
- noisy-label multi-seed sweep

The key demonstrated verifier case is now top-model verification. After real single-run search, Lucky Loop detects the best observed models and runs a matched multi-seed comparison before allowing a robust best-model claim. In `breast_cancer_accuracy`, the best single-run models tied at 0.9860, but the multi-seed comparison produced:

```text
best mean: logistic_regression_scaled
effect_size = 0.005594
seed_noise = 0.027972
verdict = inconclusive
```

The older hyperparameter verifier remains useful. For example, the `breast_cancer_accuracy` C sweep produced:

```text
best mean: C=0.1
effect_size = 0.020979
seed_noise = 0.027972
verdict = inconclusive
```

A normal agent might report a robust winner. Lucky Loop blocks that overclaim and allows only a weaker statement: C=0.1 had the best mean in this sweep, but the effect was smaller than seed noise.

## 6. Results To Report

The backend ablation compares three policies:

- `classic_autoresearch`: no world model, no mandatory top-model verification, best single-run score can become an unsupported claim.
- `classic_verified`: no world model, but deterministic top-model verification isolates the verifier contribution.
- `lucky_loop_full`: agent, Qwen-AgentWorld predictions before compute, comparator, verifier, and claim ledger.

The final report should include:

- prediction-vs-actual timeline
- metric interval coverage
- runtime interval coverage
- prediction misses
- useful world-model decisions
- claim ledger summary
- supported, weakly supported, and blocked claims

The important result is not that the world model is always correct. The important result is that Lucky Loop measures when it is correct, exposes when it is wrong, and prevents unsupported claims from entering the final report.

## 7. Limitations

- The current benchmark is small and sklearn-based.
- The verifier is conservative and simple; it is not a replacement for statistical testing.
- Qwen-AgentWorld is not fine-tuned for this exact experimental environment.
- The system demonstrates an auditable trust layer, not fully automated open-ended science.
- Live Qwen-AgentWorld is the world-model demo path. The planner API path is separated from Qwen-AgentWorld and can be replaced by an agent-in-repo operator or an API planner without changing the evidence protocol.

## 8. Conclusion

Lucky Loop wraps autoresearch with foresight and discipline. An agent proposes experiments, Qwen-AgentWorld predicts what should happen, real code tests the prediction, and only evidence-backed claims survive. The result is not just an agent that sounds like a scientist, but a loop that is forced to behave like one: predict, test, compare, and claim only what the evidence supports.

## References

- The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery, 2024.
- Agent Laboratory: Using LLM Agents as Research Assistants, 2025.
- MLE-bench, 2024.
- AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench, 2025.
- Qwen-AgentWorld: Language World Models for General Agents, 2026.
- AI Scientists Fail Without Strong Implementation Capability, 2025.
