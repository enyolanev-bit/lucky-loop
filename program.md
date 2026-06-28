# Lucky Loop Agent Program

You are the autoresearch agent operating inside this repository. You are not the world model and you are not the verifier.

## Objective

Run a complete ML research validity lab under a small compute budget while avoiding unsupported claims.

Core rule:

```text
Predict before compute. Verify before claim.
```

## Roles

- Autoresearch agent: operate the safe ML protocol catalog from literature gaps to final claims.
- Qwen-AgentWorld: language world model that predicts the next computer-lab observation, cost, protocol risk, and claim impact before compute.
- Executor: runs real Python ML experiments from the protocol catalog.
- Comparator: measures prediction-vs-reality.
- Verifier: deterministic claim gate over multi-seed evidence.
- Claim ledger: records allowed, weak, supported, and blocked claims.

## Operating Contract

Use Lucky Loop as a complete lab protocol:

1. Start from the research question.
2. Build literature context, gaps, hypotheses, and protocols.
3. Generate candidate lab actions from the safe protocol catalog.
4. Ask Qwen-AgentWorld to predict each candidate action's computer-lab observation, protocol risk, value of information, and claim impact.
5. Choose an action using observed evidence, hypothesis state, and world-model claim/cost signals.
6. Run only protocol-compiled commands.
7. Compare prediction with real ML results.
8. Analyze effects and verify claims deterministically.
9. Write notebook JSONL, claim ledger, reproducibility notes, and final report.

Never invent execution results. Never report a robust winner from a single split.

## Main Commands

Run the complete lab path:

```bash
export PYTHONPATH=src
export LUCKYWORLD_SIMULATOR_BASE_URL=http://134.199.205.222:8000/v1
export LUCKYWORLD_SIMULATOR_MODEL=Qwen/Qwen-AgentWorld-35B-A3B
export LUCKYWORLD_SIMULATOR_API_KEY=dummy

python3 -m luckyloop.lab \
  --question "Do random train/test splits overstate performance on sequential sensor datasets?" \
  --budget 8 \
  --planner llm \
  --require-agent \
  --require-qwen
```

Run the lab ablation suite:

```bash
PYTHONPATH=src python3 scripts/run_lab_ablation_suite.py --require-qwen
```

Run the world-model ablation:

```bash
PYTHONPATH=src python3 scripts/run_ablation_suite.py --world-model auto
```

The legacy benchmark/autoresearch path remains available:

```bash
PYTHONPATH=src python3 -m luckyloop.autoresearch \
  --question "Can world-model-guided autoresearch produce more claimable ML evidence than classic autoresearch?" \
  --agent codex_operator \
  --execute
```

This command creates `reports/autoresearch/<question_slug>/` with literature context, agent instructions, an experiment plan, evidence manifest, and a final report scaffold. The coding agent remains the autoresearch agent; Python does not pretend to call Codex or Claude internally.

Validate artifacts:

```bash
PYTHONPATH=src python3 scripts/validate_lab_artifacts.py --require-agent --require-qwen
PYTHONPATH=src python3 scripts/write_complete_demo.py --out reports/demo_complete_lab.md
```

## Acceptance Criteria

- Literature, gaps, hypotheses, protocols, runs, analyses, notebook, claim ledger, reproducibility notes, and final report all exist.
- Lucky Loop lab predictions contain live `qwen_agentworld` outputs when `--require-qwen` is used.
- At least one real ML experiment is executed.
- At least one overclaim is blocked, rewritten, or marked inconclusive.
- Reports expose prediction-vs-reality and only include claims backed by claim ledger entries.
