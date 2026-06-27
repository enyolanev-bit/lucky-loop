# Lucky Loop Research Report

Goal: Maximize validation accuracy on sklearn wine under a small compute budget, while avoiding unsupported claims.

## Thesis

An API-backed autoresearch planner proposes hypotheses and safe catalog actions. Qwen-AgentWorld predicts experimental consequences before compute. Lucky Loop then runs real code, compares prediction with reality, and gates scientific claims through a deterministic verifier.

## Experiment timeline

| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |
|---|---|---|---|---|---|
| run_001 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_002 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 1.0000 | prediction miss logged; no robust claim |
| run_003 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 1.0000 | prediction miss logged; no robust claim |
| run_004 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_005 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran svc; signal=selector_policy | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_006 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran random_forest; signal=selector_policy | accuracy 1.0000 | prediction miss logged; no robust claim |
| run_007 | Score-chasing autoresearch should spend budget on the next promising model family. | none | ran gradient_boosting; signal=selector_policy | accuracy 0.9556 | prediction miss logged; no robust claim |

## Autoresearch agent decisions

- run_001: preferred=action_001; backend=classic_autoresearch; evidence_needed=Collect single-run metrics and report the best observed score.
- run_002: preferred=action_logistic_regression_C-0p1_scale-True; backend=classic_autoresearch; evidence_needed=Collect single-run metrics and report the best observed score.
- run_003: preferred=action_logistic_regression_C-1p0_scale-True; backend=classic_autoresearch; evidence_needed=Collect single-run metrics and report the best observed score.
- run_004: preferred=action_logistic_regression_C-10p0_scale-True; backend=classic_autoresearch; evidence_needed=Collect single-run metrics and report the best observed score.
- run_005: preferred=action_svc_C-0p5_kernel-rbf_scale-True; backend=classic_autoresearch; evidence_needed=Collect single-run metrics and report the best observed score.
- run_006: preferred=action_random_forest_n_estimators-100; backend=classic_autoresearch; evidence_needed=Collect single-run metrics and report the best observed score.
- run_007: preferred=action_gradient_boosting_learning_rate-0p05_n_estimators-100; backend=classic_autoresearch; evidence_needed=Collect single-run metrics and report the best observed score.

## Best result

Best single run: run_002, model=logistic_regression, accuracy=1.0000, f1=1.0000.

## Top model robustness

- No top-model multi-seed verification was run.

## World model calibration

- Metric interval coverage: n/a
- Runtime interval coverage: n/a
- Prediction miss count: 7
- Useful decision signals: 0/7
- Full calibration table: `reports/world_model_calibration.md`

## Supported claims

- No claim reached supported or strongly_supported yet.

## Weakly supported claims

- No weakly supported claim was recorded.

## Blocked / inconclusive claims

- No verifier-level blocked claim was recorded.

## Claim ledger

- Full ledger: `reports/claim_ledger.json`

## Prediction misses

- run_001: no world-model prediction was made before compute
- run_002: no world-model prediction was made before compute
- run_003: no world-model prediction was made before compute
- run_004: no world-model prediction was made before compute
- run_005: no world-model prediction was made before compute
- run_006: no world-model prediction was made before compute
- run_007: no world-model prediction was made before compute

## Evidence notes

### run_001
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_001; budget_remaining=7; known_results=0
- Candidates considered: action_001:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_logistic_regression_C-0p1_scale-True:logistic_regression, action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest
- Planner decision: classic_autoresearch selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_autoresearch selected action_001 from state state_001.
- Evidence needed: Collect single-run metrics and report the best observed score.
- Claim risk: A best single-run claim can be unsupported if top models are close.
- Safety validation: selected=action_001; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 2.6147s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_002
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_002; budget_remaining=6; known_results=1
- Candidates considered: action_logistic_regression_C-0p1_scale-True:logistic_regression, action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest
- Planner decision: classic_autoresearch selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_autoresearch selected action_logistic_regression_C-0p1_scale-True from state state_002.
- Evidence needed: Collect single-run metrics and report the best observed score.
- Claim risk: A best single-run claim can be unsupported if top models are close.
- Safety validation: selected=action_logistic_regression_C-0p1_scale-True; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.024s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_003
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_003; budget_remaining=5; known_results=2
- Candidates considered: action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest
- Planner decision: classic_autoresearch selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_autoresearch selected action_logistic_regression_C-1p0_scale-True from state state_003.
- Evidence needed: Collect single-run metrics and report the best observed score.
- Claim risk: A best single-run claim can be unsupported if top models are close.
- Safety validation: selected=action_logistic_regression_C-1p0_scale-True; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.0314s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_004
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_004; budget_remaining=4; known_results=3
- Candidates considered: action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest
- Planner decision: classic_autoresearch selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_autoresearch selected action_logistic_regression_C-10p0_scale-True from state state_004.
- Evidence needed: Collect single-run metrics and report the best observed score.
- Claim risk: A best single-run claim can be unsupported if top models are close.
- Safety validation: selected=action_logistic_regression_C-10p0_scale-True; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.0123s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_005
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_005; budget_remaining=3; known_results=4
- Candidates considered: action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest, action_random_forest_max_depth-8_n_estimators-300:random_forest
- Planner decision: classic_autoresearch selected svc without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_autoresearch selected action_svc_C-0p5_kernel-rbf_scale-True from state state_005.
- Evidence needed: Collect single-run metrics and report the best observed score.
- Claim risk: A best single-run claim can be unsupported if top models are close.
- Safety validation: selected=action_svc_C-0p5_kernel-rbf_scale-True; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.0093s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_006
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_006; budget_remaining=2; known_results=5
- Candidates considered: action_verify_top_models:top_model_verification, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest, action_random_forest_max_depth-8_n_estimators-300:random_forest, action_gradient_boosting_learning_rate-0p05_n_estimators-100:gradient_boosting, action_gradient_boosting_learning_rate-0p1_n_estimators-100:gradient_boosting
- Planner decision: classic_autoresearch selected random_forest without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_autoresearch selected action_random_forest_n_estimators-100 from state state_006.
- Evidence needed: Collect single-run metrics and report the best observed score.
- Claim risk: A best single-run claim can be unsupported if top models are close.
- Safety validation: selected=action_random_forest_n_estimators-100; override=False; reason=none
- Rejected / deferred: top_model_verification: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; random_forest: classic_autoresearch deferred this catalog action; no world-model score was available.; gradient_boosting: classic_autoresearch deferred this catalog action; no world-model score was available.; gradient_boosting: classic_autoresearch deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.3872s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_007
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_007; budget_remaining=1; known_results=6
- Candidates considered: action_verify_top_models:top_model_verification, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_gradient_boosting_learning_rate-0p05_n_estimators-100:gradient_boosting, action_gradient_boosting_learning_rate-0p1_n_estimators-100:gradient_boosting, action_gradient_boosting_learning_rate-0p05_n_estimators-150:gradient_boosting, action_gradient_boosting_learning_rate-0p1_n_estimators-150:gradient_boosting, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_sweep_1_logistic_regression_C:verification_sweep
- Planner decision: classic_autoresearch selected gradient_boosting without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_autoresearch selected action_gradient_boosting_learning_rate-0p05_n_estimators-100 from state state_007.
- Evidence needed: Collect single-run metrics and report the best observed score.
- Claim risk: A best single-run claim can be unsupported if top models are close.
- Safety validation: selected=action_gradient_boosting_learning_rate-0p05_n_estimators-100; override=False; reason=none
- Rejected / deferred: top_model_verification: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; logistic_regression: classic_autoresearch deferred this catalog action; no world-model score was available.; gradient_boosting: classic_autoresearch deferred this catalog action; no world-model score was available.; gradient_boosting: classic_autoresearch deferred this catalog action; no world-model score was available.; gradient_boosting: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; svc: classic_autoresearch deferred this catalog action; no world-model score was available.; verification_sweep: classic_autoresearch deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.3981s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.
