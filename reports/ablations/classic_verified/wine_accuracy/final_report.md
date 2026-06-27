# Lucky Loop Research Report

Goal: Maximize validation accuracy on sklearn wine under a small compute budget, while avoiding unsupported claims.

## Thesis

An API-backed autoresearch planner proposes hypotheses and safe catalog actions. Qwen-AgentWorld predicts experimental consequences before compute. Lucky Loop then runs real code, compares prediction with reality, and gates scientific claims through a deterministic verifier.

## Experiment timeline

| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |
|---|---|---|---|---|---|
| run_001 | Classic autoresearch should verify the top observed models before a robust winner claim. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_002 | Classic autoresearch should verify the top observed models before a robust winner claim. | none | ran logistic_regression; signal=selector_policy | accuracy 1.0000 | prediction miss logged; no robust claim |
| run_003 | Classic autoresearch should verify the top observed models before a robust winner claim. | none | ran logistic_regression; signal=selector_policy | accuracy 1.0000 | prediction miss logged; no robust claim |
| run_004 | Classic autoresearch should verify the top observed models before a robust winner claim. | none | ran logistic_regression; signal=selector_policy | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_005 | Classic autoresearch should verify the top observed models before a robust winner claim. | none | ran svc; signal=selector_policy | accuracy 0.9778 | prediction miss logged; no robust claim |
| run_006 | Classic autoresearch should verify the top observed models before a robust winner claim. | none | verified top models: logistic_regression_scaled_C=0.1, logistic_regression_scaled_C=1.0, logistic_regression; signal=selector_policy | best mean accuracy 0.9956 | blocked: logistic_regression_scaled_C=0.1 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |
| run_007 | Classic autoresearch should verify the top observed models before a robust winner claim. | none | ran random_forest; signal=selector_policy | accuracy 1.0000 | prediction miss logged; no robust claim |

## Autoresearch agent decisions

- run_001: preferred=action_001; backend=classic_verified; evidence_needed=Run matched multi-seed verification when observed leaders are close.
- run_002: preferred=action_logistic_regression_C-0p1_scale-True; backend=classic_verified; evidence_needed=Run matched multi-seed verification when observed leaders are close.
- run_003: preferred=action_logistic_regression_C-1p0_scale-True; backend=classic_verified; evidence_needed=Run matched multi-seed verification when observed leaders are close.
- run_004: preferred=action_logistic_regression_C-10p0_scale-True; backend=classic_verified; evidence_needed=Run matched multi-seed verification when observed leaders are close.
- run_005: preferred=action_svc_C-0p5_kernel-rbf_scale-True; backend=classic_verified; evidence_needed=Run matched multi-seed verification when observed leaders are close.
- run_006: preferred=action_verify_top_models; backend=classic_verified; evidence_needed=Run matched multi-seed verification when observed leaders are close.
- run_007: preferred=action_random_forest_n_estimators-100; backend=classic_verified; evidence_needed=Run matched multi-seed verification when observed leaders are close.

## Best result

Best single run: run_002, model=logistic_regression, accuracy=1.0000, f1=1.0000.

## Top model robustness

- run_006: verified logistic_regression_scaled_C=0.1, logistic_regression_scaled_C=1.0, logistic_regression; verdict=inconclusive; effect/noise=0.400009; logistic_regression_scaled_C=0.1 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.

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

- Blocked: logistic_regression_scaled_C=0.1 is robustly better than logistic_regression_scaled_C=1.0.
  Allowed rewrite: logistic_regression_scaled_C=0.1 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.

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
- Planner decision: classic_verified selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_verified selected action_001 from state state_001.
- Evidence needed: Run matched multi-seed verification when observed leaders are close.
- Claim risk: No world model predicts the verification need before compute.
- Safety validation: selected=action_001; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 1.2726s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_002
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_002; budget_remaining=6; known_results=1
- Candidates considered: action_logistic_regression_C-0p1_scale-True:logistic_regression, action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest
- Planner decision: classic_verified selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_verified selected action_logistic_regression_C-0p1_scale-True from state state_002.
- Evidence needed: Run matched multi-seed verification when observed leaders are close.
- Claim risk: No world model predicts the verification need before compute.
- Safety validation: selected=action_logistic_regression_C-0p1_scale-True; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.0114s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_003
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_003; budget_remaining=5; known_results=2
- Candidates considered: action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest
- Planner decision: classic_verified selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_verified selected action_logistic_regression_C-1p0_scale-True from state state_003.
- Evidence needed: Run matched multi-seed verification when observed leaders are close.
- Claim risk: No world model predicts the verification need before compute.
- Safety validation: selected=action_logistic_regression_C-1p0_scale-True; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.0133s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_004
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_004; budget_remaining=4; known_results=3
- Candidates considered: action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest
- Planner decision: classic_verified selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_verified selected action_logistic_regression_C-10p0_scale-True from state state_004.
- Evidence needed: Run matched multi-seed verification when observed leaders are close.
- Claim risk: No world model predicts the verification need before compute.
- Safety validation: selected=action_logistic_regression_C-10p0_scale-True; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.0136s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_005
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_005; budget_remaining=3; known_results=4
- Candidates considered: action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest, action_random_forest_max_depth-8_n_estimators-300:random_forest
- Planner decision: classic_verified selected svc without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_verified selected action_svc_C-0p5_kernel-rbf_scale-True from state state_005.
- Evidence needed: Run matched multi-seed verification when observed leaders are close.
- Claim risk: No world model predicts the verification need before compute.
- Safety validation: selected=action_svc_C-0p5_kernel-rbf_scale-True; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; svc: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.0075s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.

### run_006
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_006; budget_remaining=2; known_results=5
- Candidates considered: action_verify_top_models:top_model_verification, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest, action_random_forest_max_depth-8_n_estimators-300:random_forest, action_gradient_boosting_learning_rate-0p05_n_estimators-100:gradient_boosting, action_gradient_boosting_learning_rate-0p1_n_estimators-100:gradient_boosting
- Planner decision: classic_verified selected top_model_verification without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_verified selected action_verify_top_models from state state_006.
- Evidence needed: Run matched multi-seed verification when observed leaders are close.
- Claim risk: No world model predicts the verification need before compute.
- Safety validation: selected=action_verify_top_models; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; gradient_boosting: classic_verified deferred this catalog action; no world-model score was available.; gradient_boosting: classic_verified deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 38.5655s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.008889; seed_noise=0.022222; effect_to_noise_ratio=0.400009
- Blocked claim: logistic_regression_scaled_C=0.1 is robustly better than logistic_regression_scaled_C=1.0.
- Allowed claim: logistic_regression_scaled_C=0.1 had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.
- Verifier rationale: Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model.

### run_007
- Prediction rationale: This baseline chooses experiments from observed state and policy rules without Qwen-AgentWorld.
- Risks: classic autoresearch baseline does not simulate this action before compute
- State before: state_007; budget_remaining=1; known_results=6
- Candidates considered: action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest, action_random_forest_max_depth-8_n_estimators-300:random_forest, action_gradient_boosting_learning_rate-0p05_n_estimators-100:gradient_boosting, action_gradient_boosting_learning_rate-0p1_n_estimators-100:gradient_boosting, action_gradient_boosting_learning_rate-0p05_n_estimators-150:gradient_boosting
- Planner decision: classic_verified selected random_forest without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against.
- Agent rationale: classic_verified selected action_random_forest_n_estimators-100 from state state_007.
- Evidence needed: Run matched multi-seed verification when observed leaders are close.
- Claim risk: No world model predicts the verification need before compute.
- Safety validation: selected=action_random_forest_n_estimators-100; override=False; reason=none
- Rejected / deferred: logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; logistic_regression: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; random_forest: classic_verified deferred this catalog action; no world-model score was available.; gradient_boosting: classic_verified deferred this catalog action; no world-model score was available.; gradient_boosting: classic_verified deferred this catalog action; no world-model score was available.; gradient_boosting: classic_verified deferred this catalog action; no world-model score was available.
- Actual status: success, runtime: 0.3336s
- Unexpected: no world-model prediction was made before compute
- Lesson: Classic baseline has no prediction-vs-reality evidence for this action.
