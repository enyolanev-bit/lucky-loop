# Lucky Loop Research Report

Goal: Maximize validation accuracy on sklearn digits under a small compute budget, while avoiding unsupported claims.

## Thesis

An API-backed autoresearch planner proposes hypotheses and safe catalog actions. Qwen-AgentWorld predicts experimental consequences before compute. Lucky Loop then runs real code, compares prediction with reality, and gates scientific claims through a deterministic verifier.

## Experiment timeline

| Run | Agent hypothesis | Qwen predicted | Action run | Reality showed | Claim verdict |
|---|---|---|---|---|---|
| run_001 | A simple unscaled linear baseline should anchor the search before interventions. | Establish a baseline before testing scaling, nonlinear models, or sweeps. | ran logistic_regression; signal=mixed | accuracy 0.9622 | observation only; no robust claim |
| run_002 | Feature scaling should improve or stabilize scale-sensitive linear classification. | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9711 | observation only; no robust claim |
| run_003 | Testing a new model family can reveal whether the current best score depends on inductive bias. | SVC tests a margin-based nonlinear hypothesis after linear/tree baselines. | ran svc; signal=mixed | accuracy 0.9778 | observation only; no robust claim |
| run_004 | Testing a new model family can reveal whether the current best score depends on inductive bias. | Use random forest to test whether nonlinear interactions improve over the linear baseline. | ran random_forest; signal=mixed | accuracy 0.9600 | observation only; no robust claim |
| run_005 | Testing a new model family can reveal whether the current best score depends on inductive bias. | Use boosting to test a staged-tree alternative under the remaining budget. | ran hist_gradient_boosting; signal=mixed | accuracy 0.9578 | observation only; no robust claim |
| run_006 | The best single-run model may not be robust across matched seeds. | Verify the top observed models on matched seeds before allowing a best-model claim. | verified top models: svc_scaled_C=0.5_kernel=rbf, logistic_regression_scaled_C=0.1, logistic_regression; signal=mixed | best mean accuracy 0.9764 | blocked: svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed. |
| run_007 | A prediction miss should trigger exploration of a different inductive bias. | Feature scaling is the direct intervention for a linear model after an unscaled baseline. | ran logistic_regression; signal=mixed | accuracy 0.9778 | observation only; no robust claim |

## Autoresearch agent decisions

- run_001: preferred=action_001; backend=replay; evidence_needed=Establish baseline accuracy, runtime, and convergence behavior.
- run_002: preferred=action_logistic_regression_C-0p1_scale-True; backend=replay; evidence_needed=Compare scaled logistic regression against the unscaled baseline.
- run_003: preferred=action_svc_C-0p5_kernel-rbf_scale-True; backend=replay; evidence_needed=Run a real single-model experiment from the safe catalog.
- run_004: preferred=action_random_forest_n_estimators-100; backend=replay; evidence_needed=Run a real single-model experiment from the safe catalog.
- run_005: preferred=action_hist_gradient_boosting_learning_rate-0p05_n_estimators-100; backend=replay; evidence_needed=Run a real single-model experiment from the safe catalog.
- run_006: preferred=action_verify_top_models; backend=replay; evidence_needed=Run matched multi-seed verification on the observed top models before allowing a robust winner claim.
- run_007: preferred=action_svc_C-2p0_kernel-rbf_scale-True; backend=replay; evidence_needed=Run an untested model family and compare prediction error against actual metrics.; safety override=safety selector chose a higher-scoring action after world-model and evidence-risk scoring

## Best result

Best single run: run_003, model=svc, accuracy=0.9778, f1=0.9777.

## Top model robustness

- run_006: verified svc_scaled_C=0.5_kernel=rbf, logistic_regression_scaled_C=0.1, logistic_regression; verdict=inconclusive; effect/noise=0.839978; svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.

## World model calibration

- Metric interval coverage: 100.00%
- Runtime interval coverage: 85.71%
- Prediction miss count: 1
- Useful decision signals: 7/7
- Full calibration table: `reports/world_model_calibration.md`

## Supported claims

- No claim reached supported or strongly_supported yet.

## Weakly supported claims

- No weakly supported claim was recorded.

## Blocked / inconclusive claims

- Blocked: svc_scaled_C=0.5_kernel=rbf is robustly better than logistic_regression_scaled_C=0.1.
  Allowed rewrite: svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.

## Claim ledger

- Full ledger: `reports/claim_ledger.json`

## Prediction misses

- run_006: runtime 49.13s exceeded predicted 45s

## Evidence notes

### run_001
- Prediction rationale: A simple linear baseline is cheap and informative for sklearn tabular classification.
- Risks: unscaled features can slow convergence or underperform when feature scales differ
- State before: state_001; budget_remaining=7; known_results=0
- Candidates considered: action_001:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_logistic_regression_C-0p1_scale-True:logistic_regression, action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest
- Planner decision: Selected logistic_regression because score=170.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions. Agent hypothesis: A simple unscaled linear baseline should anchor the search before interventions.. World model predicted accuracy around 0.90-0.97, runtime under 5, recommendation=run, risks=unscaled features can slow convergence or underperform when feature scales differ. Causal signal type: mixed.
- Agent rationale: Given state state_001 with budget_remaining=7, the planner chooses action_001 to collect the most decision-relevant evidence.
- Evidence needed: Establish baseline accuracy, runtime, and convergence behavior.
- Claim risk: No best-model claim is allowed from a baseline-only run.
- Safety validation: selected=action_001; override=False; reason=none
- Rejected / deferred: logistic_regression: score=136.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions; logistic_regression: score=136.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions; logistic_regression: score=136.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; new model family increases search coverage; first run should establish the unscaled baseline before interventions; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; logistic_regression: score=16.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; new model family increases search coverage; scaled baseline is deferred until after an unscaled control; logistic_regression: score=16.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; new model family increases search coverage; scaled baseline is deferred until after an unscaled control; logistic_regression: score=8.0; world model recommended run; new model family increases search coverage; scaled baseline is deferred until after an unscaled control
- Actual status: success, runtime: 2.0795s
- Lesson: Prediction was broadly consistent with the real run.

### run_002
- Prediction rationale: Scaling usually helps logistic regression on tabular numeric features.
- Risks: minor convergence warning possible
- State before: state_002; budget_remaining=6; known_results=1
- Candidates considered: action_logistic_regression_C-0p1_scale-True:logistic_regression, action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest
- Planner decision: Selected logistic_regression because score=105.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: Feature scaling should improve or stabilize scale-sensitive linear classification.. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed.
- Agent rationale: Given state state_002 with budget_remaining=6, the planner chooses action_logistic_regression_C-0p1_scale-True to collect the most decision-relevant evidence.
- Evidence needed: Compare scaled logistic regression against the unscaled baseline.
- Claim risk: A single improvement is observational until repeated-seed verification.
- Safety validation: selected=action_logistic_regression_C-0p1_scale-True; override=False; reason=none
- Rejected / deferred: logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; logistic_regression: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 0.6841s
- Lesson: Prediction was broadly consistent with the real run.

### run_003
- Prediction rationale: Scaled SVC can be strong on small numeric classification tasks but needs careful comparison.
- Risks: sensitive to scaling and C
- State before: state_003; budget_remaining=5; known_results=2
- Candidates considered: action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_logistic_regression_C-0p1_scale-False:logistic_regression, action_logistic_regression_C-1p0_scale-False:logistic_regression, action_logistic_regression_C-10p0_scale-False:logistic_regression, action_svc_C-0p5_kernel-rbf_scale-True:svc, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest
- Planner decision: Selected svc because score=90.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage. Agent hypothesis: Testing a new model family can reveal whether the current best score depends on inductive bias.. World model predicted accuracy around 0.90-0.99, runtime under 10, recommendation=run, risks=sensitive to scaling and C. Causal signal type: mixed.
- Agent rationale: Given state state_003 with budget_remaining=5, the planner chooses action_svc_C-0p5_kernel-rbf_scale-True to collect the most decision-relevant evidence.
- Evidence needed: Run a real single-model experiment from the safe catalog.
- Claim risk: The result is still an observation until robustness is checked.
- Safety validation: selected=action_svc_C-0p5_kernel-rbf_scale-True; override=False; reason=none
- Rejected / deferred: logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; svc: score=48.0; world model recommended run; new model family increases search coverage; svc: score=48.0; world model recommended run; new model family increases search coverage; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; logistic_regression: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; logistic_regression: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 0.1627s
- Lesson: Prediction was broadly consistent with the real run.

### run_004
- Prediction rationale: Tree ensembles test a different inductive bias, but may not beat a strong scaled linear model on small tabular datasets.
- Risks: overfitting or split variance if depth is unconstrained
- State before: state_004; budget_remaining=4; known_results=3
- Candidates considered: action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_random_forest_n_estimators-100:random_forest, action_random_forest_max_depth-4_n_estimators-100:random_forest, action_random_forest_max_depth-8_n_estimators-100:random_forest, action_random_forest_n_estimators-300:random_forest, action_random_forest_max_depth-4_n_estimators-300:random_forest, action_random_forest_max_depth-8_n_estimators-300:random_forest, action_hist_gradient_boosting_learning_rate-0p05_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p05_max_depth-8_n_estimators-100:hist_gradient_boosting
- Planner decision: Selected random_forest because score=86.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; world model predicted overfitting risk. Agent hypothesis: Testing a new model family can reveal whether the current best score depends on inductive bias.. World model predicted accuracy around 0.90-0.98, runtime under 10, recommendation=run, risks=overfitting or split variance if depth is unconstrained. Causal signal type: mixed.
- Agent rationale: Given state state_004 with budget_remaining=4, the planner chooses action_random_forest_n_estimators-100 to collect the most decision-relevant evidence.
- Evidence needed: Run a real single-model experiment from the safe catalog.
- Claim risk: The result is still an observation until robustness is checked.
- Safety validation: selected=action_random_forest_n_estimators-100; override=False; reason=none
- Rejected / deferred: logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; random_forest: score=52.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; random_forest: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; svc: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; svc: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 0.4564s
- Lesson: Prediction was broadly consistent with the real run.

### run_005
- Prediction rationale: Boosting is a useful late comparison when linear and bagged-tree baselines are known.
- Risks: can overfit with too many estimators
- State before: state_005; budget_remaining=3; known_results=4
- Candidates considered: action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_hist_gradient_boosting_learning_rate-0p05_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p05_max_depth-8_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_max_depth-8_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p05_n_estimators-150:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p05_max_depth-8_n_estimators-150:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_n_estimators-150:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_max_depth-8_n_estimators-150:hist_gradient_boosting
- Planner decision: Selected hist_gradient_boosting because score=86.0; autoresearch agent preferred this catalog action; world model recommended run; new model family increases search coverage; world model predicted overfitting risk. Agent hypothesis: Testing a new model family can reveal whether the current best score depends on inductive bias.. World model predicted accuracy around 0.90-0.98, runtime under 15, recommendation=run, risks=can overfit with too many estimators. Causal signal type: mixed.
- Agent rationale: Given state state_005 with budget_remaining=3, the planner chooses action_hist_gradient_boosting_learning_rate-0p05_n_estimators-100 to collect the most decision-relevant evidence.
- Evidence needed: Run a real single-model experiment from the safe catalog.
- Claim risk: The result is still an observation until robustness is checked.
- Safety validation: selected=action_hist_gradient_boosting_learning_rate-0p05_n_estimators-100; override=False; reason=none
- Rejected / deferred: logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; hist_gradient_boosting: score=52.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; hist_gradient_boosting: score=44.0; world model recommended run; new model family increases search coverage; world model predicted overfitting risk; svc: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; svc: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 3.9033s
- Lesson: Prediction was broadly consistent with the real run.

### run_006
- Prediction rationale: A multi-seed top-model comparison is required before reporting a robust best model.
- Risks: top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap
- State before: state_006; budget_remaining=2; known_results=5
- Candidates considered: action_verify_top_models:top_model_verification, action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_sweep_1_svc_C:verification_sweep
- Planner decision: Selected top_model_verification because score=152.0; autoresearch agent preferred this catalog action; world model recommended run; top observed models need multi-seed verification before a robust best-model claim; world model predicted top-model robustness or claim risk. Agent hypothesis: The best single-run model may not be robust across matched seeds.. World model predicted accuracy around 0.90-0.99, runtime under 45, recommendation=run, risks=top single-run models may be tied across seeds, seed variance may exceed the observed top-model gap. Causal signal type: mixed.
- Agent rationale: Given state state_006 with budget_remaining=2, the planner chooses action_verify_top_models to collect the most decision-relevant evidence.
- Evidence needed: Run matched multi-seed verification on the observed top models before allowing a robust winner claim.
- Claim risk: A single-split leaderboard can overstate a fragile best-model claim.
- Safety validation: selected=action_verify_top_models; override=False; reason=none
- Rejected / deferred: logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; verification_sweep: score=63.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; world model predicted robustness or seed-variance risk; svc: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; svc: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family
- Actual status: success, runtime: 49.126s
- Unexpected: runtime 49.13s exceeded predicted 45s
- Lesson: Run succeeded, but the prediction missed at least one quantitative detail.
- Verifier verdict: inconclusive; trustworthy=False; effect_size=0.009333; seed_noise=0.011111; effect_to_noise_ratio=0.839978
- Blocked claim: svc_scaled_C=0.5_kernel=rbf is robustly better than logistic_regression_scaled_C=0.1.
- Allowed claim: svc_scaled_C=0.5_kernel=rbf had the best multi-seed mean accuracy, but the effect was smaller than seed noise; no robust best-model claim is allowed.
- Verifier rationale: Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model.

### run_007
- Prediction rationale: Scaling usually helps logistic regression on tabular numeric features.
- Risks: minor convergence warning possible
- State before: state_007; budget_remaining=1; known_results=6
- Candidates considered: action_logistic_regression_C-1p0_scale-True:logistic_regression, action_logistic_regression_C-10p0_scale-True:logistic_regression, action_svc_C-2p0_kernel-rbf_scale-True:svc, action_svc_C-5p0_kernel-rbf_scale-True:svc, action_hist_gradient_boosting_learning_rate-0p05_max_depth-8_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_max_depth-8_n_estimators-100:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p05_n_estimators-150:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p05_max_depth-8_n_estimators-150:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_n_estimators-150:hist_gradient_boosting, action_hist_gradient_boosting_learning_rate-0p08_max_depth-8_n_estimators-150:hist_gradient_boosting, action_sweep_1_svc_C:verification_sweep
- Planner decision: Selected logistic_regression because score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test. Agent hypothesis: A prediction miss should trigger exploration of a different inductive bias.. World model predicted accuracy around 0.95-0.98, runtime under 5, recommendation=run, risks=minor convergence warning possible. Causal signal type: mixed.
- Agent rationale: Given state state_007 with budget_remaining=1, the planner chooses action_svc_C-2p0_kernel-rbf_scale-True to collect the most decision-relevant evidence.
- Evidence needed: Run an untested model family and compare prediction error against actual metrics.
- Claim risk: Prediction misses must be logged, not hidden behind a better score.
- Safety validation: selected=action_logistic_regression_C-1p0_scale-True; override=True; reason=safety selector chose a higher-scoring action after world-model and evidence-risk scoring
- Rejected / deferred: logistic_regression: score=71.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; after an unscaled logistic baseline, scaling is the direct world-model intervention to test; svc: score=60.0; autoresearch agent preferred this catalog action; world model recommended run; candidate is a variant of an already tested family; verification_sweep: score=55.0; world model recommended run; world model predicted robustness or seed-variance risk; svc: score=26.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; hist_gradient_boosting: score=22.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; world model predicted overfitting risk; hist_gradient_boosting: score=22.0; autoresearch agent included this action in its candidate shortlist; world model recommended run; candidate is a variant of an already tested family; world model predicted overfitting risk; hist_gradient_boosting: score=14.0; world model recommended run; candidate is a variant of an already tested family; world model predicted overfitting risk; hist_gradient_boosting: score=14.0; world model recommended run; candidate is a variant of an already tested family; world model predicted overfitting risk; hist_gradient_boosting: score=14.0; world model recommended run; candidate is a variant of an already tested family; world model predicted overfitting risk; hist_gradient_boosting: score=14.0; world model recommended run; candidate is a variant of an already tested family; world model predicted overfitting risk; hist_gradient_boosting: score=14.0; world model recommended run; candidate is a variant of an already tested family; world model predicted overfitting risk
- Actual status: success, runtime: 0.3112s
- Lesson: Prediction was broadly consistent with the real run.
