# World Model Calibration

Lucky Loop records whether Qwen-AgentWorld predictions matched real experiment observations. Misses are kept as evidence; they are not hidden.

## Summary

- Metric interval coverage: n/a
- Runtime interval coverage: n/a
- Mean metric absolute error outside interval: n/a
- Mean runtime relative error above bound: n/a
- Prediction miss count: 7
- Risk recall approximation: 0.00%
- Useful decision signals: 0/7

## Prediction vs Reality

| Run | Model | Predicted metric | Actual metric | Metric hit | Predicted runtime | Actual runtime | Runtime hit | Observed miss |
|---|---|---|---:|---|---|---:|---|---|
| run_001 | logistic_regression | no pre-compute world-model prediction | 0.9622 | no | not predicted | 2.82s | no | no world-model prediction was made before compute |
| run_002 | logistic_regression | no pre-compute world-model prediction | 0.9711 | no | not predicted | 0.32s | no | no world-model prediction was made before compute |
| run_003 | logistic_regression | no pre-compute world-model prediction | 0.9778 | no | not predicted | 0.48s | no | no world-model prediction was made before compute |
| run_004 | logistic_regression | no pre-compute world-model prediction | 0.9644 | no | not predicted | 0.30s | no | no world-model prediction was made before compute |
| run_005 | svc | no pre-compute world-model prediction | 0.9778 | no | not predicted | 0.09s | no | no world-model prediction was made before compute |
| run_006 | top_model_verification | no pre-compute world-model prediction | 0.9764 | no | not predicted | 32.69s | no | no world-model prediction was made before compute |
| run_007 | random_forest | no pre-compute world-model prediction | 0.9600 | no | not predicted | 0.40s | no | no world-model prediction was made before compute |

## Risk Signals

| Run | Predicted risks | Observed evidence | Risk hit |
|---|---|---|---|
| run_001 | classic autoresearch baseline does not simulate this action before compute | no world-model prediction was made before compute | no |
| run_002 | classic autoresearch baseline does not simulate this action before compute | no world-model prediction was made before compute | no |
| run_003 | classic autoresearch baseline does not simulate this action before compute | no world-model prediction was made before compute | no |
| run_004 | classic autoresearch baseline does not simulate this action before compute | no world-model prediction was made before compute | no |
| run_005 | classic autoresearch baseline does not simulate this action before compute | no world-model prediction was made before compute | no |
| run_006 | classic autoresearch baseline does not simulate this action before compute | no world-model prediction was made before compute; Measured top-model effect is not larger than inter-seed noise. The report must not claim a robust best model. | no |
| run_007 | classic autoresearch baseline does not simulate this action before compute | no world-model prediction was made before compute | no |

## Decision Usefulness

| Run | Selected action | World-model signal used | Decision useful | Reason |
|---|---|---|---|---|
| run_001 | logistic_regression | no | no | classic_verified selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against. |
| run_002 | logistic_regression | no | no | classic_verified selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against. |
| run_003 | logistic_regression | no | no | classic_verified selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against. |
| run_004 | logistic_regression | no | no | classic_verified selected logistic_regression without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against. |
| run_005 | svc | no | no | classic_verified selected svc without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against. |
| run_006 | top_model_verification | no | no | classic_verified selected top_model_verification without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against. |
| run_007 | random_forest | no | no | classic_verified selected random_forest without pre-compute world-model simulation. This is the baseline Lucky Loop is compared against. |