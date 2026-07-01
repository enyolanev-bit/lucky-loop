# Reproducibility

Run commands used by the lab:

```bash
python experiments/ml_validity_lab.py --action inspect_dataset --study random_vs_blocked_split --dataset eeg_eye_state --seeds 0 1 2 3 --out-dir /Users/nevil/dev/sandbox/lucky-loop/reports/lab/split-validity-sensor-does-feature-scaling-improve-accuracy --step 1
python experiments/ml_validity_lab.py --action run_main_protocol --study random_vs_blocked_split --dataset eeg_eye_state --seeds 0 1 2 3 --out-dir /Users/nevil/dev/sandbox/lucky-loop/reports/lab/split-validity-sensor-does-feature-scaling-improve-accuracy --step 2
python experiments/ml_validity_lab.py --action run_random_split_baseline --study random_vs_blocked_split --dataset eeg_eye_state --seeds 0 1 2 3 --out-dir /Users/nevil/dev/sandbox/lucky-loop/reports/lab/split-validity-sensor-does-feature-scaling-improve-accuracy --step 3
python experiments/ml_validity_lab.py --action run_replication --study random_vs_blocked_split --dataset eeg_eye_state --seeds 0 1 2 3 --out-dir /Users/nevil/dev/sandbox/lucky-loop/reports/lab/split-validity-sensor-does-feature-scaling-improve-accuracy --step 4
python experiments/ml_validity_lab.py --action run_negative_control --study random_vs_blocked_split --dataset eeg_eye_state --seeds 0 1 2 3 --out-dir /Users/nevil/dev/sandbox/lucky-loop/reports/lab/split-validity-sensor-does-feature-scaling-improve-accuracy --step 5
python -c "print('stop_and_report: no further lab compute requested')"
```

Qwen-AgentWorld predictions are pre-execution forecasts. Scientific evidence comes from the Python experiment artifacts.
