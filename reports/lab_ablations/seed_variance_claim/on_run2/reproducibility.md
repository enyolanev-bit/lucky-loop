# Reproducibility

Run commands used by the lab:

```bash
python experiments/ml_validity_lab.py --action inspect_dataset --study single_run_vs_repeated_seeds --dataset wine --seeds 0 1 2 3 --out-dir /Users/nevil/dev/sandbox/lucky-loop/reports/lab/seed-variance-claim-does-feature-scaling-improve-accuracy --step 1
python experiments/ml_validity_lab.py --action run_main_protocol --study single_run_vs_repeated_seeds --dataset wine --seeds 0 1 2 3 --out-dir /Users/nevil/dev/sandbox/lucky-loop/reports/lab/seed-variance-claim-does-feature-scaling-improve-accuracy --step 2
python experiments/ml_validity_lab.py --action run_baseline --study single_run_vs_repeated_seeds --dataset wine --seeds 0 1 2 3 --out-dir /Users/nevil/dev/sandbox/lucky-loop/reports/lab/seed-variance-claim-does-feature-scaling-improve-accuracy --step 3
python experiments/ml_validity_lab.py --action run_replication --study single_run_vs_repeated_seeds --dataset wine --seeds 0 1 2 3 --out-dir /Users/nevil/dev/sandbox/lucky-loop/reports/lab/seed-variance-claim-does-feature-scaling-improve-accuracy --step 4
python -c "print('stop_and_report: no further lab compute requested')"
```

Qwen-AgentWorld predictions are pre-execution forecasts. Scientific evidence comes from the Python experiment artifacts.
