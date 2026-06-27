#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time


SCENARIOS = {
    "weak_effect": {
        "metric": "accuracy",
        "protocol_warning": None,
        "runs": [
            ("baseline", [0.948, 0.958, 0.968, 0.978]),
            ("candidate", [0.960, 0.970, 0.980, 0.990]),
        ],
    },
    "real_effect": {
        "metric": "accuracy",
        "protocol_warning": None,
        "runs": [
            ("baseline", [0.760, 0.765, 0.770, 0.775]),
            ("candidate", [0.860, 0.865, 0.870, 0.875]),
        ],
    },
    "data_leakage_trap": {
        "metric": "accuracy",
        "protocol_warning": "suspiciously high score; label-derived feature was included before the split",
        "runs": [
            ("proper_protocol", [0.948, 0.952, 0.956, 0.960]),
            ("leaky_protocol", [0.998, 0.999, 1.000, 1.000]),
        ],
    },
    "metric_misuse": {
        "metric": "balanced_accuracy",
        "protocol_warning": "accuracy is misleading on an imbalanced dataset; balanced_accuracy is the verifier metric",
        "runs": [
            ("accuracy_only_baseline", [0.500, 0.505, 0.510, 0.515]),
            ("balanced_objective", [0.710, 0.715, 0.720, 0.725]),
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    args = parser.parse_args()

    t0 = time.perf_counter()
    spec = SCENARIOS[args.scenario]
    metric = spec["metric"]
    rows = []
    by_value = {}
    for config, values in spec["runs"]:
        by_value[config] = {
            f"mean_{metric}": sum(values) / len(values),
            f"min_{metric}": min(values),
            f"max_{metric}": max(values),
            "n": len(values),
        }
        for seed, value in enumerate(values):
            rows.append(
                {
                    "config_key": config,
                    "label": config,
                    "sweep_param": "scenario_config",
                    "value": config,
                    "seed": seed,
                    metric: value,
                    "accuracy": value if metric == "accuracy" else None,
                    "runtime_seconds": 0.001,
                }
            )

    best_value = max(by_value, key=lambda k: by_value[k][f"mean_{metric}"])
    out = {
        "status": "success",
        "scenario": args.scenario,
        "metric": metric,
        "runs": rows,
        "summary_by_value": by_value,
        "best": {"value": best_value, **by_value[best_value]},
        "values": list(by_value.keys()),
        "seeds": [0, 1, 2, 3],
        "protocol_warning": spec["protocol_warning"],
        "runtime_seconds": round(time.perf_counter() - t0, 4),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
