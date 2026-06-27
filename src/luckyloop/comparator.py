from __future__ import annotations
import re
from .schemas import ActualResult, Comparison, Prediction


def _range_from_text(text: str):
    nums = [float(x) for x in re.findall(r"0\.\d+|1\.0+", text)]
    if len(nums) >= 2:
        return min(nums), max(nums)
    if len(nums) == 1:
        return max(0, nums[0] - 0.03), min(1, nums[0] + 0.03)
    return None


def compare(pred: Prediction, actual: ActualResult) -> Comparison:
    unexpected = []
    metric_match = False
    rng = _range_from_text(pred.expected_metric)
    actual_metric = actual.accuracy
    if actual_metric is None and actual.raw.get("best", {}).get("mean_accuracy") is not None:
        actual_metric = float(actual.raw["best"]["mean_accuracy"])
    if actual.status != "success":
        unexpected.append(f"execution status was {actual.status}")
    if actual_metric is not None and rng:
        lo, hi = rng
        metric_match = lo <= actual_metric <= hi
        if not metric_match:
            unexpected.append(f"accuracy {actual_metric:.4f} outside predicted range {lo:.2f}-{hi:.2f}")
    elif actual_metric is not None:
        metric_match = True
    runtime_match = True
    if actual.runtime_seconds is not None:
        m = re.search(r"under\s+(\d+)", pred.expected_runtime_seconds.lower())
        if m and actual.runtime_seconds > float(m.group(1)):
            runtime_match = False
            unexpected.append(f"runtime {actual.runtime_seconds:.2f}s exceeded predicted {m.group(1)}s")
    if metric_match and runtime_match and actual.status == "success":
        lesson = "Prediction was broadly consistent with the real run."
    elif actual.status == "success":
        lesson = "Run succeeded, but the prediction missed at least one quantitative detail."
    else:
        lesson = "Run failed; inspect stderr and adjust the next action."
    return Comparison(metric_match=metric_match, runtime_match=runtime_match, unexpected_events=unexpected, lesson=lesson)
