from __future__ import annotations

CORE_TASK_PATHS = [
    "configs/tasks/breast_cancer_accuracy.json",
    "configs/tasks/wine_accuracy.json",
    "configs/tasks/digits_accuracy.json",
    "configs/tasks/sonar_accuracy.json",
]

OPTIONAL_TASK_PATHS = [
    "configs/tasks/eeg_eye_state_accuracy.json",
    "configs/tasks/har_accuracy.json",
]

CORE_POLICIES = [
    "classic_autoresearch",
    "classic_verified",
    "lucky_loop_full",
]

OPTIONAL_POLICIES = [
    "fixed_order",
    "score_chaser",
    "lucky_loop_no_qwen",
    "lucky_loop_qwen_cost_aware",
]

ALL_TASK_PATHS = [*CORE_TASK_PATHS, *OPTIONAL_TASK_PATHS]
ALL_POLICIES = [*OPTIONAL_POLICIES[:2], *CORE_POLICIES, *OPTIONAL_POLICIES[2:]]

