#!/usr/bin/env python3
"""Post-training LoRA probe for Lucky Loop feedback.

Purpose
-------
The hackathon feedback asked for a small-model post-training attempt, e.g.
Qwen3-0.6B on a Hugging Face dataset with a clear train/test split.

This file is intentionally additive. It does not affect the main Lucky Loop
verifier, claim ledger, or demo path.

Modes
-----
1. Dry run, no heavy dependencies required:
   python experiments/post_training_lora_probe.py --dry-run

2. Dependency check:
   python experiments/post_training_lora_probe.py --check-deps

3. Real LoRA attempt, when transformers/datasets/peft/torch are installed:
   python experiments/post_training_lora_probe.py \
     --model Qwen/Qwen3-0.6B \
     --dataset SetFit/sst2 \
     --text-field text \
     --label-field label \
     --max-train 256 \
     --max-test 128 \
     --output-dir reports/post_training_lora_probe

The default real target is deliberately small enough for a hackathon GPU, but
large enough to answer the feedback seriously: a small Qwen model, LoRA adapters,
and a dataset with explicit train/test splits.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "post_training_lora_probe"
REQUIRED_DEPS = ["torch", "transformers", "datasets", "peft"]


@dataclass(frozen=True)
class ProbeConfig:
    model: str
    dataset: str
    text_field: str
    label_field: str
    max_train: int
    max_test: int
    lora_r: int
    lora_alpha: int
    learning_rate: float
    epochs: float
    output_dir: str


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    available: bool


def dependency_status() -> list[DependencyStatus]:
    return [DependencyStatus(name=dep, available=importlib.util.find_spec(dep) is not None) for dep in REQUIRED_DEPS]


def missing_dependencies() -> list[str]:
    return [dep.name for dep in dependency_status() if not dep.available]


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    lines = [
        "# Post-training LoRA probe",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Config",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in payload["config"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Result", ""])
    if payload["status"] == "dry_run":
        lines.append("Dry run only. No model weights were downloaded and no training was executed.")
    elif payload["status"] == "missing_dependencies":
        lines.append("The probe is ready, but the local environment is missing required ML dependencies.")
        lines.append("")
        lines.append("Missing: " + ", ".join(f"`{dep}`" for dep in payload["missing_dependencies"]))
    elif payload["status"] == "completed":
        lines.extend(
            [
                f"- Baseline accuracy: `{payload['baseline_accuracy']:.4f}`",
                f"- Post-training accuracy: `{payload['post_training_accuracy']:.4f}`",
                f"- Delta: `{payload['delta']:+.4f}`",
                f"- Runtime seconds: `{payload['runtime_seconds']:.2f}`",
            ]
        )
    else:
        lines.append(json.dumps(payload, indent=2, sort_keys=True))
    (path / "README.md").write_text("\n".join(lines) + "\n")


def dry_run(config: ProbeConfig) -> dict[str, Any]:
    payload = {
        "status": "dry_run",
        "config": asdict(config),
        "dependency_status": [asdict(dep) for dep in dependency_status()],
        "planned_steps": [
            "load Hugging Face dataset with explicit train/test split",
            "load small causal LM/tokenizer",
            "evaluate baseline zero/few-shot classification proxy",
            "attach LoRA adapters",
            "train only adapter parameters",
            "evaluate post-training on held-out test split",
            "write result.json and README.md",
        ],
        "claim_guardrail": "No improvement claim is allowed unless held-out accuracy improves after post-training.",
    }
    write_report(Path(config.output_dir), payload)
    return payload


def check_deps(config: ProbeConfig) -> dict[str, Any]:
    missing = missing_dependencies()
    status = "ready" if not missing else "missing_dependencies"
    payload = {
        "status": status,
        "config": asdict(config),
        "dependency_status": [asdict(dep) for dep in dependency_status()],
        "missing_dependencies": missing,
    }
    write_report(Path(config.output_dir), payload)
    return payload


def run_real_probe(config: ProbeConfig) -> dict[str, Any]:
    missing = missing_dependencies()
    if missing:
        payload = {
            "status": "missing_dependencies",
            "config": asdict(config),
            "dependency_status": [asdict(dep) for dep in dependency_status()],
            "missing_dependencies": missing,
        }
        write_report(Path(config.output_dir), payload)
        return payload

    # Imports stay inside this function so --dry-run and --check-deps work in a
    # minimal environment.
    import torch  # type: ignore[import-not-found]
    from datasets import load_dataset  # type: ignore[import-not-found]
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # type: ignore[import-not-found]

    started = time.perf_counter()
    dataset = load_dataset(config.dataset)
    if "train" not in dataset or "test" not in dataset:
        raise ValueError(f"Dataset {config.dataset!r} must expose explicit train and test splits.")

    train_ds = dataset["train"].select(range(min(config.max_train, len(dataset["train"]))))
    test_ds = dataset["test"].select(range(min(config.max_test, len(dataset["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    label_values = sorted({str(row[config.label_field]) for row in train_ds})
    if len(label_values) < 2:
        raise ValueError("Need at least two labels for a classification-style probe.")

    def prompt(row: dict[str, Any]) -> str:
        labels = ", ".join(label_values)
        return f"Classify the text into one of [{labels}].\nText: {row[config.text_field]}\nLabel:"

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        answer = " " + str(row[config.label_field])
        text = prompt(row) + answer + tokenizer.eos_token
        encoded = tokenizer(text, truncation=True, max_length=256, padding="max_length")
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    tokenized_train = train_ds.map(tokenize, remove_columns=train_ds.column_names)

    def score_accuracy(eval_ds: Any) -> float:
        model.eval()
        hits = 0
        total = 0
        for row in eval_ds:
            encoded = tokenizer(prompt(row), return_tensors="pt", truncation=True, max_length=256)
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = model.generate(**encoded, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(output[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
            prediction = min(label_values, key=lambda label: 0 if generated.startswith(label) else 1)
            hits += int(prediction == str(row[config.label_field]))
            total += 1
        return hits / total if total else math.nan

    baseline_accuracy = score_accuracy(test_ds)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    args = TrainingArguments(
        output_dir=str(Path(config.output_dir) / "trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_train)
    trainer.train()
    post_accuracy = score_accuracy(test_ds)

    payload = {
        "status": "completed",
        "config": asdict(config),
        "baseline_accuracy": baseline_accuracy,
        "post_training_accuracy": post_accuracy,
        "delta": post_accuracy - baseline_accuracy,
        "runtime_seconds": time.perf_counter() - started,
        "claim_guardrail": "Only claim post-training helped if delta > 0 on held-out test split.",
    }
    write_report(Path(config.output_dir), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lucky Loop post-training LoRA probe")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", default="SetFit/sst2")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--max-train", type=int, default=256)
    parser.add_argument("--max-test", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-deps", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProbeConfig(
        model=args.model,
        dataset=args.dataset,
        text_field=args.text_field,
        label_field=args.label_field,
        max_train=args.max_train,
        max_test=args.max_test,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )
    if args.dry_run:
        payload = dry_run(config)
    elif args.check_deps:
        payload = check_deps(config)
    else:
        payload = run_real_probe(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
