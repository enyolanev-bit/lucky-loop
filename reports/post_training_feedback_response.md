# Feedback response: post-training probe

Source: Notion meeting feedback, `Réunion @aujourd’hui 19:31`.

## Feedback received

Action items from the meeting:

1. Select a small model, e.g. **Qwen 3 0.6B**.
2. Select a Hugging Face dataset with a clearly defined train/test split.
3. Run a quick LoRA fine-tuning, around **10M trainable parameters**, suitable for a small GPU.
4. Try post-training and evaluate whether the approach works.
5. Consider a cancer/medical dataset or another well-suited dataset.

## Response implemented

Added an additive probe script:

`experiments/post_training_lora_probe.py`

The script is designed to answer the feedback without touching the main Lucky Loop verifier or demo path.

## Default experiment design

| Decision | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` |
| Dataset | `SetFit/sst2` |
| Split requirement | explicit `train` and `test` splits required |
| Method | LoRA adapters on a causal LM |
| Default train size | 256 examples |
| Default test size | 128 examples |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Claim guardrail | only claim post-training helped if held-out test accuracy improves |

## Commands

Dry run, works without heavy ML dependencies:

```bash
python experiments/post_training_lora_probe.py --dry-run
```

Dependency readiness check:

```bash
python experiments/post_training_lora_probe.py --check-deps
```

Real LoRA attempt when `torch`, `transformers`, `datasets`, and `peft` are installed:

```bash
python experiments/post_training_lora_probe.py \
  --model Qwen/Qwen3-0.6B \
  --dataset SetFit/sst2 \
  --text-field text \
  --label-field label \
  --max-train 256 \
  --max-test 128 \
  --output-dir reports/post_training_lora_probe
```

## Output artifacts

The script writes:

| File | Purpose |
|---|---|
| `reports/post_training_lora_probe/result.json` | machine-readable probe result |
| `reports/post_training_lora_probe/README.md` | human-readable summary |

## Honest status

This is a **ready-to-run post-training probe**, not yet a completed LoRA result on the local Mac environment. Current local environment lacks at least some required ML dependencies outside the project venv. The dry run and dependency-check paths are executable and verified.

For the hackathon GPU path, install:

```bash
pip install torch transformers datasets peft accelerate
```

Then run the real command above.

## Why this helps selection

This directly addresses the feedback while preserving the original project story:

- Lucky Loop still demonstrates **predict before compute** and **verify before claim**.
- The post-training probe becomes the next candidate experiment the loop can reason about.
- The same claim discipline applies: no improvement claim without held-out test evidence.
