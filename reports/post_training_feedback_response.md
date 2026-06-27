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

## Real run result (Team Pegasus / MI300X)

The probe ran end-to-end on the hackathon GPU container. Real numbers (`reports/post_training_lora_probe/result.json`, `status: completed`):

| Metric | Value |
|---|---|
| Baseline held-out accuracy | `0.484375` (31/64) |
| Post-training held-out accuracy | `0.5` (32/64) |
| Delta | `+0.015625` |
| Trainable params (LoRA) | `5,046,272` (0.84% of 601M) |
| Runtime | `37.05 s` |
| Dataset | `SetFit/sst2`, train=64 / test=64 |

### Claim-calibrated verdict

The delta is **one flipped test example**. It is indistinguishable from zero:

- Two-proportion z-test: `z ≈ 0.18`, `p ≈ 0.86`.
- McNemar (≥1 discordant pair): `p ≈ 1.0`.

**No improvement claim.** Applying Lucky Loop's own effect-vs-noise gate, this is `inconclusive` → blocked. The system refuses to oversell its own experiment. What is established: the post-training path runs end-to-end (Qwen3-0.6B, clean train/test split, 5.0M trainable params, 37s) — a real candidate the loop can reason about under the same "verify before claim" discipline.

## Why this helps selection

This directly addresses the feedback while preserving the original project story:

- Lucky Loop still demonstrates **predict before compute** and **verify before claim**.
- The post-training probe becomes the next candidate experiment the loop can reason about.
- The same claim discipline applies: no improvement claim without held-out test evidence.
