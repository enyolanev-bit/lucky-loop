# Post-training LoRA probe — Team Pegasus run

Status: `completed` (real run on Team Pegasus / MI300X container).

> Smoke validation that the post-training path runs end-to-end. **Not** a robust benchmark claim.

## Config

| Field | Value |
|---|---|
| `model` | `Qwen/Qwen3-0.6B` |
| `dataset` | `SetFit/sst2` (explicit train/test split) |
| `max_train` | `64` |
| `max_test` | `64` |
| `lora_r` | `8` |
| `lora_alpha` | `16` |
| `learning_rate` | `0.0002` |
| `epochs` | `1.0` |

## Result

| Metric | Value |
|---|---|
| Baseline held-out accuracy | `0.484375` (31/64) |
| Post-training held-out accuracy | `0.5` (32/64) |
| Delta | `+0.015625` (one test example) |
| Total params | `601,096,192` |
| Trainable params (LoRA) | `5,046,272` (0.84%) |
| Runtime | `37.05 s` |

## Honest verdict (claim-calibrated)

The delta is **one flipped test example** out of 64. Under any reasonable test it is indistinguishable from zero:

- Two-proportion z-test: `0.0156 / 0.088 SE ≈ z = 0.18` → `p ≈ 0.86`.
- McNemar (≥1 discordant pair): `p ≈ 1.0`.

**We do NOT claim post-training improved the model.** Applying Lucky Loop's own claim discipline (effect must exceed noise), this result is `inconclusive` and the improvement claim is **blocked**.

What this run *does* establish: the LoRA post-training path runs end-to-end on the hackathon GPU — `Qwen3-0.6B`, HF dataset with a clean train/test split, 5.0M trainable params, 37s. It becomes a real candidate experiment the loop can reason about, under the same "verify before claim" gate as every other experiment.
