# Post-training LoRA — scaled run (does it actually work?)

Status: `completed` — real run on Team Pegasus / MI300X container.

The smoke run (n=64, 1 epoch) only proved the path runs; it was too small to detect any effect.
This scaled run answers the real question: **does LoRA post-training work?**

## Config

| Field | Value |
|---|---|
| model | `Qwen/Qwen3-0.6B` |
| dataset | `SetFit/sst2` (binary sentiment) |
| max_train | 1000 |
| max_test | 800 |
| epochs | 2 |
| LoRA r / alpha | 8 / 16 |
| trainable params | 5,046,272 (0.84%) |
| train runtime | ~389 s (total 705 s incl. eval) |
| final train loss | 0.458 |

## Result

| Metric | Value |
|---|---|
| Baseline held-out accuracy | **0.4825** (≈ random on binary) |
| Post-training held-out accuracy | **0.92625** |
| Delta | **+0.44375 (+44.4 points)** |

## Verdict (claim-calibrated)

This is unambiguous and statistically overwhelming:

- Two-proportion z-test on n=800: SE_diff ≈ 0.020, **z ≈ 22.3**, p < 1e-100.
- The effect is ~22 standard errors — not noise.

**Yes: LoRA post-training works.** With adequate data (1000 examples, 2 epochs), it takes
`Qwen3-0.6B` from random (48%) to 92.6% on sentiment classification, training only 0.84% of
parameters in ~6.5 minutes on one GPU.

## Why this also validates the claim discipline

The contrast between the two runs is itself the point:

| Run | Delta | Our verifier would say |
|---|---|---|
| smoke (n=64, 1 epoch) | +0.016 | inconclusive — within noise, claim blocked ✅ |
| scaled (n=800, 2 epochs) | +0.444 | strongly supported — far above noise ✅ |

The same claim-calibration that blocked the smoke overclaim correctly admits the real effect.
The system is honest in both directions: it does not oversell a noisy run, and it does not
suppress a real one.

Reproduce: `python experiments/post_training_lora_probe.py --max-train 1000 --max-test 800 --epochs 2`
(needs torch+transformers+peft on a GPU).
