# Post-training LoRA probe

Status: `missing_dependencies`

## Config

| Field | Value |
|---|---|
| `model` | `Qwen/Qwen3-0.6B` |
| `dataset` | `SetFit/sst2` |
| `text_field` | `text` |
| `label_field` | `label` |
| `max_train` | `256` |
| `max_test` | `128` |
| `lora_r` | `8` |
| `lora_alpha` | `16` |
| `learning_rate` | `0.0002` |
| `epochs` | `1.0` |
| `output_dir` | `/Users/nevil/dev/sandbox/lucky-loop/reports/post_training_lora_probe` |

## Result

The probe is ready, but the local environment is missing required ML dependencies.

Missing: `transformers`, `datasets`, `peft`
