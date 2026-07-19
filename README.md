# Posttraining Toolkit

A fully automated posttraining pipeline covering **RL (DPO/PPO/GRPO)**, **SFT**, **LoRA**, and **QLoRA** — designed to be operated identically well by AI agents and experienced humans.

A single declarative YAML/JSON config drives the entire pipeline: from synthetic data generation or bring-your-own-dataset, through training, evaluation, and export — scaling transparently from one GPU to distributed multi-node clusters.

## Quick Start

```bash
pip install -e ".[dev]"

# Scaffold a config (sample datasets live in data/)
ptk init --method sft --output my-config.yaml

# Validate and plan
ptk validate my-config.yaml --json
ptk plan my-config.yaml --json

# Run full pipeline
ptk run my-config.yaml

# Agent mode (JSONL logs, stable exit codes)
ptk run my-config.yaml --machine
```

Example configs in `configs/examples/` point at `../../data/train.jsonl` and `../../data/preferences.jsonl` (resolved relative to the config file).

## Supported Methods

| Method | Description |
|--------|-------------|
| `sft` | Supervised fine-tuning via TRL SFTTrainer |
| `lora` | Low-rank adapter fine-tuning via PEFT |
| `qlora` | Quantized LoRA (bitsandbytes on CUDA, true 4-bit MLX on Apple Silicon) |
| `dpo` | Direct Preference Optimization |
| `ppo` | Proximal Policy Optimization with reward model |
| `grpo` | Group Relative Policy Optimization |

## CLI Reference

| Command | Purpose |
|---------|---------|
| `ptk init [--method X]` | Scaffold a starter config |
| `ptk validate <config>` | Validate config (`--json` for structured errors) |
| `ptk plan <config>` | Dry-run resource estimates |
| `ptk run <config>` | Execute full pipeline |
| `ptk resume <run_id>` | Resume from last checkpoint |
| `ptk data gen <config>` | Synthetic data generation only |
| `ptk eval <run_id>` | Run evaluation harness |
| `ptk export <run_id> --format gguf` | Export trained artifacts |
| `ptk schema --json` | Emit config JSON Schema |
| `ptk list` | List runs in registry |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Validation error |
| 2 | Runtime error |
| 3 | Resume config drift |

## Configuration

See `configs/examples/` for method-specific templates. Get the full JSON Schema:

```bash
ptk schema --json > schema.json
```

Minimal SFT config:

```yaml
run_name: my-sft-run
base_model: distilgpt2
method: sft

data:
  source: dataset
  dataset:
    path: ./data/train.jsonl
    format: jsonl
    text_column: text

training:
  epochs: 3
  learning_rate: 2.0e-5
  batch_size: 4

output:
  dir: ./outputs
```

## Dependencies

Core runtime dependencies are limited to the Hugging Face training stack:

- `torch`, `transformers`, `trl`, `peft`, `pydantic`

The toolkit implements its own YAML config I/O, CLI (`argparse`), terminal logging, in-memory data tables, safetensors reader, and Hugging Face Hub client. Optional extras:

- `cuda` — bitsandbytes + DeepSpeed
- `mlx` — true 4-bit QLoRA on Apple Silicon via MLX / mlx-lm
- `gguf` — GGUF export
- `parquet` — Parquet dataset loading (`pyarrow`)

## Hardware Support

- **NVIDIA CUDA** — full feature set including bitsandbytes QLoRA, DeepSpeed, multi-GPU/multi-node
- **Apple Silicon** — SFT, LoRA, and **true 4-bit QLoRA via MLX** when the `mlx` extra is installed
- **CPU** — smoke tests and tiny-model development (QLoRA runs unquantized LoRA)

Install extras:

```bash
pip install -e ".[cuda]"   # NVIDIA QLoRA + DeepSpeed
pip install -e ".[mlx]"    # Apple Silicon true 4-bit QLoRA
```

On Apple Silicon, `method: qlora` with `training.quantization.backend: auto` selects the MLX backend (quantize with `mlx_lm.convert`, train adapters with mlx-lm). Without `.[mlx]`, QLoRA on MPS fails closed with install instructions (set `allow_unquantized_fallback: true` only for legacy fp16 PEFT smoke tests). See `configs/examples/qlora-mlx.yaml`.

## Synthetic Data Generation

```yaml
data:
  source: synthetic
  synthetic:
    backend: self_instruct  # anthropic | openai | self_instruct
    seed_prompts: ./prompts.txt
    n_samples: 1000
```

API backends require `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

## Run Registry & Resume

Every run gets a UUID with persisted metadata. Resume with:

```bash
ptk resume <run_id>
```

Registry stored in `.ptk/registry/` by default. Set `PTK_S3_BUCKET` for S3-compatible storage.

## Testing

```bash
pip install -e ".[dev,gguf,parquet]"
pytest tests/ -v
pytest tests/test_integration.py tests/test_gguf_export.py -v -m slow
```

CI runs on push/PR via GitHub Actions (unit tests on Python 3.10–3.12, integration tests, optional macOS MPS smoke).

## Project Structure

```
ptk/
├── cli.py              # ptk entrypoint
├── config/             # Pydantic schema, defaults, JSON Schema export
├── data/               # loaders, pipeline, synthetic generators
├── training/           # SFT, LoRA, QLoRA, DPO, PPO, GRPO
├── distributed/        # hardware detection, accelerate/deepspeed configs
├── eval/               # benchmark harness
├── registry/           # run metadata and resumability
└── export/             # merge adapters, GGUF, Hub push
```

## License

MIT — see [LICENSE](LICENSE).

## For AI Agents

See [AGENTS.md](AGENTS.md) for machine-oriented operating instructions, error recovery patterns, and self-improvement protocol.
