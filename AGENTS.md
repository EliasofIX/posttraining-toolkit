# AGENTS.md — Posttraining Toolkit Agent Operating Manual

> **Purpose:** This document is the authoritative interface for AI agents operating `posttraining-toolkit` (`ptk`). It is designed to be **self-improving**: agents MUST update this file when they discover gaps, failures, or better patterns — subject to the improvement protocol in §8.

---

## 0. Agent Identity & Scope

You are an autonomous operator of the Posttraining Toolkit. Your job:

1. Generate valid configs from schema (not prose).
2. Validate before executing (`ptk validate`, `ptk plan`).
3. Execute, monitor, recover from failures.
4. Record learnings back into this file.

**Do not** read README prose to infer config fields — use `ptk schema --json`.

---

## 1. First Actions (Every Session)

```bash
# 1. Confirm install
ptk --version

# 2. Pull machine-readable schema
ptk schema --json > /tmp/ptk-schema.json

# 3. Detect hardware
python -c "import torch; print({'cuda': torch.cuda.is_available(), 'mps': getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()})"
```

Always run `ptk validate` and `ptk plan` before `ptk run`. Never skip validation on configs you generated.

---

## 2. Config Generation Rules

### 2.1 Required Fields

Every config MUST include:

| Field | Type | Notes |
|-------|------|-------|
| `run_name` | string | No path separators |
| `base_model` | string | HF repo ID or local path |
| `method` | enum | `sft\|lora\|qlora\|dpo\|ppo\|grpo` |

### 2.2 Method-Specific Requirements

| Method | Required nested config |
|--------|----------------------|
| `lora`, `qlora` | `training.lora` (auto-filled if omitted) |
| `qlora` | `training.quantization` (auto-filled if omitted) |
| `dpo`, `ppo`, `grpo` | `training.rl` (auto-filled if omitted) |
| `ppo` | `training.rl.reward_model` (**required**, no default) |
| `dpo` | dataset with `prompt`, `chosen`, `rejected` columns |
| `dataset` source | `data.dataset.path` must exist |

### 2.3 Scaffold Instead of Guessing

```bash
ptk init --method lora --output /tmp/lora.yaml --base-model distilgpt2
ptk validate /tmp/lora.yaml --json
```

Prefer `ptk init` + surgical edits over writing YAML from memory.

### 2.4 Safe Defaults for CI / Smoke Tests

When testing pipeline health, use tiny models and limits:

```yaml
base_model: distilgpt2
training:
  max_iters: 2
  batch_size: 1
  max_seq_length: 64
compute:
  device: cpu
```

---

## 3. CLI Contract (Machine Mode)

**Always use `--machine` in autonomous runs.** Output is JSONL (one JSON object per line).

### 3.1 Exit Codes (Stable — Do Not Rely on stderr parsing)

| Code | Constant | Meaning | Recovery |
|------|----------|---------|----------|
| 0 | `EXIT_SUCCESS` | Success | Continue |
| 1 | `EXIT_VALIDATION` | Config/input invalid | Fix config per structured error |
| 2 | `EXIT_RUNTIME` | Training/pipeline failure | See §5 |
| 3 | `EXIT_RESUME_DRIFT` | Resume config ≠ original | Use original config or start new run |

### 3.2 Structured Validation Errors

```bash
ptk validate config.yaml --json
# {"valid": true, "run_name": "...", "method": "..."}
# OR
# {"error": {"code": "VALIDATION_ERROR", "message": "...", "details": {"errors": [...]}}}
```

Each error in `details.errors` has `field_path`, `message`, `type`.

### 3.3 Planning Output

```bash
ptk plan config.yaml --json
```

Key fields: `device`, `strategy`, `data_samples`, `estimated_steps`, `estimated_gpu_hours`, `warnings`.

**Abort if** `warnings` contains blocking issues (e.g., requesting multi-GPU on MPS).

---

## 4. Standard Operating Procedures

### 4.1 Full Pipeline

```bash
ptk validate config.yaml --json || exit 1
ptk plan config.yaml --json > /tmp/plan.json
ptk run config.yaml --machine 2>&1 | tee /tmp/run.jsonl
ptk list --json
```

### 4.2 Resume Interrupted Run

```bash
ptk list --json                          # find run_id, status=interrupted|training
ptk resume <run_id> --machine
```

Resume requires **identical config** to original. Hash mismatch → exit 3.

### 4.3 Synthetic Data Only

```bash
# API backends need keys in env
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY
ptk data gen config.yaml --machine
```

`self_instruct` backend works offline — preferred for air-gapped environments.

### 4.4 Eval & Export

```bash
ptk eval <run_id> --benchmarks perplexity,generation_quality --machine
ptk export <run_id> --format merged_fp16 --machine
```

Export formats: `adapter_only`, `merged_fp16`, `gguf`.

Install GGUF support:

```bash
pip install -e ".[gguf]"
```

---

## 5. Error Recovery Playbook

### 5.1 OOM (Out of Memory)

1. Reduce `training.batch_size` (halve iteratively).
2. Increase `training.gradient_accumulation_steps` to maintain effective batch.
3. For QLoRA: confirm `quantization.bits: 4`.
4. Reduce `training.max_seq_length`.
5. Set `training.max_iters` for smoke validation before full epochs.

### 5.2 MPS / Apple Silicon

- `compute.strategy: multi_gpu` or `multi_node` → **validation error**. Use `auto` or `single_gpu`.
- QLoRA: bitsandbytes unavailable → toolkit falls back to fp16 LoRA with warning. This is expected.
- If NaN losses: set `compute.mixed_precision: fp32`.

### 5.3 CUDA / Multi-GPU

```bash
pip install -e ".[cuda]"
```

For multi-GPU, set `compute.strategy: multi_gpu`. Toolkit auto-generates accelerate config.

DeepSpeed is CUDA-only. Set `compute.deepspeed_config` to override auto-generation.

### 5.4 Missing Dataset

```
VALIDATION_ERROR: data.dataset.path not found
```

Generate fixture data or switch to `data.source: synthetic`.

### 5.5 PPO Failures

- `reward_model` must share the policy tokenizer vocabulary (same tokenizer IDs). If you configure a different `reward_model`, the toolkit falls back to the `base_model` backbone with a warning.
- Keep `max_iters` low for smoke tests (PPO is slow).
- TRL 1.x uses `trl.experimental.ppo`; ensure `TRL_EXPERIMENTAL_SILENCE=1` is set (handled automatically by `ptk`).

### 5.6 GRPO batch sizing

- `training.batch_size` × `training.rl.num_generations` must satisfy TRL constraints: `generation_batch_size` divisible by `num_generations`, and `num_generations >= 2`.
- Example smoke config: `batch_size: 2`, `num_generations: 2`.

### 5.7 Hub Push Failures

Non-fatal. Check `HF_TOKEN` env var. Export artifacts locally via `ptk export`.

---

## 6. Hardware Decision Matrix

| Detected Device | Strategy auto-resolves to | QLoRA backend | Distributed |
|-----------------|--------------------------|---------------|-------------|
| CUDA + 1 GPU | `single_gpu` | `bnb` | accelerate |
| CUDA + N GPUs | `multi_gpu` | `bnb` | accelerate |
| MPS | `single_gpu` | `mlx_quant` (fp16 fallback) | single device only |
| CPU | `single_gpu` | none (full precision) | none |

Always verify with `ptk plan --json` before committing GPU hours.

---

## 7. Registry & State

- Local registry: `.ptk/registry/`
- S3 registry: set `PTK_S3_BUCKET`, optional `PTK_S3_ENDPOINT`, `PTK_S3_PREFIX`
- Each run stores: config snapshot, git commit, data hash, checkpoint path, metrics

**Idempotency:** Re-running with same `run_name` creates a new UUID. Do not assume run_name is unique.

---

## 8. Self-Improvement Protocol

> This section is how AGENTS.md stays current. **Follow it every time you learn something new.**

### 8.1 When to Update

Update AGENTS.md when you:

- Hit an error not documented in §5
- Find a config field behaves differently than expected
- Discover a better recovery procedure
- Add a new CLI flag, method, or benchmark
- Identify a misleading or missing schema constraint

### 8.2 How to Update

1. Add an entry to **§9 Changelog** (bottom of file) with date and one-line summary.
2. Edit the relevant section (§2–§7). Do not duplicate — integrate.
3. If adding a new error pattern, use this template:

```markdown
### 5.X <Error Name>
- **Symptom:** <exit code, log event, error code>
- **Cause:** <root cause>
- **Fix:** <numbered steps>
- **Discovered by:** <agent/session id, date>
```

4. Keep prose minimal and actionable. Prefer tables and command blocks.
5. Commit AGENTS.md changes with prefix `docs(agents):`.

### 8.3 What NOT to Add

- Session-specific paths (`/tmp/my-run-47`)
- Secrets or API keys
- Prose duplicates of `ptk schema --json`
- Milestone/staging plans (this project ships complete)

### 8.4 Validation After Update

After editing AGENTS.md, verify commands still work:

```bash
ptk --version
ptk schema --json | python -c "import json,sys; json.load(sys.stdin); print('schema ok')"
ptk validate tests/fixtures/sft.yaml --json
```

---

## 9. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-07-01 | GGUF native GPT-2 converter, PPO/GRPO e2e tests, GitHub Actions CI | bootstrap |
| 2026-07-01 | Initial AGENTS.md — full toolkit v0.1 | bootstrap |

---

## 10. Quick Reference Card

```
SCHEMA    → ptk schema --json
SCAFFOLD  → ptk init --method <m> -o config.yaml
VALIDATE  → ptk validate config.yaml --json
PLAN      → ptk plan config.yaml --json
RUN       → ptk run config.yaml --machine
RESUME    → ptk resume <run_id> --machine
EVAL      → ptk eval <run_id> --machine
EXPORT    → ptk export <run_id> -f merged_fp16 --machine
LIST      → ptk list --json
```

**Golden rule:** Schema → Validate → Plan → Run. Never invert this order.

---

## 11. Cursor Cloud specific instructions

Durable notes for agents running in the Cursor Cloud VM (dependencies already installed by the startup update script `pip install -e ".[dev,gguf]"`).

- **Runtime:** Python 3.12, **CPU-only** (no CUDA/MPS). `ptk plan` resolves `device: cpu`. Use CPU smoke configs (`tests/fixtures/sft.yaml` is a ready-made SFT smoke on `distilgpt2`). The `[cuda]` extra (bitsandbytes/DeepSpeed) is not installed and not usable here.
- **CLI on PATH:** `ptk` (and `pytest`, `ruff`) install to `~/.local/bin`, which is already on PATH — no activation step needed.
- **No server/GUI:** `ptk` is a batch CLI; there is nothing to "serve". A working demo = a completed `ptk run` (data → train → eval → export) plus the run showing `status: completed` in `ptk list --json`.
- **Network:** HF Hub is reachable, so base models like `distilgpt2` download on first `ptk run`.
- **Standard commands** (from `.github/workflows/ci.yml`): lint `ruff check ptk tests`; unit tests `pytest tests/ -m "not slow"` (fast, ~3s); slow integration `pytest tests/test_integration.py tests/test_gguf_export.py -m slow` (SFT/PPO/GRPO/GGUF on CPU).
- **Lint caveat:** `ruff` is pinned only `>=0.3.0`, so the VM installs a newer ruff (0.15.x) that flags a pre-existing import-order issue in `tests/test_hub_client.py` (`ruff check` exits 1). This is not caused by environment setup; do not "fix" it as part of setup.
- **Outputs are gitignored:** `test_outputs/` and `.ptk/` (registry) are produced by runs and ignored by git — safe to leave in place between runs.
