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
# Path check is on by default; skip until data exists
ptk validate /tmp/lora.yaml --json --skip-path-check
ptk plan /tmp/lora.yaml --json --skip-path-check
# Example configs resolve data paths relative to the config file:
ptk validate configs/examples/lora.yaml --json
```

Prefer `ptk init` + surgical edits over writing YAML from memory. Sample datasets live in `data/` (examples use `../../data/...`).

### 2.4 Safe Defaults for CI / Smoke Tests

When testing pipeline health, use tiny models and limits:

```yaml
base_model: distilgpt2
training:
  max_iters: 2
  batch_size: 1   # GRPO: see §5.6 — must be compatible with num_generations
  max_seq_length: 64
compute:
  device: cpu
```

For GRPO smoke tests use `batch_size: 2` with `num_generations: 2` (not `batch_size: 1`).

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
VALIDATION_ERROR: Dataset path not found: ...
```

- **When:** `ptk validate` / `ptk plan` / `ptk run` (path check on by default)
- **Skip:** `--skip-path-check` on validate/plan/`data gen`; `ptk run --dry-run` skips (non-dry-run ignores `--skip-path-check`)
- **Resolution:** relative paths resolve **only** against the config file's directory (never cwd). Registry keeps relative paths + `config_dir` for resume.
- **Fix:** Place data relative to the config (or use `data/` / `tests/fixtures/data/`), generate fixtures, or set `data.source: synthetic`

### 5.5 PPO Failures

- PPO always tokenizes the train set to `input_ids` before TRL `PPOTrainer`.
- Policy / reward / value load as **separate** backbones (3× memory vs SFT).
- Reward **score head is randomly initialized** — fine for smoke tests, not a trained RM. Use a real reward model for meaningful RL.
- `reward_model` must share the policy tokenizer; otherwise toolkit falls back to `base_model` with a warning.
- Keep `max_iters` low for smoke tests.
- TRL 1.x uses `trl.experimental.ppo`; `TRL_EXPERIMENTAL_SILENCE=1` is set automatically.

### 5.6 GRPO batch sizing

- `num_generations >= 2` always.
- Single-process (`auto` / `single_gpu`): `batch_size * gradient_accumulation_steps` must be divisible by `num_generations` (schema-enforced).
- `multi_gpu` / `multi_node`: schema skips the batch check; TRL validates `batch * world_size * steps_per_generation` at runtime.
- Example smoke config: `batch_size: 2`, `num_generations: 2`.

### 5.7 Hub Push Failures

Non-fatal. Check `HF_TOKEN` env var. Export artifacts locally via `ptk export`.

### 5.8 Processed Cache / Rank Barrier

- **Symptom:** `TimeoutError: Timed out waiting for processed cache manifest` (exit code 2)
- **Cause:** Non-zero ranks started before rank 0 finished writing `data_cache/processed/<cache_key>/manifest.json`
- **Fix:**
  1. Confirm rank 0 can write to `output.dir` (shared filesystem on multi-node).
  2. Increase wait via `PTK_CACHE_WAIT_SECONDS` env (default 600).
  3. Re-run; cache is content-keyed by dataset mtime + data config hash.

### 5.9 Cache Key Mismatch on Resume

- **Symptom:** Resume re-preprocesses dataset despite prior run on same data
- **Cause:** Dataset file changed (mtime), data filters/split config changed, or different `output.dir`/`run_name` path
- **Fix:** Use identical data config; cache lives under `{output.dir}/{run_name}/data_cache/processed/{cache_key}/`

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
- Milestone/staging plans (prefer concrete CI/runtime facts over roadmap prose)

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
| 2026-07-19 | Path resolve: config-dir only (no cwd bind), keep relative paths in registry + config_dir, reject directory datasets, dry-run-only path skip on run | agent |
| 2026-07-19 | Review follow-ups: YAML empty-list dump, config-relative dataset paths, GRPO multi-GPU schema exemption, PPO reward-head warning, plan/run --skip-path-check | agent |
| 2026-07-19 | CI/training hardening: PPO always-tokenize + separate backbones, DPO use_cpu, GRPO schema batch rules, validate path check, sample data/, method e2e fixtures, dep floors | agent |
| 2026-07-05 | Efficiency overhaul: Parquet cache, rank-0 gating, hardware defaults, DPO fp16, lazy PPO tokenization, DeepSpeed wiring, batched eval | agent |
| 2026-07-03 | Pipeline optimizations: eval skip fix, data cache, DPO memory, distributed launch, dataloader knobs | agent |
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
