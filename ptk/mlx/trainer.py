"""MLX QLoRA training via mlx-lm Python APIs."""

from __future__ import annotations

import json
import shutil
import types
from pathlib import Path
from typing import Any

from ptk.config.schema import PTKConfig
from ptk.data.table import TableDict
from ptk.exceptions import RuntimeError as PTKRuntimeError
from ptk.exceptions import ValidationError
from ptk.logging import Logger
from ptk.mlx.availability import require_mlx
from ptk.mlx.data_bridge import write_mlx_dataset
from ptk.mlx.model_cache import ensure_mlx_model
from ptk.training.base_trainer import TrainerResult, resolve_resume_checkpoint


def _lora_scale(r: int, alpha: int) -> float:
    """Map PEFT-style (r, alpha) to mlx-lm LoRA scale (alpha/r)."""
    return float(alpha) / float(max(r, 1))


def _estimate_iters(config: PTKConfig, n_train: int) -> int:
    t = config.training
    if t.max_iters is not None:
        return int(t.max_iters)
    steps_per_epoch = max(1, n_train // max(t.batch_size * t.gradient_accumulation_steps, 1))
    return max(1, steps_per_epoch * t.epochs)


def _num_layers(model: Any, requested: int | None = None) -> int:
    n = len(model.layers)
    if requested is None:
        return min(16, n)
    if requested < 0 or requested > n:
        return n
    return requested


def _log_ignored_bnb_fields(quant, logger: Logger) -> None:
    """MLX uses group quant; bitsandbytes-only fields are ignored."""
    logger.warn(
        "MLX QLoRA ignores quant_type/double_quant (uses bits + group_size group quantization)"
    )


def _write_checkpoint_meta(
    dest: Path,
    *,
    config: PTKConfig,
    mlx_model_path: Path,
    global_step: int,
    lora_parameters: dict[str, Any],
    num_layers: int,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    assert config.training.quantization is not None
    quant = config.training.quantization
    assert config.training.lora is not None
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_parameters,
        "ptk_backend": "mlx_quant",
        "base_model": config.base_model,
    }
    (dest / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2), encoding="utf-8")
    runtime = {
        "ptk_backend": "mlx_quant",
        "base_model": config.base_model,
        "mlx_model_path": str(mlx_model_path),
        "bits": int(quant.bits.value),
        "group_size": int(quant.group_size),
        "global_step": global_step,
        "method": config.method.value,
    }
    (dest / "ptk_runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


def _copy_adapters(src_file: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "adapters.safetensors"
    if src_file.resolve() != target.resolve():
        shutil.copy2(src_file, target)


def train_qlora_mlx(
    config: PTKConfig,
    dataset: TableDict,
    *,
    output_dir: Path,
    logger: Logger,
    resume_from: Path | None = None,
) -> TrainerResult:
    """Run true MLX QLoRA training and return a standard TrainerResult."""
    require_mlx()
    assert config.training.lora is not None
    assert config.training.quantization is not None

    quant = config.training.quantization
    lora_cfg = config.training.lora
    _log_ignored_bnb_fields(quant, logger)

    mlx_model_path, cache_hit = ensure_mlx_model(config, logger=logger)
    logger.start(
        "QLoRA training",
        model=config.base_model,
        quant_backend="mlx_quant",
        mlx_model=str(mlx_model_path),
        mlx_cache_hit=cache_hit,
    )

    data_dir = write_mlx_dataset(
        dataset,
        config.output_path() / "mlx_data",
        batch_size=config.training.batch_size,
    )
    n_train = len(dataset["train"])
    iters = _estimate_iters(config, n_train)

    try:
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
        from mlx_lm.tuner.trainer import TrainingArgs, train
        from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
        from mlx_lm.utils import load
    except ImportError as exc:
        raise PTKRuntimeError(f"MLX training imports failed: {exc}", code="MLX_IMPORT_FAILED") from exc

    model, tokenizer = load(
        str(mlx_model_path),
        tokenizer_config={"trust_remote_code": True},
    )

    # load_local_dataset expects an args-like object with mask_prompt etc.
    ds_args = types.SimpleNamespace(mask_prompt=False)
    train_set, valid_set, _test_set = load_local_dataset(data_dir, tokenizer, ds_args)
    if len(train_set) < config.training.batch_size:
        raise ValidationError(
            f"After filtering empty rows, train set has {len(train_set)} examples "
            f"(batch_size={config.training.batch_size}).",
            field_path="training.batch_size",
        )

    r = int(lora_cfg.r)
    alpha = int(lora_cfg.alpha)
    lora_parameters: dict[str, Any] = {
        "rank": r,
        "dropout": float(lora_cfg.dropout),
        "scale": _lora_scale(r, alpha),
    }
    if lora_cfg.target_modules:
        # mlx-lm keys are module path suffixes within a layer (e.g. self_attn.q_proj).
        lora_parameters["keys"] = list(lora_cfg.target_modules)

    num_layers = _num_layers(model)

    model.freeze()
    linear_to_lora_layers(model, num_layers, lora_parameters, use_dora=False)

    # Resume weights if requested
    resume_path = resolve_resume_checkpoint(output_dir, resume_from, auto_resume=resume_from is not None)
    resume_adapter_file: Path | None = None
    start_step = 0
    if resume_path:
        resume_dir = Path(resume_path)
        candidate = resume_dir / "adapters.safetensors" if resume_dir.is_dir() else resume_dir
        if candidate.exists():
            logger.start("Resuming MLX adapters", path=str(candidate))
            model.load_weights(str(candidate), strict=False)
            resume_adapter_file = candidate
            runtime_file = resume_dir / "ptk_runtime.json" if resume_dir.is_dir() else None
            if runtime_file and runtime_file.exists():
                try:
                    start_step = int(json.loads(runtime_file.read_text(encoding="utf-8")).get("global_step", 0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    start_step = 0

    print_trainable_parameters(model)

    work_adapter_dir = output_dir / "_mlx_work"
    work_adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = work_adapter_dir / "adapters.safetensors"

    training_args = TrainingArgs(
        batch_size=config.training.batch_size,
        iters=iters,
        val_batches=25 if valid_set and len(valid_set) >= config.training.batch_size else 0,
        steps_per_report=max(1, config.training.logging_steps),
        steps_per_eval=max(1, config.training.save_steps),
        steps_per_save=max(1, config.training.save_steps),
        adapter_file=str(adapter_file),
        max_seq_length=config.training.max_seq_length,
        grad_checkpoint=bool(config.training.gradient_checkpointing),
        grad_accumulation_steps=config.training.gradient_accumulation_steps,
    )

    # Avoid empty val dataset issues
    val_dataset = None
    if valid_set and len(valid_set) >= config.training.batch_size and training_args.val_batches != 0:
        val_dataset = CacheDataset(valid_set)

    opt = optim.Adam(learning_rate=config.training.learning_rate)
    mx.random.seed(config.data.seed)

    try:
        train(
            model=model,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=val_dataset,
            args=training_args,
        )
    except Exception as exc:
        raise PTKRuntimeError(
            f"MLX QLoRA training failed: {exc}",
            code="MLX_TRAIN_FAILED",
        ) from exc

    global_step = start_step + iters
    if not adapter_file.exists():
        # train() always saves at end on rank 0; surface a clear error if missing
        raise PTKRuntimeError(
            f"MLX training finished but adapter file missing: {adapter_file}",
            code="MLX_TRAIN_FAILED",
        )

    # Materialize final + last step checkpoints in PTK layout
    final_dir = output_dir / "final"
    _copy_adapters(adapter_file, final_dir)
    _write_checkpoint_meta(
        final_dir,
        config=config,
        mlx_model_path=mlx_model_path,
        global_step=global_step,
        lora_parameters=lora_parameters,
        num_layers=num_layers,
    )

    step_dir = output_dir / f"checkpoint-{global_step}"
    _copy_adapters(adapter_file, step_dir)
    _write_checkpoint_meta(
        step_dir,
        config=config,
        mlx_model_path=mlx_model_path,
        global_step=global_step,
        lora_parameters=lora_parameters,
        num_layers=num_layers,
    )

    # Copy any intermediate mlx step files into PTK checkpoint dirs when present
    for step_file in sorted(work_adapter_dir.glob("*_adapters.safetensors")):
        # names like 0000100_adapters.safetensors
        stem = step_file.name.split("_")[0]
        try:
            step_num = int(stem)
        except ValueError:
            continue
        ckpt = output_dir / f"checkpoint-{step_num}"
        _copy_adapters(step_file, ckpt)
        _write_checkpoint_meta(
            ckpt,
            config=config,
            mlx_model_path=mlx_model_path,
            global_step=step_num,
            lora_parameters=lora_parameters,
            num_layers=num_layers,
        )

    result = TrainerResult(
        output_dir=output_dir,
        checkpoint_path=final_dir,
        global_step=global_step,
        model=None,
        tokenizer=None,
        metrics={},
    )
    logger.complete("QLoRA training complete", step=global_step, quant_backend="mlx_quant")
    return result


def is_mlx_checkpoint(path: Path | None) -> bool:
    """Return True if checkpoint looks like a PTK MLX adapter dir."""
    if path is None:
        return False
    p = Path(path)
    if not p.exists():
        return False
    runtime = p / "ptk_runtime.json"
    if runtime.exists():
        try:
            data = json.loads(runtime.read_text(encoding="utf-8"))
            return data.get("ptk_backend") == "mlx_quant"
        except (json.JSONDecodeError, OSError):
            pass
    adapter_cfg = p / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            data = json.loads(adapter_cfg.read_text(encoding="utf-8"))
            return data.get("ptk_backend") == "mlx_quant" or (
                (p / "adapters.safetensors").exists() and "lora_parameters" in data
            )
        except (json.JSONDecodeError, OSError):
            pass
    return (p / "adapters.safetensors").exists() and not (p / "adapter_model.safetensors").exists()
