"""Full pipeline orchestration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ptk.config.hardware_defaults import apply_hardware_defaults
from ptk.config.loader import config_to_yaml
from ptk.config.schema import ComputeStrategy, DataSource, PTKConfig
from ptk.data.generation import generate_synthetic_data
from ptk.data.pipeline import (
    compute_data_cache_key,
    dataset_stats,
    hash_dataset,
    load_processed_cache,
    preprocess_dataset,
    save_processed_cache,
)
from ptk.data.table import TableDict
from ptk.distributed.accelerate_gen import generate_accelerate_config
from ptk.distributed.deepspeed_gen import generate_deepspeed_config
from ptk.distributed.detect import (
    detect_environment,
    distributed_barrier,
    is_main_process,
    wait_for_cache_manifest,
)
from ptk.eval.harness import EvalHarness
from ptk.export.formats import export_run
from ptk.logging import Logger
from ptk.registry.runs import RunRegistry, RunStatus
from ptk.training.base_trainer import get_trainer, resolve_resume_checkpoint


def plan_run(config: PTKConfig) -> dict[str, Any]:
    """Dry-run resource estimates without side effects."""
    env = detect_environment(
        device=config.compute.device,
        strategy=config.compute.strategy,
        nodes=config.compute.nodes,
        gpus_per_node=config.compute.gpus_per_node,
    )
    config = apply_hardware_defaults(config, env)

    n_samples = 0
    cache_hit = False
    if config.data.source == DataSource.SYNTHETIC and config.data.synthetic:
        n_samples = config.data.synthetic.n_samples
    elif config.data.dataset:
        try:
            from ptk.data.loaders import count_dataset_samples

            n_samples = count_dataset_samples(config.data.dataset)
        except Exception:
            n_samples = 0

        cache_key = compute_data_cache_key(config.data, config.method)
        cache_dir = config.output_path() / "data_cache" / "processed" / cache_key
        cache_hit = (cache_dir / ".ready").exists() and (cache_dir / "manifest.json").exists()

    t = config.training
    steps_per_epoch = max(1, n_samples // max(t.batch_size * t.gradient_accumulation_steps, 1))
    total_steps = t.max_iters or (steps_per_epoch * t.epochs)

    gpu_hours = total_steps * 0.001 * max(env.num_gpus, 1)
    disk_mb = n_samples * 0.5 + 500
    preprocess_seconds = 0.0 if cache_hit else round(max(n_samples, 1) * 0.0001, 2)

    return {
        "run_name": config.run_name,
        "method": config.method.value,
        "device": env.device.value,
        "strategy": env.strategy.value,
        "num_gpus": env.num_gpus,
        "num_nodes": env.num_nodes,
        "data_samples": n_samples,
        "estimated_steps": total_steps,
        "estimated_gpu_hours": round(gpu_hours, 3),
        "estimated_disk_mb": round(disk_mb, 1),
        "estimated_preprocess_seconds": preprocess_seconds,
        "data_cache_hit": cache_hit,
        "warnings": env.warnings,
        "output_dir": str(config.output_path()),
    }


def _is_distributed_active() -> bool:
    return (
        os.environ.get("PTK_DISTRIBUTED_ACTIVE") == "1"
        or os.environ.get("LOCAL_RANK") is not None
        or int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


def _should_launch_distributed(env) -> bool:
    return (
        env.strategy in (ComputeStrategy.MULTI_GPU, ComputeStrategy.MULTI_NODE)
        and env.num_gpus > 1
        and not _is_distributed_active()
        and shutil.which("accelerate") is not None
    )


def _ensure_deepspeed_config(config: PTKConfig, env, output_dir: Path) -> PTKConfig:
    """Auto-generate DeepSpeed config for multi-GPU when not explicitly set."""
    if config.compute.deepspeed_config:
        return config
    if env.strategy not in (ComputeStrategy.MULTI_GPU, ComputeStrategy.MULTI_NODE):
        return config
    if env.device.value != "cuda":
        return config

    ds_path = output_dir / "deepspeed_config.json"
    generate_deepspeed_config(
        env,
        ds_path,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        train_batch_size=config.training.batch_size,
    )
    compute = config.compute.model_copy(update={"deepspeed_config": str(ds_path)})
    return config.model_copy(update={"compute": compute})


def _launch_distributed(
    config: PTKConfig,
    logger: Logger,
    *,
    run_id: str | None,
    resume_from: Path | None,
    skip_eval: bool,
    skip_export: bool,
) -> str:
    """Spawn accelerate launch for multi-GPU training."""
    registry = RunRegistry()
    if run_id:
        registry.validate_resume(run_id, config)
    else:
        run_id = registry.create_run(config).run_id

    env = detect_environment(
        device=config.compute.device,
        strategy=config.compute.strategy,
        nodes=config.compute.nodes,
        gpus_per_node=config.compute.gpus_per_node,
    )
    config = apply_hardware_defaults(config, env)
    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _ensure_deepspeed_config(config, env, output_dir)

    if config.compute.accelerate_config:
        accel_cfg = Path(config.compute.accelerate_config)
    else:
        accel_cfg = output_dir / "accelerate_config.yaml"
        deepspeed_path = Path(config.compute.deepspeed_config) if config.compute.deepspeed_config else None
        generate_accelerate_config(env, accel_cfg, deepspeed_config=deepspeed_path)

    config_path = output_dir / "pipeline_config.yaml"
    config_path.write_text(config_to_yaml(config), encoding="utf-8")

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        str(accel_cfg),
        "-m",
        "ptk.cli",
        "run",
        str(config_path),
    ]
    if skip_eval:
        cmd.append("--skip-eval")
    if skip_export:
        cmd.append("--skip-export")
    if logger.machine:
        cmd.append("--machine")

    child_env = os.environ.copy()
    child_env["PTK_DISTRIBUTED_ACTIVE"] = "1"
    if run_id:
        child_env["PTK_RUN_ID"] = run_id
    if resume_from:
        child_env["PTK_RESUME_FROM"] = str(resume_from)

    logger.start("Launching distributed training", command=" ".join(cmd))
    result = subprocess.run(cmd, env=child_env, check=False)
    if result.returncode != 0:
        registry.update_run(run_id, status=RunStatus.FAILED, error=f"exit code {result.returncode}")
        raise RuntimeError(f"Distributed launch failed with exit code {result.returncode}")

    return run_id


def run_pipeline(
    config: PTKConfig,
    logger: Logger,
    *,
    run_id: str | None = None,
    resume_from: Path | None = None,
    dry_run: bool = False,
    skip_eval: bool = False,
    skip_export: bool = False,
) -> str:
    """Execute full pipeline: data → train → eval → export."""
    if dry_run:
        plan = plan_run(config)
        logger.complete("Dry run complete", **plan)
        return run_id or "dry-run"

    run_id = run_id or os.environ.get("PTK_RUN_ID")
    resume_from = resume_from or (
        Path(os.environ["PTK_RESUME_FROM"]) if os.environ.get("PTK_RESUME_FROM") else None
    )

    env = detect_environment(
        device=config.compute.device,
        strategy=config.compute.strategy,
        nodes=config.compute.nodes,
        gpus_per_node=config.compute.gpus_per_node,
    )
    config = apply_hardware_defaults(config, env)

    if _should_launch_distributed(env):
        return _launch_distributed(
            config,
            logger,
            run_id=run_id,
            resume_from=resume_from,
            skip_eval=skip_eval,
            skip_export=skip_export,
        )

    registry = RunRegistry()
    if run_id:
        record = registry.validate_resume(run_id, config)
    else:
        record = registry.create_run(config)
        run_id = record.run_id

    try:
        existing_hash = record.data_hash if run_id and record else None
        dataset_dict = _prepare_data(config, logger, existing_data_hash=existing_hash)
        distributed_barrier()

        content_hash = hash_dataset(dataset_dict)
        if is_main_process():
            registry.update_run(run_id, status=RunStatus.TRAINING, data_hash=content_hash)

        resume_requested = resume_from is not None or os.environ.get("PTK_RESUME_FROM") is not None
        if resume_from is None and resume_requested and run_id:
            resume_from = registry.find_latest_checkpoint(run_id)

        trainer = get_trainer(config, env, logger)
        result = trainer.train(dataset_dict, resume_from=resume_from)

        if is_main_process():
            registry.update_run(
                run_id,
                status=RunStatus.EVAL if not skip_eval and config.eval.benchmarks else RunStatus.EXPORT,
                checkpoint_path=str(result.checkpoint_path) if result.checkpoint_path else None,
                global_step=result.global_step,
                metrics=result.metrics,
            )

        if not skip_eval and config.eval.benchmarks and is_main_process():
            harness = EvalHarness(logger)
            if result.model is not None and result.tokenizer is not None:
                eval_results = harness.run_with_model(config, result.model, result.tokenizer)
                _save_eval_results(config, eval_results)
                result.model = None
                result.tokenizer = None
            else:
                eval_results = harness.run(config, checkpoint_path=result.checkpoint_path)
                _save_eval_results(config, eval_results)

        distributed_barrier()

        if is_main_process():
            registry.update_run(run_id, status=RunStatus.EXPORT)

            if not skip_export:
                export_run(
                    config,
                    checkpoint_path=result.checkpoint_path,
                    logger=logger,
                    model=result.model,
                    tokenizer=result.tokenizer,
                )

            registry.update_run(run_id, status=RunStatus.COMPLETED)
            logger.complete("Pipeline complete", run_id=run_id)

        distributed_barrier()
        return run_id

    except Exception as exc:
        if is_main_process():
            registry.update_run(run_id, status=RunStatus.FAILED, error=str(exc))
        raise


def _save_eval_results(config: PTKConfig, results: dict[str, Any]) -> None:
    out_path = config.output_path() / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _prepare_data(
    config: PTKConfig,
    logger: Logger,
    *,
    existing_data_hash: str | None = None,
) -> TableDict:
    cache_key = compute_data_cache_key(config.data, config.method)
    cache_dir = config.output_path() / "data_cache" / "processed" / cache_key

    if not is_main_process():
        wait_for_cache_manifest(cache_dir)
        cached = load_processed_cache(cache_dir, expected_hash=existing_data_hash)
        if cached is None:
            raise RuntimeError(f"Rank > 0 could not load processed cache from {cache_dir}")
        return cached

    if existing_data_hash:
        cached = load_processed_cache(cache_dir, expected_hash=existing_data_hash)
        if cached is not None:
            stats = dataset_stats(cached)
            logger.complete("Loaded cached dataset", **stats, hash=existing_data_hash)
            return cached

    cached = load_processed_cache(cache_dir)
    if cached is not None:
        stats = dataset_stats(cached)
        logger.complete("Loaded cached dataset", **stats, cache_key=cache_key)
        return cached

    if config.data.source == DataSource.SYNTHETIC:
        raw = generate_synthetic_data(config, logger)
        split = raw.train_test_split(test_size=1.0 - config.data.train_split, seed=config.data.seed)
        dataset_dict = TableDict({"train": split["train"], "validation": split["test"]})
        content_hash = hash_dataset(dataset_dict)
        save_processed_cache(cache_dir, dataset_dict, content_hash, config.method.value, cache_key=cache_key)
        return dataset_dict

    logger.start("Loading dataset")
    dataset_dict = preprocess_dataset(
        config.data,
        config.method,
        num_proc=config.training.dataset_num_proc,
    )
    stats = dataset_stats(dataset_dict)
    logger.complete("Dataset ready", **stats)

    content_hash = hash_dataset(dataset_dict)
    save_processed_cache(cache_dir, dataset_dict, content_hash, config.method.value, cache_key=cache_key)
    return dataset_dict
