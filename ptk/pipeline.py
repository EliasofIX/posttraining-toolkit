"""Full pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import DatasetDict

from ptk.config.schema import DataSource, PTKConfig
from ptk.data.generation import generate_synthetic_data
from ptk.data.pipeline import dataset_stats, hash_dataset, preprocess_dataset
from ptk.distributed.detect import detect_environment
from ptk.eval.harness import EvalHarness
from ptk.export.formats import export_run
from ptk.logging import Logger
from ptk.registry.runs import RunRegistry, RunStatus
from ptk.training.base_trainer import get_trainer


def plan_run(config: PTKConfig) -> dict[str, Any]:
    """Dry-run resource estimates without side effects."""
    env = detect_environment(
        device=config.compute.device,
        strategy=config.compute.strategy,
        nodes=config.compute.nodes,
        gpus_per_node=config.compute.gpus_per_node,
    )

    n_samples = 0
    if config.data.source == DataSource.SYNTHETIC and config.data.synthetic:
        n_samples = config.data.synthetic.n_samples
    elif config.data.dataset:
        try:
            from ptk.data.loaders import load_raw_dataset

            ds = load_raw_dataset(config.data.dataset)
            n_samples = len(ds)
        except Exception:
            n_samples = 0

    t = config.training
    steps_per_epoch = max(1, n_samples // max(t.batch_size * t.gradient_accumulation_steps, 1))
    total_steps = t.max_iters or (steps_per_epoch * t.epochs)

    gpu_hours = total_steps * 0.001 * max(env.num_gpus, 1)
    disk_mb = n_samples * 0.5 + 500

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
        "warnings": env.warnings,
        "output_dir": str(config.output_path()),
    }


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

    registry = RunRegistry()
    if run_id:
        record = registry.validate_resume(run_id, config)
    else:
        record = registry.create_run(config)
        run_id = record.run_id

    env = detect_environment(
        device=config.compute.device,
        strategy=config.compute.strategy,
        nodes=config.compute.nodes,
        gpus_per_node=config.compute.gpus_per_node,
    )

    try:
        dataset_dict = _prepare_data(config, logger, registry, run_id)

        registry.update_run(run_id, status=RunStatus.TRAINING, data_hash=hash_dataset(dataset_dict))
        trainer = get_trainer(config, env, logger)
        result = trainer.train(dataset_dict, resume_from=resume_from)

        registry.update_run(
            run_id,
            status=RunStatus.EVAL if not skip_eval else RunStatus.EXPORT,
            checkpoint_path=str(result.checkpoint_path) if result.checkpoint_path else None,
            global_step=result.global_step,
            metrics=result.metrics,
        )

        if not skip_eval and (config.eval.benchmarks or True):
            harness = EvalHarness(logger)
            harness.run(config, checkpoint_path=result.checkpoint_path)

        registry.update_run(run_id, status=RunStatus.EXPORT)

        if not skip_export:
            export_run(config, checkpoint_path=result.checkpoint_path, logger=logger)

        registry.update_run(run_id, status=RunStatus.COMPLETED)
        logger.complete("Pipeline complete", run_id=run_id)
        return run_id

    except Exception as exc:
        registry.update_run(run_id, status=RunStatus.FAILED, error=str(exc))
        raise


def _prepare_data(config: PTKConfig, logger: Logger, registry: RunRegistry, run_id: str) -> DatasetDict:
    registry.update_run(run_id, status=RunStatus.DATA_GEN)
    data_cache = config.output_path() / "data_cache"

    if config.data.source == DataSource.SYNTHETIC:
        raw = generate_synthetic_data(config, logger)
        data_cache.mkdir(parents=True, exist_ok=True)
        raw.save_to_disk(str(data_cache / "raw"))
        from datasets import DatasetDict as DD
        from datasets import load_from_disk

        loaded = load_from_disk(str(data_cache / "raw"))
        split = loaded.train_test_split(test_size=0.1, seed=config.data.seed)
        return DD({"train": split["train"], "validation": split["test"]})

    logger.start("Loading dataset")
    dataset_dict = preprocess_dataset(config.data, config.method)
    stats = dataset_stats(dataset_dict)
    logger.complete("Dataset ready", **stats)
    return dataset_dict
