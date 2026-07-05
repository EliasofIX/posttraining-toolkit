"""Apply hardware-aware defaults to training configuration."""

from __future__ import annotations

from ptk.config.schema import DeviceType, PTKConfig, TrainingMethod
from ptk.distributed.detect import ComputeEnvironment, detect_environment


def apply_hardware_defaults(
    config: PTKConfig,
    env: ComputeEnvironment | None = None,
) -> PTKConfig:
    """Fill unset dataloader and memory defaults based on detected hardware."""
    if env is None:
        env = detect_environment(
            device=config.compute.device,
            strategy=config.compute.strategy,
            nodes=config.compute.nodes,
            gpus_per_node=config.compute.gpus_per_node,
        )

    training_updates: dict = {}
    t = config.training

    if env.device == DeviceType.CUDA:
        if t.dataloader_num_workers is None:
            training_updates["dataloader_num_workers"] = 4
        if t.dataloader_pin_memory is None:
            training_updates["dataloader_pin_memory"] = True
        workers = t.dataloader_num_workers if t.dataloader_num_workers is not None else 4
        if workers > 0:
            if t.dataloader_prefetch_factor is None:
                training_updates["dataloader_prefetch_factor"] = 2
            if t.dataloader_persistent_workers is None:
                training_updates["dataloader_persistent_workers"] = True

    if config.method in (
        TrainingMethod.QLORA,
        TrainingMethod.DPO,
        TrainingMethod.PPO,
        TrainingMethod.GRPO,
    ):
        # Only recommend gradient checkpointing via method defaults in defaults.py.
        # Do not override an explicit training.gradient_checkpointing: false.
        pass

    if not training_updates:
        return config

    return config.model_copy(update={"training": t.model_copy(update=training_updates)})
