"""Shared trainer interface and utilities."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase, TrainingArguments

from ptk.config.schema import PTKConfig, TrainingConfig
from ptk.data.table import TableDict
from ptk.distributed.detect import ComputeEnvironment, resolve_mixed_precision, torch_device_string
from ptk.logging import Logger


@dataclass
class TrainerResult:
    """Standardized training result."""

    output_dir: Path
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    global_step: int = 0
    model: Any | None = field(default=None, repr=False)
    tokenizer: PreTrainedTokenizerBase | None = field(default=None, repr=False)


class BaseTrainer(ABC):
    """Shared trainer interface for all methods."""

    def __init__(
        self,
        config: PTKConfig,
        env: ComputeEnvironment,
        logger: Logger,
    ) -> None:
        self.config = config
        self.env = env
        self.logger = logger
        self.output_dir = config.output_path() / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def train(self, dataset: TableDict, *, resume_from: Path | None = None) -> TrainerResult:
        """Execute training and return result."""

    def build_training_args(self, **overrides: Any) -> TrainingArguments:
        """Build HuggingFace TrainingArguments from config."""
        t = self.config.training
        precision = resolve_mixed_precision(self.env.device, self.config.compute.mixed_precision)
        fp16 = precision == "fp16"
        bf16 = precision == "bf16"

        args = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": t.epochs,
            "per_device_train_batch_size": t.batch_size,
            "per_device_eval_batch_size": t.batch_size,
            "gradient_accumulation_steps": t.gradient_accumulation_steps,
            "learning_rate": t.learning_rate,
            "warmup_ratio": t.warmup_ratio,
            "weight_decay": t.weight_decay,
            "logging_steps": t.logging_steps,
            "save_steps": t.save_steps,
            "eval_strategy": "steps",
            "eval_steps": t.save_steps,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "report_to": "none",
            "fp16": fp16,
            "bf16": bf16,
            "max_steps": t.max_iters if t.max_iters else -1,
            "remove_unused_columns": False,
            "gradient_checkpointing": t.gradient_checkpointing,
            **dataloader_kwargs(t),
        }
        args.update(overrides)
        return TrainingArguments(**args)

    def build_sft_config_kwargs(self, *, has_validation: bool = False) -> dict[str, Any]:
        """Shared SFTConfig kwargs for SFT, LoRA, and QLoRA trainers."""
        t = self.config.training
        precision = resolve_mixed_precision(self.env.device, self.config.compute.mixed_precision)
        device = torch_device_string(self.env.device)
        run_mid_training_eval = has_validation and bool(self.config.eval.benchmarks)
        eval_steps = self.config.eval.eval_steps if run_mid_training_eval else t.save_steps
        return {
            "output_dir": str(self.output_dir),
            "num_train_epochs": t.epochs,
            "per_device_train_batch_size": t.batch_size,
            "per_device_eval_batch_size": t.batch_size,
            "gradient_accumulation_steps": t.gradient_accumulation_steps,
            "learning_rate": t.learning_rate,
            "warmup_ratio": t.warmup_ratio,
            "weight_decay": t.weight_decay,
            "logging_steps": t.logging_steps,
            "save_steps": t.save_steps,
            "save_strategy": t.save_strategy,
            "eval_strategy": "steps" if run_mid_training_eval else "no",
            "eval_steps": eval_steps,
            "save_total_limit": 3,
            "load_best_model_at_end": False,
            "report_to": "none",
            "max_steps": t.max_iters if t.max_iters else -1,
            "max_length": t.max_seq_length,
            "dataset_text_field": "text",
            "use_cpu": device == "cpu",
            "fp16": precision == "fp16",
            "bf16": precision == "bf16",
            "gradient_checkpointing": t.gradient_checkpointing,
            **dataloader_kwargs(t),
            **deepspeed_kwargs(self.config),
        }

    def save_training_metadata(self, result: TrainerResult) -> None:
        """Persist training metadata for registry."""
        meta_path = self.config.output_path() / "training_meta.json"
        meta = {
            "method": self.config.method.value,
            "metrics": result.metrics,
            "global_step": result.global_step,
            "checkpoint": str(result.checkpoint_path) if result.checkpoint_path else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def is_distributed(env: ComputeEnvironment) -> bool:
    """Return True when running under accelerate/torchrun."""
    if os.environ.get("LOCAL_RANK") is not None:
        return True
    if os.environ.get("PTK_DISTRIBUTED_ACTIVE") == "1":
        return True
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def place_model(model: Any, env: ComputeEnvironment) -> Any:
    """Move model to device only for single-process runs."""
    if is_distributed(env):
        return model
    device = torch_device_string(env.device)
    if device != "cpu":
        model.to(device)
    return model


def dataloader_kwargs(training: TrainingConfig) -> dict[str, Any]:
    """Map training config dataloader fields to TRL/HF kwargs."""
    kwargs: dict[str, Any] = {}
    if training.dataloader_num_workers is not None:
        kwargs["dataloader_num_workers"] = training.dataloader_num_workers
    if training.dataloader_pin_memory is not None:
        kwargs["dataloader_pin_memory"] = training.dataloader_pin_memory
    if training.dataloader_prefetch_factor is not None:
        kwargs["dataloader_prefetch_factor"] = training.dataloader_prefetch_factor
    if training.dataloader_persistent_workers is not None:
        kwargs["dataloader_persistent_workers"] = training.dataloader_persistent_workers
    return kwargs


def deepspeed_kwargs(config: PTKConfig) -> dict[str, Any]:
    """Pass DeepSpeed config path to TRL/HF when configured."""
    if config.compute.deepspeed_config:
        return {"deepspeed": config.compute.deepspeed_config}
    return {}


def resolve_resume_checkpoint(output_dir: Path, resume_from: Path | None = None) -> str | None:
    """Resolve the best checkpoint path for resume."""
    if resume_from is not None and resume_from.exists():
        return str(resume_from)

    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", maxsplit=1)[-1]),
    )
    if checkpoints:
        return str(checkpoints[-1])

    final = output_dir / "final"
    if final.exists():
        return str(final)
    return None


def dataset_map_kwargs(training: TrainingConfig) -> dict[str, Any]:
    """Kwargs for HuggingFace Dataset.map multiprocessing."""
    if training.dataset_num_proc is None:
        return {}
    return {"num_proc": training.dataset_num_proc}


def get_trainer(config: PTKConfig, env: ComputeEnvironment, logger: Logger) -> BaseTrainer:
    """Factory for method-specific trainers."""
    method = config.method.value
    if method == "sft":
        from ptk.training.sft import SFTTrainerWrapper

        return SFTTrainerWrapper(config, env, logger)
    if method == "lora":
        from ptk.training.lora import LoRATrainer

        return LoRATrainer(config, env, logger)
    if method == "qlora":
        from ptk.training.qlora import QLoRATrainer

        return QLoRATrainer(config, env, logger)
    if method == "dpo":
        from ptk.training.rl.dpo import DPOTrainerWrapper

        return DPOTrainerWrapper(config, env, logger)
    if method == "ppo":
        from ptk.training.rl.ppo import PPOTrainerWrapper

        return PPOTrainerWrapper(config, env, logger)
    if method == "grpo":
        from ptk.training.rl.grpo import GRPOTrainerWrapper

        return GRPOTrainerWrapper(config, env, logger)
    raise ValueError(f"Unknown training method: {method}")


def prepare_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load and configure tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
