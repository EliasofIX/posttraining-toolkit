"""Shared trainer interface and utilities."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase, TrainingArguments

from ptk.config.schema import PTKConfig
from ptk.distributed.detect import ComputeEnvironment, resolve_mixed_precision
from ptk.logging import Logger


@dataclass
class TrainerResult:
    """Standardized training result."""

    output_dir: Path
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    global_step: int = 0


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
    def train(self, dataset: DatasetDict, *, resume_from: Path | None = None) -> TrainerResult:
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
        }
        args.update(overrides)
        return TrainingArguments(**args)

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
