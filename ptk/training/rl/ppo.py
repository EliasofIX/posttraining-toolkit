"""PPO trainer via TRL experimental API (TRL >= 1.0)."""

from __future__ import annotations

import os
from pathlib import Path

import torch.nn as nn
from transformers import AutoModelForCausalLM

from ptk.data.hf_adapter import to_hf_dataset_dict
from ptk.data.table import TableDict
from ptk.distributed.detect import resolve_mixed_precision, torch_device_string
from ptk.training.base_trainer import (
    BaseTrainer,
    TrainerResult,
    dataloader_kwargs,
    dataset_map_kwargs,
    deepspeed_kwargs,
    is_distributed,
    prepare_tokenizer,
    resolve_resume_checkpoint,
)

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from trl.experimental.ppo import (  # noqa: E402
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)


class _PPORewardModel(nn.Module):
    """GPT-2 compatible reward model sharing the policy tokenizer vocabulary."""

    base_model_prefix = "transformer"

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.transformer = backbone.get_decoder() if hasattr(backbone, "get_decoder") else backbone.transformer
        hidden = (
            backbone.config.hidden_size
            if hasattr(backbone.config, "hidden_size")
            else backbone.config.n_embd
        )
        self.score = nn.Linear(hidden, 1)

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)


class _PPOValueModel(nn.Module):
    """Adapter exposing TRL PPO's expected ``base_model_prefix`` and ``score`` API."""

    base_model_prefix = "pretrained_model"

    def __init__(self, wrapped: AutoModelForCausalLMWithValueHead) -> None:
        super().__init__()
        self.pretrained_model = wrapped.pretrained_model
        self._v_head = wrapped.v_head

    @property
    def score(self) -> nn.Module:
        return self._v_head


class PPOTrainerWrapper(BaseTrainer):
    """Proximal Policy Optimization with reward and value models."""

    def train(self, dataset: TableDict, *, resume_from: Path | None = None) -> TrainerResult:
        hf_data = to_hf_dataset_dict(dataset)
        assert self.config.training.rl is not None
        rl = self.config.training.rl
        assert rl.reward_model is not None

        self.logger.start("PPO training", model=self.config.base_model)
        device = torch_device_string(self.env.device)
        t = self.config.training
        precision = resolve_mixed_precision(self.env.device, self.config.compute.mixed_precision)
        use_cpu = device == "cpu"
        distributed = is_distributed(self.env)

        tokenizer = prepare_tokenizer(self.config.base_model)

        reward_source = rl.reward_model
        if reward_source != self.config.base_model:
            self.logger.warn(
                "PPO reward model must share the policy tokenizer; using base_model for reward scoring",
                configured=reward_source,
                using=self.config.base_model,
            )
            reward_source = self.config.base_model

        # Separate backbones so policy / reward / value do not share parameters.
        # Note: three full model loads — expect higher RAM than SFT/LoRA smoke runs.
        self.logger.warn(
            "PPO loads three model copies (policy, reward, value); watch memory on small hosts",
            base_model=self.config.base_model,
        )
        policy = AutoModelForCausalLM.from_pretrained(self.config.base_model, trust_remote_code=True)
        reward_backbone = AutoModelForCausalLM.from_pretrained(reward_source, trust_remote_code=True)
        value_backbone = AutoModelForCausalLM.from_pretrained(self.config.base_model, trust_remote_code=True)
        reward_model = _PPORewardModel(reward_backbone)
        self.logger.warn(
            "PPO reward score head is randomly initialized; use a trained reward model for real RL",
            reward_source=reward_source,
        )
        value_wrapped = AutoModelForCausalLMWithValueHead(value_backbone)
        value_model = _PPOValueModel(value_wrapped)

        if not distributed and device != "cpu":
            policy.to(device)
            reward_model.to(device)
            value_model.pretrained_model.to(device)
            value_model._v_head.to(device)

        def tokenize(example):
            return tokenizer(example["text"], truncation=True, max_length=t.max_seq_length)

        # TRL PPOTrainer requires input_ids; always pretokenizeize.
        train_ds = hf_data["train"].map(
            tokenize,
            remove_columns=hf_data["train"].column_names,
            **dataset_map_kwargs(t),
        )

        max_steps = t.max_iters or 2
        batch_size = max(1, t.batch_size * t.gradient_accumulation_steps)

        ppo_config = PPOConfig(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=t.batch_size,
            learning_rate=t.learning_rate,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            total_episodes=max_steps * batch_size,
            response_length=rl.max_completion_length,
            kl_coef=rl.kl_coef,
            num_ppo_epochs=1,
            local_rollout_forward_batch_size=max(1, t.batch_size),
            num_sample_generations=0,
            report_to="none",
            use_cpu=use_cpu,
            fp16=precision == "fp16",
            bf16=precision == "bf16",
            gradient_checkpointing=t.gradient_checkpointing,
            **dataloader_kwargs(t),
            **deepspeed_kwargs(self.config),
        )

        trainer = PPOTrainer(
            args=ppo_config,
            processing_class=tokenizer,
            model=policy,
            ref_model=None,
            reward_model=reward_model,
            train_dataset=train_ds,
            value_model=value_model,
        )

        checkpoint = resolve_resume_checkpoint(self.output_dir, resume_from, auto_resume=resume_from is not None)
        if checkpoint:
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            trainer.train()

        final_path = self.output_dir / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        global_step = trainer.state.global_step if hasattr(trainer, "state") else max_steps
        result = TrainerResult(
            output_dir=self.output_dir,
            checkpoint_path=final_path,
            global_step=global_step,
            model=policy,
            tokenizer=tokenizer,
        )
        self.save_training_metadata(result)
        self.logger.complete("PPO training complete", steps=global_step)
        return result
