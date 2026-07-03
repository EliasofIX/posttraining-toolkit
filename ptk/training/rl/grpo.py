"""GRPO trainer via TRL."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from ptk.data.hf_adapter import to_hf_dataset_dict
from ptk.data.table import TableDict
from ptk.training.base_trainer import (
    BaseTrainer,
    TrainerResult,
    dataloader_kwargs,
    dataset_map_kwargs,
    place_model,
    prepare_tokenizer,
)
from ptk.distributed.detect import resolve_mixed_precision, torch_device_string


def _reward_length(completions: list, **kwargs) -> list[float]:
    """Simple length-based reward for smoke tests."""
    rewards = []
    for completion in completions:
        if isinstance(completion, str):
            text = completion
        elif isinstance(completion, list) and completion:
            item = completion[0]
            text = item.get("content", str(item)) if isinstance(item, dict) else str(item)
        else:
            text = str(completion)
        length = len(text.split())
        rewards.append(min(1.0, length / 50.0))
    return rewards


class GRPOTrainerWrapper(BaseTrainer):
    """Group Relative Policy Optimization."""

    def train(self, dataset: TableDict, *, resume_from: Path | None = None) -> TrainerResult:
        hf_data = to_hf_dataset_dict(dataset)
        assert self.config.training.rl is not None
        rl = self.config.training.rl
        self.logger.start("GRPO training", model=self.config.base_model)

        tokenizer = prepare_tokenizer(self.config.base_model)
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model, trust_remote_code=True)
        model = place_model(model, self.env)

        t = self.config.training
        precision = resolve_mixed_precision(self.env.device, self.config.compute.mixed_precision)
        device = torch_device_string(self.env.device)

        def to_prompt(example):
            text = example.get("text", example.get("prompt", ""))
            return {"prompt": text[: rl.max_prompt_length]}

        train_ds = hf_data["train"].map(to_prompt, **dataset_map_kwargs(t))

        grpo_config = GRPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=t.epochs,
            per_device_train_batch_size=t.batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            learning_rate=t.learning_rate,
            logging_steps=t.logging_steps,
            save_steps=t.save_steps,
            report_to="none",
            max_steps=t.max_iters if t.max_iters else -1,
            num_generations=rl.num_generations,
            max_completion_length=rl.max_completion_length,
            beta=rl.beta,
            use_cpu=device == "cpu",
            fp16=precision == "fp16",
            bf16=precision == "bf16",
            gradient_checkpointing=t.gradient_checkpointing,
            **dataloader_kwargs(t),
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_ds,
            processing_class=tokenizer,
            reward_funcs=_reward_length,
        )

        if resume_from:
            trainer.train(resume_from_checkpoint=str(resume_from))
        else:
            trainer.train()

        final_path = self.output_dir / "final"
        trainer.save_model(str(final_path))

        result = TrainerResult(
            output_dir=self.output_dir,
            checkpoint_path=final_path,
            global_step=trainer.state.global_step,
            model=trainer.model,
            tokenizer=tokenizer,
        )
        self.save_training_metadata(result)
        self.logger.complete("GRPO training complete", step=result.global_step)
        return result
