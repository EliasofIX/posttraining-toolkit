"""PPO trainer via TRL."""

from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOConfig, PPOTrainer

from ptk.distributed.detect import torch_device_string
from ptk.training.base_trainer import BaseTrainer, TrainerResult, prepare_tokenizer


class PPOTrainerWrapper(BaseTrainer):
    """Proximal Policy Optimization with reward model."""

    def train(self, dataset: DatasetDict, *, resume_from: Path | None = None) -> TrainerResult:
        assert self.config.training.rl is not None
        rl = self.config.training.rl
        assert rl.reward_model is not None

        self.logger.start("PPO training", model=self.config.base_model)
        device = torch_device_string(self.env.device)

        tokenizer = prepare_tokenizer(self.config.base_model)
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model, trust_remote_code=True)
        if device != "cpu":
            model.to(device)

        reward_tokenizer = AutoTokenizer.from_pretrained(rl.reward_model)
        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_model = AutoModelForSequenceClassification.from_pretrained(rl.reward_model)
        if device != "cpu":
            reward_model.to(device)

        t = self.config.training
        ppo_config = PPOConfig(
            output_dir=str(self.output_dir),
            learning_rate=t.learning_rate,
            batch_size=t.batch_size,
            mini_batch_size=max(1, t.batch_size // 2),
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            log_with=None,
        )

        def tokenize(example):
            return tokenizer(example["text"], truncation=True, max_length=t.max_seq_length)

        train_ds = dataset["train"].map(tokenize)

        trainer = PPOTrainer(
            args=ppo_config,
            model=model,
            processing_class=tokenizer,
            reward_model=reward_model,
            train_dataset=train_ds,
            value_model=reward_model,
        )

        generation_kwargs = {
            "max_new_tokens": rl.max_completion_length,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": tokenizer.pad_token_id,
        }

        max_steps = t.max_iters or min(len(train_ds), 50)
        for step, batch in enumerate(trainer.dataloader):
            if step >= max_steps:
                break
            query_tensors = [row["input_ids"] for row in batch]
            response_tensors = trainer.generate(query_tensors, **generation_kwargs)
            rewards = [float(r) for r in trainer.compute_rewards(query_tensors, response_tensors)]
            stats = trainer.step(query_tensors, response_tensors, rewards)
            if step % t.logging_steps == 0:
                self.logger.metric("ppo_step", step, **{k: v for k, v in stats.items() if isinstance(v, (int, float))})

        final_path = self.output_dir / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        result = TrainerResult(
            output_dir=self.output_dir,
            checkpoint_path=final_path,
            global_step=max_steps,
        )
        self.save_training_metadata(result)
        self.logger.complete("PPO training complete", steps=max_steps)
        return result
