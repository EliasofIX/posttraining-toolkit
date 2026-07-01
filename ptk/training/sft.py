"""Supervised fine-tuning trainer."""

from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict
from transformers import AutoModelForCausalLM
from trl import SFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from ptk.distributed.detect import resolve_mixed_precision, torch_device_string
from ptk.training.base_trainer import BaseTrainer, TrainerResult, prepare_tokenizer


class SFTTrainerWrapper(BaseTrainer):
    """Standard SFT via TRL SFTTrainer."""

    def train(self, dataset: DatasetDict, *, resume_from: Path | None = None) -> TrainerResult:
        self.logger.start("SFT training", model=self.config.base_model)
        tokenizer = prepare_tokenizer(self.config.base_model)
        device = torch_device_string(self.env.device)

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if device != "cpu":
            model.to(device)

        t = self.config.training
        precision = resolve_mixed_precision(self.env.device, self.config.compute.mixed_precision)
        use_cpu = device == "cpu"
        sft_config = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=t.epochs,
            per_device_train_batch_size=t.batch_size,
            per_device_eval_batch_size=t.batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            learning_rate=t.learning_rate,
            warmup_ratio=t.warmup_ratio,
            weight_decay=t.weight_decay,
            logging_steps=t.logging_steps,
            save_steps=t.save_steps,
            eval_strategy="steps",
            eval_steps=t.save_steps,
            save_total_limit=3,
            load_best_model_at_end=False,
            report_to="none",
            max_steps=t.max_iters if t.max_iters else -1,
            max_length=t.max_seq_length,
            dataset_text_field="text",
            use_cpu=use_cpu,
            fp16=precision == "fp16",
            bf16=precision == "bf16",
        )

        trainer = TRLSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            processing_class=tokenizer,
        )

        if resume_from:
            trainer.train(resume_from_checkpoint=str(resume_from))
        else:
            trainer.train()

        trainer.save_model(str(self.output_dir / "final"))
        metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}

        result = TrainerResult(
            output_dir=self.output_dir,
            metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            checkpoint_path=self.output_dir / "final",
            global_step=trainer.state.global_step,
        )
        self.save_training_metadata(result)
        self.logger.complete("SFT training complete", step=result.global_step)
        return result
