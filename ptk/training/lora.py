"""LoRA fine-tuning trainer."""

from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from ptk.distributed.detect import resolve_mixed_precision, torch_device_string
from ptk.training.base_trainer import BaseTrainer, TrainerResult, prepare_tokenizer


class LoRATrainer(BaseTrainer):
    """LoRA adapter training via PEFT + TRL."""

    def train(self, dataset: DatasetDict, *, resume_from: Path | None = None) -> TrainerResult:
        self.logger.start("LoRA training", model=self.config.base_model)
        assert self.config.training.lora is not None
        lora_cfg = self.config.training.lora
        tokenizer = prepare_tokenizer(self.config.base_model)
        device = torch_device_string(self.env.device)

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if device != "cpu":
            model.to(device)

        target_modules = lora_cfg.target_modules
        if not target_modules:
            target_modules = ["c_attn", "c_proj"] if "gpt" in self.config.base_model.lower() else ["q_proj", "v_proj"]

        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=target_modules,
            bias=lora_cfg.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)

        t = self.config.training
        precision = resolve_mixed_precision(self.env.device, self.config.compute.mixed_precision)
        use_cpu = device == "cpu"
        sft_config = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=t.epochs,
            per_device_train_batch_size=t.batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            learning_rate=t.learning_rate,
            logging_steps=t.logging_steps,
            save_steps=t.save_steps,
            eval_strategy="steps" if "validation" in dataset else "no",
            eval_steps=t.save_steps,
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

        final_path = self.output_dir / "final"
        trainer.save_model(str(final_path))
        model.save_pretrained(str(final_path))

        result = TrainerResult(
            output_dir=self.output_dir,
            checkpoint_path=final_path,
            global_step=trainer.state.global_step,
        )
        self.save_training_metadata(result)
        self.logger.complete("LoRA training complete", step=result.global_step)
        return result
