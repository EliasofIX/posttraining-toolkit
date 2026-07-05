"""LoRA fine-tuning trainer."""

from __future__ import annotations

from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from ptk.data.hf_adapter import to_hf_dataset_dict
from ptk.data.table import TableDict
from ptk.training.base_trainer import BaseTrainer, TrainerResult, place_model, prepare_tokenizer, resolve_resume_checkpoint


class LoRATrainer(BaseTrainer):
    """LoRA adapter training via PEFT + TRL."""

    def train(self, dataset: TableDict, *, resume_from: Path | None = None) -> TrainerResult:
        hf_data = to_hf_dataset_dict(dataset)
        self.logger.start("LoRA training", model=self.config.base_model)
        assert self.config.training.lora is not None
        lora_cfg = self.config.training.lora
        tokenizer = prepare_tokenizer(self.config.base_model)

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        model = place_model(model, self.env)

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

        sft_config = SFTConfig(**self.build_sft_config_kwargs(has_validation="validation" in hf_data))

        trainer = TRLSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=hf_data["train"],
            eval_dataset=hf_data.get("validation"),
            processing_class=tokenizer,
        )

        checkpoint = resolve_resume_checkpoint(self.output_dir, resume_from, auto_resume=resume_from is not None)
        if checkpoint:
            trainer.train(resume_from_checkpoint=checkpoint)
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
        self.logger.complete("LoRA training complete", step=result.global_step)
        return result
