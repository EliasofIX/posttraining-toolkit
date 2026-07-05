"""DPO trainer via TRL."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from ptk.data.hf_adapter import to_hf_dataset_dict
from ptk.data.table import TableDict
from ptk.distributed.detect import resolve_mixed_precision
from ptk.training.base_trainer import (
    BaseTrainer,
    TrainerResult,
    dataloader_kwargs,
    deepspeed_kwargs,
    place_model,
    prepare_tokenizer,
    resolve_resume_checkpoint,
)


class DPOTrainerWrapper(BaseTrainer):
    """Direct Preference Optimization."""

    def train(self, dataset: TableDict, *, resume_from: Path | None = None) -> TrainerResult:
        hf_data = to_hf_dataset_dict(dataset)
        assert self.config.training.rl is not None
        rl = self.config.training.rl
        self.logger.start("DPO training", model=self.config.base_model, beta=rl.beta)

        tokenizer = prepare_tokenizer(self.config.base_model)
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model, trust_remote_code=True)
        model = place_model(model, self.env)

        ref_model = self._load_ref_model()

        t = self.config.training
        precision = resolve_mixed_precision(self.env.device, self.config.compute.mixed_precision)
        dpo_config = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=t.epochs,
            per_device_train_batch_size=t.batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            learning_rate=t.learning_rate,
            logging_steps=t.logging_steps,
            save_steps=t.save_steps,
            save_strategy=t.save_strategy,
            report_to="none",
            max_steps=t.max_iters if t.max_iters else -1,
            beta=rl.beta,
            max_length=t.max_seq_length,
            gradient_checkpointing=t.gradient_checkpointing,
            fp16=precision == "fp16",
            bf16=precision == "bf16",
            precompute_ref_log_probs=rl.precompute_ref_log_probs,
            **dataloader_kwargs(t),
            **deepspeed_kwargs(self.config),
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
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
        self.logger.complete("DPO training complete", step=result.global_step)
        return result

    def _load_ref_model(self):
        """Load reference model with optional 4-bit quantization."""
        assert self.config.training.rl is not None
        rl = self.config.training.rl
        if rl.ref_model_quantization != "4bit":
            return None

        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
            )
            ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
            return place_model(ref_model, self.env)
        except Exception as exc:
            self.logger.warn("4-bit ref model unavailable; using TRL implicit reference", error=str(exc))
            return None
