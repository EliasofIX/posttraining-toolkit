"""Supervised fine-tuning trainer."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForCausalLM
from trl import SFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from ptk.data.hf_adapter import to_hf_dataset_dict
from ptk.data.table import TableDict
from ptk.training.base_trainer import BaseTrainer, TrainerResult, place_model, prepare_tokenizer


class SFTTrainerWrapper(BaseTrainer):
    """Standard SFT via TRL SFTTrainer."""

    def train(self, dataset: TableDict, *, resume_from: Path | None = None) -> TrainerResult:
        hf_data = to_hf_dataset_dict(dataset)
        self.logger.start("SFT training", model=self.config.base_model)
        tokenizer = prepare_tokenizer(self.config.base_model)

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        model = place_model(model, self.env)

        sft_config = SFTConfig(**self.build_sft_config_kwargs(has_validation="validation" in hf_data))

        trainer = TRLSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=hf_data["train"],
            eval_dataset=hf_data.get("validation"),
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
            model=trainer.model,
            tokenizer=tokenizer,
        )
        self.save_training_metadata(result)
        self.logger.complete("SFT training complete", step=result.global_step)
        return result
