"""DPO trainer via TRL."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from ptk.data.hf_adapter import to_hf_dataset_dict
from ptk.data.table import TableDict
from ptk.training.base_trainer import (
    BaseTrainer,
    TrainerResult,
    place_model,
    prepare_tokenizer,
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

        dpo_config = DPOConfig(
            **self.build_dpo_config_kwargs(has_validation="validation" in hf_data),
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=hf_data["train"],
            eval_dataset=hf_data.get("validation"),
            processing_class=tokenizer,
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
        self.logger.complete("DPO training complete", step=result.global_step)
        return result
