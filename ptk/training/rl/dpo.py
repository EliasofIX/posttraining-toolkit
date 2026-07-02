"""DPO trainer via TRL."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from ptk.data.hf_adapter import to_hf_dataset_dict
from ptk.data.table import TableDict
from ptk.distributed.detect import torch_device_string
from ptk.training.base_trainer import BaseTrainer, TrainerResult, prepare_tokenizer


class DPOTrainerWrapper(BaseTrainer):
    """Direct Preference Optimization."""

    def train(self, dataset: TableDict, *, resume_from: Path | None = None) -> TrainerResult:
        hf_data = to_hf_dataset_dict(dataset)
        assert self.config.training.rl is not None
        rl = self.config.training.rl
        self.logger.start("DPO training", model=self.config.base_model, beta=rl.beta)

        tokenizer = prepare_tokenizer(self.config.base_model)
        device = torch_device_string(self.env.device)
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model, trust_remote_code=True)
        if device != "cpu":
            model.to(device)

        ref_model = AutoModelForCausalLM.from_pretrained(self.config.base_model, trust_remote_code=True)
        if device != "cpu":
            ref_model.to(device)

        t = self.config.training
        dpo_config = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=t.epochs,
            per_device_train_batch_size=t.batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            learning_rate=t.learning_rate,
            logging_steps=t.logging_steps,
            save_steps=t.save_steps,
            report_to="none",
            max_steps=t.max_iters if t.max_iters else -1,
            beta=rl.beta,
            max_length=t.max_seq_length,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
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
        )
        self.save_training_metadata(result)
        self.logger.complete("DPO training complete", step=result.global_step)
        return result
