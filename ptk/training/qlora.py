"""QLoRA fine-tuning: bitsandbytes on CUDA, true MLX 4-bit on Apple Silicon."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from ptk.config.schema import DeviceType
from ptk.data.hf_adapter import to_hf_dataset_dict
from ptk.data.table import TableDict
from ptk.distributed.detect import torch_device_string
from ptk.training.base_trainer import (
    BaseTrainer,
    TrainerResult,
    is_distributed,
    prepare_tokenizer,
    resolve_resume_checkpoint,
)
from ptk.training.quant_backend import resolve_quantization_backend


class QLoRATrainer(BaseTrainer):
    """QLoRA with bitsandbytes on CUDA and true MLX quantization on Apple Silicon."""

    def train(self, dataset: TableDict, *, resume_from: Path | None = None) -> TrainerResult:
        assert self.config.training.lora is not None
        assert self.config.training.quantization is not None
        quant = self.config.training.quantization

        backend = resolve_quantization_backend(self.env.device, quant)

        if backend == "mlx_quant":
            from ptk.mlx.trainer import train_qlora_mlx

            result = train_qlora_mlx(
                self.config,
                dataset,
                output_dir=self.output_dir,
                logger=self.logger,
                resume_from=resume_from,
            )
            self.save_training_metadata(result)
            return result

        return self._train_torch(dataset, backend=backend, resume_from=resume_from)

    def _train_torch(
        self,
        dataset: TableDict,
        *,
        backend: str,
        resume_from: Path | None,
    ) -> TrainerResult:
        hf_data = to_hf_dataset_dict(dataset)
        quant = self.config.training.quantization
        lora_cfg = self.config.training.lora
        assert quant is not None and lora_cfg is not None

        self.logger.start("QLoRA training", model=self.config.base_model, quant_backend=backend)

        tokenizer = prepare_tokenizer(self.config.base_model)
        device = torch_device_string(self.env.device)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}

        if backend == "bnb":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant.bits.value == 4,
                load_in_8bit=quant.bits.value == 8,
                bnb_4bit_quant_type=quant.quant_type.value,
                bnb_4bit_use_double_quant=quant.double_quant,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            # none — unquantized LoRA (CPU smoke or explicit allow_unquantized_fallback)
            msg = (
                "QLoRA running without quantization (backend=none). "
                "This is full-precision/fp LoRA, not true QLoRA."
            )
            if self.env.device == DeviceType.MPS and quant.allow_unquantized_fallback:
                msg = (
                    "MPS QLoRA allow_unquantized_fallback=true: using fp16 PEFT LoRA "
                    "without MLX 4-bit quantization."
                )
            warnings.warn(msg, stacklevel=2)
            if self.env.device == DeviceType.MPS:
                model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(self.config.base_model, **model_kwargs)
        if backend == "bnb":
            model = prepare_model_for_kbit_training(model)
        elif not is_distributed(self.env) and device != "cpu":
            model.to(device)

        target_modules = lora_cfg.target_modules or (
            ["c_attn", "c_proj"] if "gpt" in self.config.base_model.lower() else ["q_proj", "v_proj"]
        )
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
        self.logger.complete("QLoRA training complete", step=result.global_step, quant_backend=backend)
        return result
