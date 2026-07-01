"""QLoRA fine-tuning with CUDA bitsandbytes and MPS fallback."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer as TRLSFTTrainer
from trl import SFTConfig

from ptk.config.schema import DeviceType, QuantBackend
from ptk.distributed.detect import resolve_mixed_precision, torch_device_string
from ptk.training.base_trainer import BaseTrainer, TrainerResult, prepare_tokenizer


def resolve_quantization_backend(device: DeviceType, requested: QuantBackend) -> str:
    """Select quantization backend based on device."""
    if device == DeviceType.CUDA:
        return "bnb"
    if device == DeviceType.MPS:
        if requested in (QuantBackend.MLX_QUANT, QuantBackend.AUTO):
            return "mlx_quant"
        return "mlx_quant"
    return "none"


class QLoRATrainer(BaseTrainer):
    """QLoRA with bitsandbytes on CUDA and MPS-compatible fallback."""

    def train(self, dataset: DatasetDict, *, resume_from: Path | None = None) -> TrainerResult:
        assert self.config.training.lora is not None
        assert self.config.training.quantization is not None
        quant = self.config.training.quantization
        lora_cfg = self.config.training.lora

        backend = resolve_quantization_backend(self.env.device, quant.backend)
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
        elif backend == "mlx_quant":
            warnings.warn(
                "MPS detected: bitsandbytes has no MPS kernels. "
                "Using reduced-precision LoRA (fp16) without bitsandbytes quantization. "
                "Set training.quantization.backend=mlx_quant explicitly to suppress this warning.",
                stacklevel=2,
            )
            model_kwargs["torch_dtype"] = torch.float16
        else:
            warnings.warn(
                "No compatible quantization backend; falling back to full-precision LoRA.",
                stacklevel=2,
            )

        model = AutoModelForCausalLM.from_pretrained(self.config.base_model, **model_kwargs)
        if backend == "bnb":
            model = prepare_model_for_kbit_training(model)
        elif device != "cpu":
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
        self.logger.complete("QLoRA training complete", step=result.global_step)
        return result
