"""Merge LoRA adapters into base model weights."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptk.config.schema import PTKConfig


def merge_adapter(
    config: PTKConfig,
    adapter_path: Path,
    output_path: Path,
) -> Path:
    """Merge PEFT adapter into base weights and save fp16 checkpoint."""
    output_path.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_path))

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))
    return output_path
