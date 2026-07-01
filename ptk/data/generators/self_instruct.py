"""Self-instruct style local model data generation."""

from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptk.config.schema import DeviceType
from ptk.distributed.detect import detect_device, torch_device_string


class SelfInstructGenerator:
    """Bootstrap synthetic data using a local causal LM (air-gapped friendly)."""

    def __init__(
        self,
        model: str = "distilgpt2",
        max_new_tokens: int = 128,
        device: DeviceType = DeviceType.AUTO,
    ) -> None:
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.device = detect_device(device)

    def generate(self, seed_prompts: list[str], n_samples: int, **kwargs: Any) -> Dataset:
        device_str = torch_device_string(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.to(device_str)
        model.eval()

        seeds = seed_prompts or [
            "Write a question about machine learning.",
            "Write a question about Python programming.",
            "Write a question about cooking.",
        ]
        records: list[dict[str, str]] = []

        for i in range(n_samples):
            seed = seeds[i % len(seeds)]
            instruction = self._generate_text(model, tokenizer, device_str, seed)
            critique_prompt = f"Answer this instruction helpfully:\n{instruction}\n\nResponse:"
            response = self._generate_text(model, tokenizer, device_str, critique_prompt)
            text = f"Instruction: {instruction.strip()}\nResponse: {response.strip()}"
            if self._passes_filter(text):
                records.append({"text": text})

        if not records:
            records.append({"text": "Instruction: What is fine-tuning?\nResponse: Fine-tuning adapts a pretrained model to a specific task."})

        return Dataset.from_list(records)

    def _generate_text(self, model, tokenizer, device: str, prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def _passes_filter(self, text: str) -> bool:
        return 20 <= len(text) <= 4096
