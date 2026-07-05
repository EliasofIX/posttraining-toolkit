"""Self-instruct style local model data generation."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptk.config.schema import DeviceType
from ptk.data.table import Table
from ptk.distributed.detect import detect_device, torch_device_string


class SelfInstructGenerator:
    """Bootstrap synthetic data using a local causal LM (air-gapped friendly)."""

    def __init__(
        self,
        model: str = "distilgpt2",
        max_new_tokens: int = 128,
        device: DeviceType = DeviceType.AUTO,
        batch_size: int = 4,
    ) -> None:
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.device = detect_device(device)
        self.batch_size = max(1, batch_size)

    def generate(self, seed_prompts: list[str], n_samples: int, **kwargs: Any) -> Table:
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

        for start in range(0, n_samples, self.batch_size):
            batch_count = min(self.batch_size, n_samples - start)
            batch_seeds = [seeds[(start + i) % len(seeds)] for i in range(batch_count)]
            instructions = self._generate_batch(model, tokenizer, device_str, batch_seeds)
            critique_prompts = [
                f"Answer this instruction helpfully:\n{instruction}\n\nResponse:"
                for instruction in instructions
            ]
            responses = self._generate_batch(model, tokenizer, device_str, critique_prompts)
            for instruction, response in zip(instructions, responses, strict=True):
                text = f"Instruction: {instruction.strip()}\nResponse: {response.strip()}"
                if self._passes_filter(text):
                    records.append({"text": text})

        if not records:
            records.append({"text": "Instruction: What is fine-tuning?\nResponse: Fine-tuning adapts a pretrained model to a specific task."})

        return Table.from_records(records)

    def _generate_batch(self, model, tokenizer, device: str, prompts: list[str]) -> list[str]:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_lengths = inputs["attention_mask"].sum(dim=1)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts: list[str] = []
        for row, prompt_len in zip(outputs, prompt_lengths, strict=True):
            generated = row[int(prompt_len) :]
            texts.append(tokenizer.decode(generated, skip_special_tokens=True))
        return texts

    def _passes_filter(self, text: str) -> bool:
        return 20 <= len(text) <= 4096
