"""Anthropic API-based synthetic data generation."""

from __future__ import annotations

import os
import time
from typing import Any

from datasets import Dataset

from ptk.exceptions import RuntimeError as PTKRuntimeError


class AnthropicGenerator:
    """Generate training data via Anthropic Messages API."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        max_retries: int = 3,
        rate_limit_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def generate(self, seed_prompts: list[str], n_samples: int, **kwargs: Any) -> Dataset:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise PTKRuntimeError(
                "ANTHROPIC_API_KEY environment variable required",
                code="MISSING_API_KEY",
            )

        try:
            import anthropic
        except ImportError as exc:
            raise PTKRuntimeError(
                "anthropic package required: pip install anthropic",
                code="MISSING_DEPENDENCY",
            ) from exc

        client = anthropic.Anthropic(api_key=api_key)
        records: list[dict[str, str]] = []
        seeds = seed_prompts or ["Generate a helpful instruction-response pair for LLM fine-tuning."]

        for i in range(n_samples):
            seed = seeds[i % len(seeds)]
            text = self._generate_one(client, seed)
            records.append({"text": text})
            time.sleep(self.rate_limit_delay)

        return Dataset.from_list(records)

    def _generate_one(self, client: Any, seed: str) -> str:
        prompt = (
            f"{seed}\n\n"
            "Respond with a single instruction-response training example in the format:\n"
            "Instruction: ...\nResponse: ..."
        )
        for attempt in range(self.max_retries):
            try:
                message = client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = message.content[0].text
                self.total_tokens += message.usage.input_tokens + message.usage.output_tokens
                return text
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise PTKRuntimeError(f"Anthropic API failed: {exc}") from exc
                time.sleep(2**attempt)
        return ""
