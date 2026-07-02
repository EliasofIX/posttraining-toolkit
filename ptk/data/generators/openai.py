"""OpenAI API-based synthetic data generation."""

from __future__ import annotations

import os
import time
from typing import Any

from ptk.data.table import Table
from ptk.exceptions import RuntimeError as PTKRuntimeError


class OpenAIGenerator:
    """Generate training data via OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        rate_limit_delay: float = 0.5,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.total_tokens = 0

    def generate(self, seed_prompts: list[str], n_samples: int, **kwargs: Any) -> Table:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise PTKRuntimeError(
                "OPENAI_API_KEY environment variable required",
                code="MISSING_API_KEY",
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PTKRuntimeError(
                "openai package required: pip install openai",
                code="MISSING_DEPENDENCY",
            ) from exc

        client = OpenAI(api_key=api_key)
        records: list[dict[str, str]] = []
        seeds = seed_prompts or ["Generate a helpful instruction-response pair for LLM fine-tuning."]

        for i in range(n_samples):
            seed = seeds[i % len(seeds)]
            text = self._generate_one(client, seed)
            records.append({"text": text})
            time.sleep(self.rate_limit_delay)

        return Table.from_records(records)

    def _generate_one(self, client: Any, seed: str) -> str:
        prompt = (
            f"{seed}\n\n"
            "Respond with a single instruction-response training example in the format:\n"
            "Instruction: ...\nResponse: ..."
        )
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )
                text = response.choices[0].message.content or ""
                if response.usage:
                    self.total_tokens += response.usage.total_tokens
                return text
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise PTKRuntimeError(f"OpenAI API failed: {exc}") from exc
                time.sleep(2**attempt)
        return ""
