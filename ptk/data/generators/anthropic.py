"""Anthropic API-based synthetic data generation."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import Any

from ptk.data.table import Table
from ptk.exceptions import RuntimeError as PTKRuntimeError


class AnthropicGenerator:
    """Generate training data via Anthropic Messages API."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        max_retries: int = 3,
        rate_limit_delay: float = 1.0,
        max_concurrency: int = 4,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.max_concurrency = max(1, max_concurrency)
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def generate(self, seed_prompts: list[str], n_samples: int, **kwargs: Any) -> Table:
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
        seeds = seed_prompts or ["Generate a helpful instruction-response pair for LLM fine-tuning."]
        semaphore = Semaphore(self.max_concurrency)
        records: list[dict[str, str]] = [{}] * n_samples

        def worker(index: int) -> tuple[int, str]:
            seed = seeds[index % len(seeds)]
            with semaphore:
                text = self._generate_one(client, seed)
                if self.rate_limit_delay:
                    time.sleep(self.rate_limit_delay)
            return index, text

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = [executor.submit(worker, i) for i in range(n_samples)]
            for future in as_completed(futures):
                index, text = future.result()
                records[index] = {"text": text}

        return Table.from_records(records)

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
