"""Data generator protocol and registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from datasets import Dataset


@runtime_checkable
class DataGenerator(Protocol):
    """Pluggable synthetic data generation interface."""

    def generate(self, seed_prompts: list[str], n_samples: int, **kwargs) -> Dataset: ...


def get_generator(backend: str, **kwargs) -> DataGenerator:
    """Factory for data generators."""
    if backend == "anthropic":
        from ptk.data.generators.anthropic import AnthropicGenerator

        return AnthropicGenerator(**kwargs)
    if backend == "openai":
        from ptk.data.generators.openai import OpenAIGenerator

        return OpenAIGenerator(**kwargs)
    if backend == "self_instruct":
        from ptk.data.generators.self_instruct import SelfInstructGenerator

        return SelfInstructGenerator(**kwargs)
    raise ValueError(f"Unknown generator backend: {backend}")
