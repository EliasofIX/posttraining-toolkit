"""Synthetic data generation orchestration."""

from __future__ import annotations

from pathlib import Path

from ptk.config.schema import PTKConfig
from ptk.data.generators.base import get_generator
from ptk.data.table import Table
from ptk.logging import Logger


def load_seed_prompts(path: str | None) -> list[str]:
    """Load seed prompts from a text file (one per line)."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    return [line.strip() for line in lines if line.strip()]


def generate_synthetic_data(config: PTKConfig, logger: Logger) -> Table:
    """Run synthetic data generation for a config."""
    assert config.data.synthetic is not None
    syn = config.data.synthetic
    seeds = load_seed_prompts(syn.seed_prompts)
    model = syn.model or config.base_model

    logger.start(
        "Generating synthetic data",
        backend=syn.backend.value,
        n_samples=syn.n_samples,
    )

    generator = get_generator(
        syn.backend.value,
        model=model,
    )
    dataset = generator.generate(seeds, syn.n_samples)
    logger.complete("Synthetic data generation complete", num_rows=len(dataset))
    return dataset
