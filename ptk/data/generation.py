"""Synthetic data generation orchestration."""

from __future__ import annotations

import json
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


def _raw_cache_path(config: PTKConfig) -> Path:
    assert config.data.synthetic is not None
    syn = config.data.synthetic
    cache_dir = config.output_path() / "data_cache" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{syn.backend.value}_{syn.n_samples}_{syn.model or config.base_model}"
    return cache_dir / f"{key}.jsonl"


def _load_raw_cache(path: Path) -> Table | None:
    if not path.exists():
        return None
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return Table.from_records(records) if records else None


def _save_raw_cache(path: Path, dataset: Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(row, default=str) + "\n")


def generate_synthetic_data(config: PTKConfig, logger: Logger) -> Table:
    """Run synthetic data generation for a config."""
    assert config.data.synthetic is not None
    syn = config.data.synthetic
    cache_path = _raw_cache_path(config)
    cached = _load_raw_cache(cache_path)
    if cached is not None:
        logger.complete("Loaded cached synthetic data", num_rows=len(cached), path=str(cache_path))
        return cached

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
    _save_raw_cache(cache_path, dataset)
    logger.complete("Synthetic data generation complete", num_rows=len(dataset), path=str(cache_path))
    return dataset
