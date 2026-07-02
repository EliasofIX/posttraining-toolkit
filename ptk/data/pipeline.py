"""Data preprocessing pipeline: dedup, filtering, splits."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ptk.config.schema import DataConfig, SyntheticFilters, TrainingMethod
from ptk.data.loaders import load_raw_dataset, normalize_dpo_columns, normalize_sft_columns
from ptk.data.table import Table, TableDict


def preprocess_dataset(
    config: DataConfig,
    method: TrainingMethod,
) -> TableDict:
    """Full data pipeline: load → normalize → filter → dedup → split."""
    if config.source.value == "dataset":
        assert config.dataset is not None
        dataset = load_raw_dataset(config.dataset)
        if method in (TrainingMethod.DPO,):
            dataset = normalize_dpo_columns(dataset, config.dataset)
        else:
            dataset = normalize_sft_columns(dataset, config.dataset)
    else:
        raise ValueError("Synthetic data must be generated before preprocessing")

    filters = config.synthetic.filters if config.synthetic else SyntheticFilters()
    dataset = apply_filters(dataset, filters, method=method)
    dataset = deduplicate(dataset, threshold=filters.dedup_threshold)
    return split_dataset(dataset, train_ratio=config.train_split, seed=config.seed)


def apply_filters(
    dataset: Table,
    filters: SyntheticFilters,
    *,
    method: TrainingMethod,
) -> Table:
    """Apply length and quality filters."""

    def score_quality(text: str) -> float:
        if not text or not text.strip():
            return 0.0
        words = text.split()
        if len(words) < 3:
            return 0.2
        unique_ratio = len(set(words)) / len(words)
        return min(1.0, 0.5 + unique_ratio * 0.5)

    def filter_fn(example: dict) -> bool:
        if method == TrainingMethod.DPO:
            texts = [example.get("prompt", ""), example.get("chosen", ""), example.get("rejected", "")]
        else:
            texts = [example.get("text", "")]
        for text in texts:
            length = len(text)
            if length < filters.min_length or length > filters.max_length:
                return False
            if score_quality(text) < filters.min_quality_score:
                return False
        return True

    return dataset.filter(filter_fn)


def deduplicate(dataset: Table, *, threshold: float = 0.95) -> Table:
    """Hash-based exact deduplication (embedding dedup at threshold≈1.0)."""
    seen: set[str] = set()
    keep_indices: list[int] = []

    text_key = "text" if "text" in dataset.column_names else "prompt"

    for i, example in enumerate(dataset):
        text = _normalize_text(str(example.get(text_key, "")))
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            keep_indices.append(i)

    return dataset.select(keep_indices)


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_dataset(
    dataset: Table,
    *,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> TableDict:
    """Train/validation split."""
    split = dataset.train_test_split(test_size=1.0 - train_ratio, seed=seed)
    return TableDict({"train": split["train"], "validation": split["test"]})


def dataset_stats(dataset_dict: TableDict) -> dict[str, Any]:
    """Compute dataset statistics for planning."""
    stats: dict[str, Any] = {}
    for split_name, ds in dataset_dict.items():
        stats[split_name] = {
            "num_rows": len(ds),
            "columns": ds.column_names,
        }
    return stats


def hash_dataset(dataset_dict: TableDict) -> str:
    """Deterministic hash of dataset content for registry."""
    h = hashlib.sha256()
    for split in sorted(dataset_dict.keys()):
        ds = dataset_dict[split]
        for i in range(min(len(ds), 1000)):
            row = ds[i]
            h.update(str(sorted(row.items())).encode())
    return h.hexdigest()[:16]
