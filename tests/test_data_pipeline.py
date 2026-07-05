"""Data pipeline tests."""

import json
from pathlib import Path

from ptk.config.schema import SyntheticFilters, TrainingMethod
from ptk.data.pipeline import apply_filters, deduplicate, load_processed_cache, save_processed_cache, split_dataset
from ptk.data.table import Table, TableDict


def test_deduplicate():
    ds = Table.from_dict({"text": ["hello world", "hello world", "unique text"]})
    result = deduplicate(ds)
    assert len(result) == 2


def test_apply_filters():
    ds = Table.from_dict({"text": ["hi", "a" * 5000, "good quality text here"]})
    filters = SyntheticFilters(min_length=5, max_length=100, min_quality_score=0.0)
    result = apply_filters(ds, filters, method=TrainingMethod.SFT)
    assert len(result) >= 1


def test_split_dataset():
    ds = Table.from_dict({"text": [f"sample {i}" for i in range(100)]})
    split = split_dataset(ds, train_ratio=0.8, seed=42)
    assert len(split["train"]) == 80
    assert len(split["validation"]) == 20


def test_split_dataset_single_row():
    ds = Table.from_dict({"text": ["only sample"]})
    split = split_dataset(ds, train_ratio=0.9, seed=42)
    assert len(split["train"]) == 1
    assert len(split["validation"]) == 0


def test_parquet_cache_round_trip(tmp_path):
    dataset_dict = TableDict(
        {
            "train": Table.from_dict({"text": ["alpha", "beta"]}),
            "validation": Table.from_dict({"text": ["gamma"]}),
        }
    )
    cache_dir = tmp_path / "parquet_cache"
    save_processed_cache(cache_dir, dataset_dict, "parquet123", "sft")

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("format") == "parquet"

    loaded = load_processed_cache(cache_dir, expected_hash="parquet123")
    assert loaded is not None
    assert len(loaded["train"]) == 2
    assert loaded["train"][0]["text"] == "alpha"
