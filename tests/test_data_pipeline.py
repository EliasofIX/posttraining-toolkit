"""Data pipeline tests."""

from ptk.config.schema import SyntheticFilters, TrainingMethod
from ptk.data.pipeline import apply_filters, deduplicate, split_dataset
from ptk.data.table import Table


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
