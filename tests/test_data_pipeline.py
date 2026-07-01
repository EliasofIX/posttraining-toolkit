"""Data pipeline tests."""

from datasets import Dataset

from ptk.config.schema import DataConfig, DatasetConfig, DatasetFormat, SyntheticFilters, TrainingMethod
from ptk.data.pipeline import apply_filters, deduplicate, split_dataset


def test_deduplicate():
    ds = Dataset.from_dict({"text": ["hello world", "hello world", "unique text"]})
    result = deduplicate(ds)
    assert len(result) == 2


def test_apply_filters():
    ds = Dataset.from_dict({"text": ["hi", "a" * 5000, "good quality text here"]})
    filters = SyntheticFilters(min_length=5, max_length=100, min_quality_score=0.0)
    result = apply_filters(ds, filters, method=TrainingMethod.SFT)
    assert len(result) >= 1


def test_split_dataset():
    ds = Dataset.from_dict({"text": [f"sample {i}" for i in range(100)]})
    split = split_dataset(ds, train_ratio=0.8, seed=42)
    assert len(split["train"]) == 80
    assert len(split["validation"]) == 20
