"""Convert ptk Table objects to HuggingFace datasets at the TRL boundary."""

from __future__ import annotations

from ptk.data.table import Table, TableDict


def to_hf_dataset(table: Table):
    """Convert a Table to a HuggingFace Dataset for TRL trainers."""
    from datasets import Dataset

    if hasattr(table, "_hf_dataset"):
        return table._hf_dataset  # type: ignore[attr-defined]

    data = table.to_dict()
    if not data:
        return Dataset.from_dict({})
    return Dataset.from_dict(data)


def to_hf_dataset_dict(table_dict: TableDict):
    """Convert a TableDict to a dict of HuggingFace Datasets."""
    return {name: to_hf_dataset(table) for name, table in table_dict.items()}


def is_hf_dataset(obj) -> bool:
    """Return True if obj is a HuggingFace Dataset."""
    try:
        from datasets import Dataset

        return isinstance(obj, Dataset)
    except ImportError:
        return False
