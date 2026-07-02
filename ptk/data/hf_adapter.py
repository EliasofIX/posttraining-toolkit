"""Convert ptk Table objects to HuggingFace datasets at the TRL boundary."""

from __future__ import annotations

from ptk.data.table import Table, TableDict


def to_hf_dataset(table: Table):
    """Convert a Table to a HuggingFace Dataset for TRL trainers."""
    from datasets import Dataset

    if not table.column_names:
        return Dataset.from_dict({})
    data = {col: [row.get(col) for row in table] for col in table.column_names}
    return Dataset.from_dict(data)


def to_hf_dataset_dict(table_dict: TableDict):
    """Convert a TableDict to a dict of HuggingFace Datasets."""
    return {name: to_hf_dataset(table) for name, table in table_dict.items()}
