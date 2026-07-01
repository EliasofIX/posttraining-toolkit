"""Data loading from multiple formats."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset

from ptk.config.schema import DatasetConfig, DatasetFormat
from ptk.exceptions import ValidationError


def load_raw_dataset(config: DatasetConfig) -> Dataset:
    """Load a dataset from the configured source."""
    path = config.path
    fmt = config.format

    if fmt == DatasetFormat.HF_HUB:
        return load_dataset(path, split="train")

    if not Path(path).exists():
        raise ValidationError(f"Dataset path not found: {path}", field_path="data.dataset.path")

    if fmt == DatasetFormat.JSONL:
        return load_dataset("json", data_files=path, split="train")
    if fmt == DatasetFormat.CSV:
        return load_dataset("csv", data_files=path, split="train")
    if fmt == DatasetFormat.PARQUET:
        return load_dataset("parquet", data_files=path, split="train")

    raise ValidationError(f"Unsupported dataset format: {fmt}", field_path="data.dataset.format")


def normalize_sft_columns(dataset: Dataset, config: DatasetConfig) -> Dataset:
    """Normalize dataset to text column for SFT."""
    col = config.text_column
    if col in dataset.column_names:
        return dataset

    if config.prompt_column and config.response_column:
        prompt_col = config.prompt_column
        response_col = config.response_column

        def merge_prompt_response(example: dict) -> dict:
            return {"text": f"{example[prompt_col]}\n{example[response_col]}"}

        return dataset.map(merge_prompt_response, remove_columns=dataset.column_names)

    raise ValidationError(
        f"Column '{col}' not found and no prompt/response columns configured",
        field_path="data.dataset.text_column",
    )


def normalize_dpo_columns(dataset: Dataset, config: DatasetConfig) -> Dataset:
    """Ensure DPO-required columns exist."""
    required = {
        "prompt": config.prompt_column or "prompt",
        "chosen": config.chosen_column or "chosen",
        "rejected": config.rejected_column or "rejected",
    }
    missing = [k for k, v in required.items() if v not in dataset.column_names]
    if missing:
        raise ValidationError(
            f"Missing DPO columns: {missing}",
            field_path="data.dataset",
            details={"required": required, "available": dataset.column_names},
        )
    rename = {v: k for k, v in required.items() if v != k}
    if rename:
        dataset = dataset.rename_columns(rename)
    return dataset
