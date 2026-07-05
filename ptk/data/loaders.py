"""Data loading from multiple formats."""

from __future__ import annotations

import csv
import json
import tempfile
import urllib.error
from pathlib import Path

from ptk.config.schema import DatasetConfig, DatasetFormat
from ptk.data.table import Table
from ptk.exceptions import ValidationError
from ptk.hub.client import HubClient

# Tried in order; validation.jsonl is a last-resort fallback when no train file exists.
_HF_HUB_CANDIDATES = ("train.jsonl", "data.jsonl", "train.csv", "validation.jsonl")


def load_raw_dataset(config: DatasetConfig) -> Table:
    """Load a dataset from the configured source."""
    path = config.path
    fmt = config.format

    if fmt == DatasetFormat.HF_HUB:
        return _load_hf_hub(path)

    if not Path(path).exists():
        raise ValidationError(f"Dataset path not found: {path}", field_path="data.dataset.path")

    if fmt == DatasetFormat.JSONL:
        return _load_jsonl(Path(path))
    if fmt == DatasetFormat.CSV:
        return _load_csv(Path(path))
    if fmt == DatasetFormat.PARQUET:
        return _load_parquet(Path(path))

    raise ValidationError(f"Unsupported dataset format: {fmt}", field_path="data.dataset.format")


def _load_jsonl(path: Path) -> Table:
    try:
        from datasets import load_dataset

        ds = load_dataset("json", data_files=str(path), split="train")
        return Table.from_dict(ds.to_dict())
    except ImportError:
        pass

    columns: list[str] = []
    data: dict[str, list] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not columns:
                columns = list(row.keys())
                data = {col: [] for col in columns}
            for col in columns:
                data[col].append(row.get(col))
    return Table(columns, data)


def _load_csv(path: Path) -> Table:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return Table.from_records(list(reader))


def _load_parquet(path: Path) -> Table:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValidationError(
            "Parquet support requires pyarrow: pip install pyarrow",
            field_path="data.dataset.format",
        ) from exc
    table = pq.read_table(path)
    try:
        return Table.from_dict(table.to_pydict())
    except ValueError as exc:
        raise ValidationError(str(exc), field_path="data.dataset.path") from exc


def _load_hf_hub(repo_id: str) -> Table:
    client = HubClient()
    for candidate in _HF_HUB_CANDIDATES:
        try:
            payload = client.download_repo_files(repo_id, filename=candidate, repo_type="dataset")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            if exc.code in (401, 403):
                raise ValidationError(
                    f"Authentication required for Hugging Face repo: {repo_id}",
                    field_path="data.dataset.path",
                ) from exc
            raise ValidationError(
                f"Failed to download {candidate} from Hugging Face repo {repo_id}: HTTP {exc.code}",
                field_path="data.dataset.path",
            ) from exc
        except urllib.error.URLError as exc:
            raise ValidationError(
                f"Failed to download from Hugging Face repo {repo_id}: {exc.reason}",
                field_path="data.dataset.path",
            ) from exc

        suffix = candidate.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        try:
            if suffix == "jsonl":
                return _load_jsonl(tmp_path)
            return _load_csv(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    raise ValidationError(
        f"Could not load dataset from Hugging Face repo: {repo_id}."
        f" Tried: {', '.join(_HF_HUB_CANDIDATES)}.",
        field_path="data.dataset.path",
    )


def count_dataset_samples(config: DatasetConfig) -> int:
    """Fast row count without loading full dataset into memory."""
    if config.format == DatasetFormat.HF_HUB:
        return 0

    path = Path(config.path)
    if not path.exists():
        return 0

    if config.format == DatasetFormat.JSONL:
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    if config.format == DatasetFormat.PARQUET:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return 0
        return pq.read_metadata(path).num_rows

    if config.format == DatasetFormat.CSV:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)

    return 0


def normalize_sft_columns(dataset: Table, config: DatasetConfig) -> Table:
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


def normalize_dpo_columns(dataset: Table, config: DatasetConfig) -> Table:
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
