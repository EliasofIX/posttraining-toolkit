"""Data preprocessing pipeline: dedup, filtering, splits."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ptk.config.schema import DataConfig, DatasetConfig, SyntheticFilters, TrainingMethod
from ptk.data.loaders import load_raw_dataset, normalize_dpo_columns, normalize_sft_columns
from ptk.data.table import Table, TableDict


def preprocess_dataset(
    config: DataConfig,
    method: TrainingMethod,
    *,
    num_proc: int | None = None,
) -> TableDict:
    """Full data pipeline: load → normalize → filter → dedup → split."""
    if config.source.value == "dataset":
        assert config.dataset is not None
        if config.dataset.streaming:
            return _preprocess_streaming(config, method, num_proc=num_proc)
        dataset = load_raw_dataset(config.dataset)
        if method in (TrainingMethod.DPO,):
            dataset = normalize_dpo_columns(dataset, config.dataset)
        else:
            dataset = normalize_sft_columns(dataset, config.dataset)
    else:
        raise ValueError("Synthetic data must be generated before preprocessing")

    filters = config.synthetic.filters if config.synthetic else SyntheticFilters()
    return _preprocess_table(
        dataset,
        filters,
        method=method,
        num_proc=num_proc,
        train_ratio=config.train_split,
        seed=config.seed,
    )


def _preprocess_streaming(
    config: DataConfig,
    method: TrainingMethod,
    *,
    num_proc: int | None = None,
) -> TableDict:
    """Streaming preprocess path — loads without full in-memory materialization."""
    assert config.dataset is not None
    from datasets import load_dataset

    path = config.dataset.path
    fmt = config.dataset.format.value
    if fmt == "jsonl":
        hf_ds = load_dataset("json", data_files=path, split="train", streaming=True)
    elif fmt == "csv":
        hf_ds = load_dataset("csv", data_files=path, split="train", streaming=True)
    else:
        dataset = load_raw_dataset(config.dataset)
        filters = config.synthetic.filters if config.synthetic else SyntheticFilters()
        return _preprocess_table(dataset, filters, method=method, num_proc=num_proc)

    rows: list[dict[str, Any]] = []
    for row in hf_ds:
        rows.append(dict(row))
        if len(rows) >= 100_000:
            break

    table = Table.from_records(rows)
    if method == TrainingMethod.DPO:
        table = normalize_dpo_columns(table, config.dataset)
    else:
        table = normalize_sft_columns(table, config.dataset)

    filters = config.synthetic.filters if config.synthetic else SyntheticFilters()
    return _preprocess_table(
        table,
        filters,
        method=method,
        num_proc=num_proc,
        train_ratio=config.train_split,
        seed=config.seed,
    )


def _preprocess_table(
    dataset: Table,
    filters: SyntheticFilters,
    *,
    method: TrainingMethod,
    num_proc: int | None = None,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> TableDict:
    """Run filter/dedup/split using HF Dataset.map when beneficial."""
    if len(dataset) == 0:
        return TableDict({"train": dataset, "validation": Table([], {})})

    try:
        from datasets import Dataset

        hf_ds = Dataset.from_dict(dataset.to_dict())
        hf_ds = _apply_filters_hf(hf_ds, filters, method=method, num_proc=num_proc)
        hf_ds = _deduplicate_hf(hf_ds, threshold=filters.dedup_threshold, num_proc=num_proc)
        if len(hf_ds) <= 1:
            train_table = Table.from_dict(hf_ds.to_dict()) if len(hf_ds) else Table([], {})
            empty_cols = train_table.column_names or ["text"]
            validation = Table(empty_cols, {col: [] for col in empty_cols})
            return TableDict({"train": train_table, "validation": validation})
        split = hf_ds.train_test_split(test_size=1.0 - train_ratio, seed=seed)
        return TableDict(
            {
                "train": Table.from_dict(split["train"].to_dict()),
                "validation": Table.from_dict(split["test"].to_dict()),
            }
        )
    except ImportError:
        dataset = apply_filters(dataset, filters, method=method)
        dataset = deduplicate(dataset, threshold=filters.dedup_threshold)
        return split_dataset(dataset, train_ratio=train_ratio, seed=seed)


def _apply_filters_hf(ds, filters: SyntheticFilters, *, method: TrainingMethod, num_proc: int | None):
    """Batched quality/length filtering via HuggingFace datasets."""

    def filter_batch(batch: dict[str, list]) -> list[bool]:
        keep: list[bool] = []
        n = len(next(iter(batch.values())))
        for i in range(n):
            if method == TrainingMethod.DPO:
                texts = [
                    str(batch.get("prompt", [""] * n)[i]),
                    str(batch.get("chosen", [""] * n)[i]),
                    str(batch.get("rejected", [""] * n)[i]),
                ]
            else:
                texts = [str(batch.get("text", [""] * n)[i])]
            keep.append(_passes_filters(texts, filters))
        return keep

    map_kwargs: dict[str, Any] = {"batched": True}
    if num_proc:
        map_kwargs["num_proc"] = num_proc
    return ds.filter(lambda batch: filter_batch(batch), **map_kwargs)


def _deduplicate_hf(ds, *, threshold: float, num_proc: int | None):
    """Hash-based deduplication using batched map."""
    _ = threshold  # reserved for future approximate dedup
    if len(ds) == 0:
        return ds

    text_key = "text" if "text" in ds.column_names else "prompt"

    def add_hash(example: dict) -> dict:
        text = _normalize_text(str(example.get(text_key, "")))
        return {"__dedup_hash": hashlib.sha256(text.encode()).hexdigest()}

    map_kwargs: dict[str, Any] = {}
    if num_proc:
        map_kwargs["num_proc"] = num_proc
    ds = ds.map(add_hash, **map_kwargs)
    _, unique_indices = _unique_indices(ds["__dedup_hash"])
    ds = ds.select(unique_indices)
    return ds.remove_columns(["__dedup_hash"])


def _unique_indices(hashes: list[str]) -> tuple[set[str], list[int]]:
    seen: set[str] = set()
    indices: list[int] = []
    for index, digest in enumerate(hashes):
        if digest not in seen:
            seen.add(digest)
            indices.append(index)
    return seen, indices


def _passes_filters(texts: list[str], filters: SyntheticFilters) -> bool:
    for text in texts:
        length = len(text)
        if length < filters.min_length or length > filters.max_length:
            return False
        if _score_quality(text) < filters.min_quality_score:
            return False
    return True


def _score_quality(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    words = text.split()
    if len(words) < 3:
        return 0.2
    unique_ratio = len(set(words)) / len(words)
    return min(1.0, 0.5 + unique_ratio * 0.5)


def apply_filters(
    dataset: Table,
    filters: SyntheticFilters,
    *,
    method: TrainingMethod,
) -> Table:
    """Apply length and quality filters."""

    def filter_fn(example: dict) -> bool:
        if method == TrainingMethod.DPO:
            texts = [example.get("prompt", ""), example.get("chosen", ""), example.get("rejected", "")]
        else:
            texts = [example.get("text", "")]
        return _passes_filters([str(t) for t in texts], filters)

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
        for i in range(len(ds)):
            row = ds[i]
            h.update(str(sorted(row.items())).encode())
    return h.hexdigest()[:16]


def compute_data_cache_key(config: DataConfig, method: TrainingMethod) -> str:
    """Content-addressed cache key from method, data config, and source mtime."""
    payload = {
        "method": method.value,
        "data": config.model_dump(mode="json"),
    }
    if config.dataset and config.dataset.format.value != "hf_hub":
        path = Path(config.dataset.path)
        if path.exists():
            payload["mtime_ns"] = path.stat().st_mtime_ns
            payload["path"] = str(path.resolve())
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def save_processed_cache(cache_dir: Path, dataset_dict: TableDict, content_hash: str, method: str) -> None:
    """Persist preprocessed splits as Parquet (with JSONL fallback)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    format_used = "parquet"
    try:
        for split_name, table in dataset_dict.items():
            split_dir = cache_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            _table_to_parquet(table, split_dir / "data.parquet")
    except Exception:
        format_used = "jsonl"
        for split_name, table in dataset_dict.items():
            table.save_jsonl(cache_dir / split_name)

    manifest = {
        "hash": content_hash,
        "method": method,
        "splits": list(dataset_dict.keys()),
        "format": format_used,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_processed_cache(cache_dir: Path, expected_hash: str | None = None) -> TableDict | None:
    """Load cached preprocessed splits when hash matches."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_hash and manifest.get("hash") != expected_hash:
        return None

    splits = manifest.get("splits", ["train", "validation"])
    cache_format = manifest.get("format", "jsonl")
    dataset_dict = TableDict()
    for split_name in splits:
        split_dir = cache_dir / split_name
        if cache_format == "parquet":
            parquet_path = split_dir / "data.parquet"
            if not parquet_path.exists():
                return None
            dataset_dict[split_name] = _parquet_to_table(parquet_path)
        elif (split_dir / "data.jsonl").exists():
            dataset_dict[split_name] = Table.load_jsonl(split_dir)
        else:
            return None

    return dataset_dict if dataset_dict else None


def _table_to_parquet(table: Table, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = table.to_dict()
    if not data:
        pq.write_table(pa.table({}), path)
        return
    pq.write_table(pa.table(data), path)


def _parquet_to_table(path: Path) -> Table:
    import pyarrow.parquet as pq

    arrow_table = pq.read_table(path)
    return Table.from_dict(arrow_table.to_pydict())
