"""Performance regression tests for data preprocessing."""

import json
import time
from pathlib import Path

from ptk.config.schema import DataConfig, DatasetConfig, DatasetFormat, PTKConfig, TrainingMethod
from ptk.data.pipeline import compute_data_cache_key, preprocess_dataset, save_processed_cache, load_processed_cache
from ptk.data.table import Table, TableDict


def test_preprocess_10k_rows_under_threshold(tmp_path):
    data_path = tmp_path / "train.jsonl"
    rows = [json.dumps({"text": f"sample {i} with enough words for quality"}) for i in range(10_000)]
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    config = DataConfig(
        dataset=DatasetConfig(path=str(data_path), format=DatasetFormat.JSONL),
    )
    start = time.monotonic()
    result = preprocess_dataset(config, TrainingMethod.SFT, num_proc=2)
    elapsed = time.monotonic() - start

    assert len(result["train"]) > 0
    assert elapsed < 30.0


def test_content_addressed_cache_key_stable(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text": "hello"}\n', encoding="utf-8")

    config = DataConfig(
        dataset=DatasetConfig(path=str(data_path), format=DatasetFormat.JSONL),
    )
    key1 = compute_data_cache_key(config, TrainingMethod.SFT)
    key2 = compute_data_cache_key(config, TrainingMethod.SFT)
    assert key1 == key2
