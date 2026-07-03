"""Pipeline optimization tests."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ptk.config.loader import load_config
from ptk.config.schema import ExportFormat, PTKConfig, TrainingMethod
from ptk.data.loaders import count_dataset_samples
from ptk.data.pipeline import load_processed_cache, save_processed_cache
from ptk.data.table import Table, TableDict
from ptk.export.formats import _export_gguf, _export_merged_fp16
from ptk.logging import Logger
from ptk.pipeline import plan_run, run_pipeline

FIXTURES = Path(__file__).parent / "fixtures"


def test_eval_skipped_when_benchmarks_empty(tmp_path):
    config = load_config(FIXTURES / "sft.yaml")
    config.output.dir = str(tmp_path / "outputs")
    config.eval.benchmarks = []
    config.training.max_iters = 1

    logger = Logger(machine=False, verbose=False)
    with patch("ptk.pipeline.EvalHarness") as mock_harness:
        run_pipeline(config, logger, skip_export=True)
        mock_harness.assert_not_called()


def test_synthetic_data_no_round_trip(tmp_path):
    config = load_config(FIXTURES / "sft.yaml")
    config.output.dir = str(tmp_path / "outputs")
    from ptk.config.schema import DataSource, SyntheticConfig

    config.data.source = DataSource.SYNTHETIC
    config.data.synthetic = SyntheticConfig(n_samples=5)
    config.training.max_iters = 1

    logger = Logger(machine=False, verbose=False)
    with patch("ptk.pipeline.generate_synthetic_data") as mock_gen:
        mock_gen.return_value = Table.from_dict({"text": [f"sample {i}" for i in range(10)]})
        run_pipeline(config, logger, skip_eval=True, skip_export=True)

    data_cache = config.output_path() / "data_cache"
    assert not (data_cache / "raw").exists()
    assert (data_cache / "processed" / "manifest.json").exists()


def test_plan_run_fast_sample_count(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("\n".join(json.dumps({"text": f"line {i}"}) for i in range(50)) + "\n")

    from ptk.config.schema import DataConfig, DatasetConfig, DatasetFormat

    config = PTKConfig(
        run_name="plan-test",
        base_model="distilgpt2",
        method=TrainingMethod.SFT,
        data=DataConfig(
            dataset=DatasetConfig(path=str(data_path), format=DatasetFormat.JSONL),
        ),
    )

    plan = plan_run(config)
    assert plan["data_samples"] == 50


def test_count_dataset_samples_jsonl(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text": "a"}\n{"text": "b"}\n\n', encoding="utf-8")

    from ptk.config.schema import DatasetConfig, DatasetFormat

    count = count_dataset_samples(DatasetConfig(path=str(data_path), format=DatasetFormat.JSONL))
    assert count == 2


def test_table_to_dict_zero_copy_columns():
    table = Table.from_dict({"text": ["a", "b"], "label": [1, 2]})
    data = table.to_dict()
    assert data["text"] == ["a", "b"]
    assert data["label"] == [1, 2]


def test_data_cache_round_trip(tmp_path):
    dataset_dict = TableDict(
        {
            "train": Table.from_dict({"text": ["one", "two"]}),
            "validation": Table.from_dict({"text": ["three"]}),
        }
    )
    cache_dir = tmp_path / "processed"
    save_processed_cache(cache_dir, dataset_dict, "abc123", "sft")

    loaded = load_processed_cache(cache_dir, expected_hash="abc123")
    assert loaded is not None
    assert len(loaded["train"]) == 2
    assert len(loaded["validation"]) == 1

    assert load_processed_cache(cache_dir, expected_hash="wrong") is None


def test_merged_export_reuses_existing(tmp_path):
    config = load_config(FIXTURES / "sft.yaml")
    export_dir = tmp_path / "export"
    merged = export_dir / "merged_fp16"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    (merged / "model.safetensors").write_text("placeholder", encoding="utf-8")

    out_dir = export_dir / "merged_copy"
    result = _export_merged_fp16(
        config,
        tmp_path / "missing-checkpoint",
        out_dir,
        export_dir=export_dir,
    )
    assert Path(result).exists()
    assert (out_dir / "config.json").exists()


@pytest.mark.slow
def test_gguf_reuses_merged_fp16(tmp_path):
    config = load_config(FIXTURES / "sft.yaml")
    export_dir = tmp_path / "export"
    merged = export_dir / "merged_fp16"
    merged.mkdir(parents=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model.save_pretrained(merged)
    tokenizer.save_pretrained(merged)

    merge_calls = 0
    original = _export_merged_fp16

    def counting_merge(*args, **kwargs):
        nonlocal merge_calls
        merge_calls += 1
        return original(*args, **kwargs)

    with patch("ptk.export.formats._export_merged_fp16", side_effect=counting_merge):
        out_path = export_dir / "model.gguf"
        result = _export_gguf(config, tmp_path / "checkpoint", out_path, export_dir)
        assert Path(result).exists()
        assert merge_calls == 0
