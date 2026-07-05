"""Distributed detection tests."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ptk.config.schema import ComputeStrategy, DeviceType
from ptk.distributed.detect import detect_device, detect_environment, is_main_process, resolve_mixed_precision


def test_detect_device_auto():
    device = detect_device(DeviceType.AUTO)
    assert device in (DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU)


def test_detect_environment():
    env = detect_environment(device=DeviceType.AUTO, strategy=ComputeStrategy.AUTO)
    assert env.device in (DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU)
    assert env.strategy is not None


def test_mps_rejects_multi_gpu():
    pytest.importorskip("torch")
    import torch

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")

    from ptk.exceptions import ValidationError

    with pytest.raises(ValidationError):
        detect_environment(device=DeviceType.MPS, strategy=ComputeStrategy.MULTI_GPU)


def test_resolve_mixed_precision():
    result = resolve_mixed_precision(DeviceType.CPU, "auto")
    assert result == "fp32"

    result = resolve_mixed_precision(DeviceType.CPU, "fp16")
    assert result == "fp16"


def test_is_main_process():
    assert is_main_process() is True

    with patch.dict(os.environ, {"LOCAL_RANK": "1"}):
        assert is_main_process() is False

    with patch.dict(os.environ, {"RANK": "2", "LOCAL_RANK": "0"}, clear=False):
        assert is_main_process() is False

    with patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}, clear=False):
        assert is_main_process() is True


def test_rank_zero_gates_data_prep(tmp_path, monkeypatch):
    from ptk.config.loader import load_config
    from ptk.pipeline import run_pipeline

    config = load_config(Path(__file__).parent / "fixtures" / "sft.yaml")
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        '{"text": "first sample long enough for quality filters"}\n'
        '{"text": "second sample long enough for quality filters"}\n',
        encoding="utf-8",
    )
    config.data.dataset.path = str(data_path)
    config.output.dir = str(tmp_path / "outputs")
    config.training.max_iters = 1
    config.eval.benchmarks = []

    logger = __import__("ptk.logging", fromlist=["Logger"]).Logger(machine=False, verbose=False)
    prepare_calls = {"count": 0}
    original_prepare = __import__("ptk.pipeline", fromlist=["_prepare_data"])._prepare_data

    def counting_prepare(*args, **kwargs):
        prepare_calls["count"] += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr("ptk.pipeline._prepare_data", counting_prepare)
    monkeypatch.setattr("ptk.pipeline.is_main_process", lambda: True)

    run_pipeline(config, logger, skip_eval=True, skip_export=True)
    assert prepare_calls["count"] == 1
