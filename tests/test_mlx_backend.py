"""Unit tests for MLX QLoRA backend resolution and bridges (no Metal required)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ptk.config.schema import (
    DeviceType,
    QuantBackend,
    QuantBits,
    QuantizationConfig,
    QuantType,
)
from ptk.data.table import Table, TableDict
from ptk.exceptions import ValidationError
from ptk.mlx.data_bridge import write_mlx_dataset
from ptk.mlx.model_cache import is_mlx_model_dir, mlx_cache_key
from ptk.training.quant_backend import resolve_quantization_backend


def _quant(**kwargs) -> QuantizationConfig:
    base = dict(
        bits=QuantBits.FOUR,
        quant_type=QuantType.NF4,
        backend=QuantBackend.AUTO,
        double_quant=True,
        group_size=64,
        allow_unquantized_fallback=False,
    )
    base.update(kwargs)
    return QuantizationConfig(**base)


def test_resolve_auto_cuda_bnb():
    assert resolve_quantization_backend(DeviceType.CUDA, _quant()) == "bnb"


def test_resolve_auto_cpu_none():
    assert resolve_quantization_backend(DeviceType.CPU, _quant()) == "none"


def test_resolve_bnb_requires_cuda():
    with pytest.raises(ValidationError, match="requires CUDA"):
        resolve_quantization_backend(DeviceType.MPS, _quant(backend=QuantBackend.BNB))


def test_resolve_auto_mps_with_mlx():
    with patch("ptk.training.quant_backend.mlx_available", return_value=True):
        assert resolve_quantization_backend(DeviceType.MPS, _quant()) == "mlx_quant"


def test_resolve_auto_mps_without_mlx_fails():
    with patch("ptk.training.quant_backend.mlx_available", return_value=False):
        with pytest.raises(ValidationError, match="requires MLX"):
            resolve_quantization_backend(DeviceType.MPS, _quant())


def test_resolve_auto_mps_fallback_allowed():
    with patch("ptk.training.quant_backend.mlx_available", return_value=False):
        q = _quant(allow_unquantized_fallback=True)
        assert resolve_quantization_backend(DeviceType.MPS, q) == "none"


def test_resolve_explicit_mlx_requires_packages():
    with patch("ptk.training.quant_backend.require_mlx", side_effect=ValidationError("no mlx")):
        with pytest.raises(ValidationError):
            resolve_quantization_backend(DeviceType.MPS, _quant(backend=QuantBackend.MLX_QUANT))


def test_mlx_cache_key_stable():
    q = _quant(group_size=64)
    a = mlx_cache_key("org/model", q)
    b = mlx_cache_key("org/model", q)
    assert a == b
    assert len(a) == 16
    c = mlx_cache_key("org/other", q)
    assert a != c


def test_is_mlx_model_dir(tmp_path: Path):
    assert not is_mlx_model_dir(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert is_mlx_model_dir(tmp_path)


def test_write_mlx_dataset(tmp_path: Path):
    train = Table.from_records([{"text": f"sample {i} enough length"} for i in range(4)])
    val = Table.from_records([{"text": "validation row enough"}])
    out = write_mlx_dataset(TableDict({"train": train, "validation": val}), tmp_path / "mlx_data", batch_size=2)
    assert (out / "train.jsonl").exists()
    assert (out / "valid.jsonl").exists()
    lines = (out / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["text"].startswith("sample")


def test_write_mlx_dataset_batch_size_guard(tmp_path: Path):
    train = Table.from_records([{"text": "only one"}])
    with pytest.raises(ValidationError, match="batch_size"):
        write_mlx_dataset(TableDict({"train": train}), tmp_path / "mlx_data", batch_size=2)
