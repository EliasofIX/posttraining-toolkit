"""Distributed detection tests."""

import pytest

from ptk.config.schema import ComputeStrategy, DeviceType
from ptk.distributed.detect import detect_device, detect_environment, resolve_mixed_precision


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
