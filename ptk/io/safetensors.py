"""Minimal safetensors reader for model weight loading."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def load_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Load tensors from a safetensors file."""
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"Invalid safetensors file: {path}")

    (header_size,) = struct.unpack("<Q", data[:8])
    header_end = 8 + header_size
    header = json.loads(data[8:header_end].decode("utf-8"))

    tensors: dict[str, torch.Tensor] = {}
    data_view = memoryview(data)
    data_start = header_end
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_str = info["dtype"]
        shape = info["shape"]
        offsets = info["data_offsets"]
        start = data_start + offsets[0]
        end = data_start + offsets[1]
        torch_dtype = _DTYPE_MAP.get(dtype_str)
        if torch_dtype is None:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_str}")

        chunk = data_view[start:end]
        tensor = torch.frombuffer(bytearray(chunk), dtype=torch_dtype)
        tensors[name] = tensor.reshape(shape)

    return tensors
