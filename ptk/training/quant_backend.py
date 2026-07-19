"""Resolve QLoRA quantization backend from device + config."""

from __future__ import annotations

from typing import Literal

from ptk.config.schema import DeviceType, QuantBackend, QuantizationConfig
from ptk.exceptions import ValidationError
from ptk.mlx.availability import mlx_available, require_mlx

QuantBackendName = Literal["bnb", "mlx_quant", "none"]


def resolve_quantization_backend(
    device: DeviceType,
    quant: QuantizationConfig,
) -> QuantBackendName:
    """Select quantization backend based on device and requested config.

    Stock rules:
    - AUTO + CUDA → bnb
    - AUTO + MPS → mlx_quant if MLX installed, else fail (unless allow_unquantized_fallback)
    - AUTO + CPU → none (unquantized LoRA smoke path)
    - Explicit bnb requires CUDA
    - Explicit mlx_quant requires MLX packages
    """
    requested = quant.backend

    if requested == QuantBackend.BNB:
        if device != DeviceType.CUDA:
            raise ValidationError(
                f"training.quantization.backend=bnb requires CUDA (detected device={device.value}). "
                "On Apple Silicon use backend=auto or mlx_quant with: pip install -e '.[mlx]'",
                field_path="training.quantization.backend",
            )
        return "bnb"

    if requested == QuantBackend.MLX_QUANT:
        require_mlx(field_path="training.quantization.backend")
        return "mlx_quant"

    # AUTO
    if device == DeviceType.CUDA:
        return "bnb"

    if device == DeviceType.MPS:
        if mlx_available():
            return "mlx_quant"
        if quant.allow_unquantized_fallback:
            return "none"
        raise ValidationError(
            "QLoRA on Apple Silicon requires MLX for true 4-bit quantization. "
            "Install with: pip install -e '.[mlx]'. "
            "Or set training.quantization.allow_unquantized_fallback: true for fp16 PEFT LoRA "
            "(not true QLoRA). For unquantized LoRA prefer method: lora.",
            field_path="training.quantization.backend",
            details={"install": "pip install -e '.[mlx]'"},
        )

    # CPU (and any other)
    return "none"
