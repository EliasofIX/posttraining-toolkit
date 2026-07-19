"""Convert and cache HF models as quantized MLX weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ptk.config.schema import PTKConfig, QuantizationConfig
from ptk.exceptions import RuntimeError as PTKRuntimeError
from ptk.logging import Logger
from ptk.mlx.availability import require_mlx


def _model_identity(base_model: str) -> dict[str, Any]:
    path = Path(base_model)
    if path.exists():
        # Local path: include mtime of config or directory for cache invalidation.
        marker = path / "config.json" if path.is_dir() else path
        try:
            mtime = marker.stat().st_mtime_ns if marker.exists() else path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        return {"path": str(path.resolve()), "mtime_ns": mtime}
    return {"repo": base_model}


def mlx_cache_key(base_model: str, quant: QuantizationConfig) -> str:
    """Stable content key for a converted/quantized MLX model."""
    payload = {
        "base_model": _model_identity(base_model),
        "q_bits": int(quant.bits.value),
        "group_size": int(quant.group_size),
        "converter": "mlx_lm.convert",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return digest[:16]


def is_mlx_model_dir(path: Path) -> bool:
    """Heuristic: local directory already in MLX weight layout."""
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("weights*.npz"))
    # mlx-lm saves model.safetensors / weights.safetensors + config.json
    return has_config and bool(weight_files)


def ensure_mlx_model(
    config: PTKConfig,
    *,
    logger: Logger | None = None,
) -> tuple[Path, bool]:
    """Return path to a (possibly quantized) MLX model, converting if needed.

    Returns:
        (mlx_model_path, cache_hit)
    """
    require_mlx()
    log = logger or Logger()
    assert config.training.quantization is not None
    quant = config.training.quantization
    base = config.base_model
    base_path = Path(base)

    # Already a local MLX model directory — use as-is (user may pass pre-quantized).
    if base_path.exists() and is_mlx_model_dir(base_path):
        log.complete("Using existing MLX model directory", path=str(base_path.resolve()))
        return base_path.resolve(), True

    cache_key = mlx_cache_key(base, quant)
    cache_root = config.output_path() / "model_cache" / "mlx"
    cache_dir = cache_root / cache_key
    ready = cache_dir / ".ready"
    if ready.exists() and is_mlx_model_dir(cache_dir):
        log.complete("Loaded cached MLX model", path=str(cache_dir), cache_key=cache_key)
        return cache_dir, True

    # mlx_lm.convert refuses non-empty existing destinations — convert to a staging dir.
    cache_root.mkdir(parents=True, exist_ok=True)
    staging = cache_root / f".staging-{cache_key}"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)

    q_bits = int(quant.bits.value)
    log.start(
        "Converting model to MLX (quantized)",
        base_model=base,
        q_bits=q_bits,
        group_size=quant.group_size,
        path=str(cache_dir),
    )

    try:
        from mlx_lm.convert import convert
    except ImportError as exc:
        raise PTKRuntimeError(
            f"mlx-lm convert unavailable: {exc}",
            code="MLX_IMPORT_FAILED",
        ) from exc

    try:
        convert(
            hf_path=base,
            mlx_path=str(staging),
            quantize=True,
            q_bits=q_bits,
            q_group_size=int(quant.group_size),
            trust_remote_code=True,
        )
    except Exception as exc:
        if staging.exists():
            import shutil

            shutil.rmtree(staging, ignore_errors=True)
        raise PTKRuntimeError(
            f"Failed to convert/quantize model for MLX: {exc}",
            code="MLX_CONVERT_FAILED",
            details={"base_model": base, "q_bits": q_bits, "group_size": quant.group_size},
        ) from exc

    if not is_mlx_model_dir(staging):
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise PTKRuntimeError(
            f"MLX convert finished but model directory looks incomplete: {staging}",
            code="MLX_CONVERT_FAILED",
            details={"path": str(staging)},
        )

    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
    staging.rename(cache_dir)
    ready.write_text(json.dumps({"base_model": base, "q_bits": q_bits}, indent=2), encoding="utf-8")
    log.complete("MLX model ready", path=str(cache_dir), cache_key=cache_key, cache_hit=False)
    return cache_dir, False
