"""Export MLX QLoRA adapters and fused models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ptk.config.schema import PTKConfig
from ptk.exceptions import RuntimeError as PTKRuntimeError
from ptk.logging import Logger
from ptk.mlx.availability import require_mlx
from ptk.mlx.trainer import is_mlx_checkpoint


def read_mlx_runtime(checkpoint: Path) -> dict:
    runtime = checkpoint / "ptk_runtime.json"
    if runtime.exists():
        return json.loads(runtime.read_text(encoding="utf-8"))
    adapter_cfg = checkpoint / "adapter_config.json"
    if adapter_cfg.exists():
        data = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        return {
            "ptk_backend": data.get("ptk_backend", "mlx_quant"),
            "base_model": data.get("base_model"),
            "mlx_model_path": data.get("mlx_model_path"),
        }
    return {"ptk_backend": "mlx_quant"}


def export_mlx_adapter_only(checkpoint: Path, out_dir: Path) -> str:
    """Copy MLX adapter directory to export location."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        shutil.copytree(checkpoint, out_dir, dirs_exist_ok=True)
    return str(out_dir)


def export_mlx_merged(
    config: PTKConfig,
    checkpoint: Path,
    out_dir: Path,
    *,
    logger: Logger | None = None,
    dequantize: bool = True,
) -> str:
    """Fuse LoRA adapters into the MLX base model and save (mlx-lm fuse pattern)."""
    require_mlx()
    log = logger or Logger()
    if not is_mlx_checkpoint(checkpoint):
        raise PTKRuntimeError(
            f"Not an MLX checkpoint: {checkpoint}",
            code="MLX_EXPORT_FAILED",
        )

    runtime = read_mlx_runtime(checkpoint)
    mlx_model = runtime.get("mlx_model_path") or config.base_model
    if not mlx_model:
        raise PTKRuntimeError(
            "MLX merge requires mlx_model_path in ptk_runtime.json",
            code="MLX_EXPORT_FAILED",
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.start("Fusing MLX adapters", model=mlx_model, adapter=str(checkpoint))

    try:
        from mlx.utils import tree_unflatten
        from mlx_lm.utils import dequantize_model, load, save
    except ImportError as exc:
        raise PTKRuntimeError(f"MLX fuse imports failed: {exc}", code="MLX_IMPORT_FAILED") from exc

    try:
        model, tokenizer, model_config = load(
            str(mlx_model),
            adapter_path=str(checkpoint),
            return_config=True,
            tokenizer_config={"trust_remote_code": True},
        )

        fused_linears = [
            (n, m.fuse(dequantize=dequantize))
            for n, m in model.named_modules()
            if hasattr(m, "fuse")
        ]
        if fused_linears:
            model.update_modules(tree_unflatten(fused_linears))

        if dequantize:
            model = dequantize_model(model)
            if isinstance(model_config, dict):
                model_config.pop("quantization", None)
                model_config.pop("quantization_config", None)

        save(
            out_dir,
            str(mlx_model),
            model,
            tokenizer,
            model_config,
            donate_model=False,
        )
        (out_dir / "ptk_runtime.json").write_text(
            json.dumps(
                {
                    "ptk_backend": "mlx_quant",
                    "fused": True,
                    "dequantized": dequantize,
                    "base_model": config.base_model,
                    "mlx_model_path": str(mlx_model),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        raise PTKRuntimeError(
            f"MLX fuse/export failed: {exc}",
            code="MLX_EXPORT_FAILED",
            details={"checkpoint": str(checkpoint), "model": str(mlx_model)},
        ) from exc

    log.complete("Fused MLX model", path=str(out_dir))
    return str(out_dir)
