"""Apple Silicon MLX integration for true 4-bit QLoRA.

Heavy imports (mlx, mlx_lm) are lazy so CUDA-only installs never load Metal packages.
"""

from __future__ import annotations

from ptk.mlx.availability import (
    mlx_available,
    mlx_import_error,
    require_mlx,
)

__all__ = [
    "mlx_available",
    "mlx_import_error",
    "require_mlx",
]
