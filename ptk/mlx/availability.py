"""Detect optional MLX / mlx-lm installation."""

from __future__ import annotations

from ptk.exceptions import ValidationError

_MLX_INSTALL_HINT = "pip install -e '.[mlx]'"


def mlx_available() -> bool:
    """Return True when both mlx and mlx_lm import successfully."""
    try:
        import mlx  # noqa: F401
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def mlx_import_error() -> str | None:
    """Return the import error message if MLX is unavailable."""
    try:
        import mlx  # noqa: F401
        import mlx_lm  # noqa: F401

        return None
    except ImportError as exc:
        return str(exc)


def require_mlx(*, field_path: str = "training.quantization.backend") -> None:
    """Raise ValidationError when MLX extras are missing."""
    err = mlx_import_error()
    if err is None:
        return
    raise ValidationError(
        f"MLX quantization requires the mlx extra packages ({_MLX_INSTALL_HINT}). "
        f"Import error: {err}",
        field_path=field_path,
        details={"install": _MLX_INSTALL_HINT, "import_error": err},
    )
