"""Configuration loading and validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from ptk.config.schema import DatasetFormat, DataSource, PTKConfig
from ptk.config.yaml_io import dump_yaml, load_yaml
from ptk.exceptions import ValidationError


def load_config(path: str | Path, *, check_dataset_path: bool = True) -> PTKConfig:
    """Load and validate a config from YAML or JSON.

    Relative ``data.dataset.path`` values are resolved against the config file's
    directory first, then the process cwd (whichever exists). When neither exists,
    the config-relative path is kept for the error message.

    When ``check_dataset_path`` is True (default), local dataset paths must exist.
    Use ``check_dataset_path=False`` for scaffolding or offline schema checks.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"Config file not found: {path}", field_path="config_path")

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            data = load_yaml(raw)
        except ValueError as exc:
            raise ValidationError(str(exc), field_path="config_path") from exc
    elif suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid JSON config: {exc.msg}", field_path="config_path") from exc
    else:
        raise ValidationError(
            f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json",
            field_path="config_path",
        )

    if not isinstance(data, dict):
        raise ValidationError("Config root must be a mapping/object", field_path="$")

    return validate_config_dict(
        data,
        check_dataset_path=check_dataset_path,
        config_dir=path.parent.resolve(),
    )


def resolve_dataset_path(path_str: str, *, config_dir: Path | None = None) -> Path:
    """Resolve a dataset path against config dir and/or cwd.

    Preference order for relative paths:
    1. Absolute path as given
    2. Path relative to ``config_dir`` if it exists
    3. Path relative to cwd if it exists
    4. Config-dir join (for stable error messages when loading from a file)
    5. Cwd join
    """
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    if config_dir is not None:
        candidates.append((config_dir / path).resolve())
    candidates.append(path.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if config_dir is not None:
        return (config_dir / path).resolve()
    return path.resolve()


def apply_dataset_path_resolution(config: PTKConfig, *, config_dir: Path | None) -> PTKConfig:
    """Return config with local dataset path resolved when possible."""
    if config.data.source != DataSource.DATASET or config.data.dataset is None:
        return config
    dataset = config.data.dataset
    if dataset.format == DatasetFormat.HF_HUB:
        return config
    resolved = resolve_dataset_path(dataset.path, config_dir=config_dir)
    if str(resolved) == dataset.path:
        return config
    return config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"dataset": dataset.model_copy(update={"path": str(resolved)})}
            )
        }
    )


def check_dataset_path_exists(config: PTKConfig) -> None:
    """Raise ValidationError if a local dataset path is missing."""
    if config.data.source != DataSource.DATASET or config.data.dataset is None:
        return
    dataset = config.data.dataset
    if dataset.format == DatasetFormat.HF_HUB:
        return
    path = Path(dataset.path)
    if not path.exists():
        raise ValidationError(
            f"Dataset path not found: {dataset.path}",
            field_path="data.dataset.path",
            details={
                "errors": [
                    {
                        "field_path": "data.dataset.path",
                        "message": f"Dataset path not found: {dataset.path}",
                        "type": "value_error",
                    }
                ]
            },
        )


def validate_config_dict(
    data: dict[str, Any],
    *,
    check_dataset_path: bool = True,
    config_dir: Path | None = None,
) -> PTKConfig:
    """Validate a config dict and return structured errors on failure."""
    try:
        config = PTKConfig.model_validate(data)
    except PydanticValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            errors.append(
                {
                    "field_path": loc or "$",
                    "message": err["msg"],
                    "type": err["type"],
                }
            )
        raise ValidationError(
            f"Config validation failed with {len(errors)} error(s)",
            details={"errors": errors},
        ) from exc

    config = apply_dataset_path_resolution(config, config_dir=config_dir)
    if check_dataset_path:
        check_dataset_path_exists(config)
    return config


def config_to_yaml(config: PTKConfig) -> str:
    """Serialize config to YAML."""
    return dump_yaml(config.model_dump(mode="json"))
