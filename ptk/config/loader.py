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
    directory only (never the process cwd) so a missing local file cannot silently
    bind to an unrelated dataset in cwd.

    Dataset paths are left as written in the config (relative paths stay relative)
    for portable registry snapshots. Runtime loaders resolve via ``config.config_dir``.

    When ``check_dataset_path`` is True (default), local dataset paths must exist
    and must be files. Use ``check_dataset_path=False`` for scaffolding.
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

    config_dir = path.parent.resolve()
    config = validate_config_dict(
        data,
        check_dataset_path=check_dataset_path,
        config_dir=config_dir,
    )
    config.set_config_dir(config_dir)
    return config


def resolve_dataset_path(path_str: str, *, config_dir: Path | None = None) -> Path:
    """Resolve a dataset path.

    - Absolute paths are returned as-is (resolved).
    - Relative paths require ``config_dir`` and resolve **only** against it.
    - Relative paths without ``config_dir`` raise ``ValidationError`` (no cwd fallback).
    """
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    if config_dir is None:
        raise ValidationError(
            f"Relative dataset path requires config_dir: {path_str}",
            field_path="data.dataset.path",
            details={
                "errors": [
                    {
                        "field_path": "data.dataset.path",
                        "message": (
                            f"Relative dataset path requires config_dir: {path_str}. "
                            "Load the config from a file, or pass an absolute path."
                        ),
                        "type": "value_error",
                    }
                ]
            },
        )
    return (config_dir / path).resolve()


def with_resolved_dataset_paths(config: PTKConfig) -> PTKConfig:
    """Return a copy with local dataset path absolutized for child process launches."""
    if config.data.source != DataSource.DATASET or config.data.dataset is None:
        return config
    dataset = config.data.dataset
    if dataset.format == DatasetFormat.HF_HUB:
        return config
    resolved = resolve_dataset_path(dataset.path, config_dir=config.config_dir)
    if str(resolved) == dataset.path:
        return config
    return config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"dataset": dataset.model_copy(update={"path": str(resolved)})}
            )
        }
    )


def check_dataset_path_exists(config: PTKConfig, *, config_dir: Path | None = None) -> None:
    """Raise ValidationError if a local dataset path is missing or not a file."""
    if config.data.source != DataSource.DATASET or config.data.dataset is None:
        return
    dataset = config.data.dataset
    if dataset.format == DatasetFormat.HF_HUB:
        return
    base = config_dir if config_dir is not None else config.config_dir
    path = resolve_dataset_path(dataset.path, config_dir=base)
    if not path.exists():
        raise ValidationError(
            f"Dataset path not found: {path}",
            field_path="data.dataset.path",
            details={
                "errors": [
                    {
                        "field_path": "data.dataset.path",
                        "message": f"Dataset path not found: {path}",
                        "type": "value_error",
                    }
                ]
            },
        )
    if path.is_dir():
        raise ValidationError(
            f"Dataset path is a directory, expected a file: {path}",
            field_path="data.dataset.path",
            details={
                "errors": [
                    {
                        "field_path": "data.dataset.path",
                        "message": f"Dataset path is a directory, expected a file: {path}",
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

    if config_dir is not None:
        config.set_config_dir(config_dir)
    if check_dataset_path:
        check_dataset_path_exists(config, config_dir=config_dir)
    return config


def config_to_yaml(config: PTKConfig) -> str:
    """Serialize config to YAML."""
    return dump_yaml(config.model_dump(mode="json"))
