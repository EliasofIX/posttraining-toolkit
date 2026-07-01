"""Configuration loading and validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from ptk.config.schema import PTKConfig
from ptk.exceptions import ValidationError


def load_config(path: str | Path) -> PTKConfig:
    """Load and validate a config from YAML or JSON."""
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"Config file not found: {path}", field_path="config_path")

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(raw)
    elif suffix == ".json":
        data = json.loads(raw)
    else:
        raise ValidationError(
            f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json",
            field_path="config_path",
        )

    if not isinstance(data, dict):
        raise ValidationError("Config root must be a mapping/object", field_path="$")

    return validate_config_dict(data)


def validate_config_dict(data: dict[str, Any]) -> PTKConfig:
    """Validate a config dict and return structured errors on failure."""
    try:
        return PTKConfig.model_validate(data)
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


def config_to_yaml(config: PTKConfig) -> str:
    """Serialize config to YAML."""
    return yaml.dump(
        config.model_dump(mode="json"),
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
