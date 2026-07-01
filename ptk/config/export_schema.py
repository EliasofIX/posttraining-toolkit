"""Emit JSON Schema for agent consumption."""

from __future__ import annotations

import json
from typing import Any

from ptk.config.schema import PTKConfig


def export_json_schema() -> dict[str, Any]:
    """Return the full JSON Schema for PTKConfig."""
    return PTKConfig.model_json_schema()


def export_json_schema_string(*, indent: int = 2) -> str:
    """Return JSON Schema as a formatted string."""
    return json.dumps(export_json_schema(), indent=indent)


def write_json_schema(path: str) -> None:
    """Write JSON Schema to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(export_json_schema_string())
