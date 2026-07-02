"""Minimal YAML loader/dumper for config files (no PyYAML dependency)."""

from __future__ import annotations

import re
from typing import Any

_SCALAR_RE = re.compile(
    r"^(?P<value>"
    r"null|true|false|"
    r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|"
    r"'(?:[^'\\]|\\.)*'|"
    r'"(?:[^"\\]|\\.)*"|'
    r"[^#'\"]+?"
    r")\s*(?:#.*)?$"
)


def load_yaml(text: str) -> Any:
    """Parse a YAML document into Python objects."""
    lines = text.splitlines()
    root, _ = _parse_block(lines, 0, 0)
    return root


def dump_yaml(obj: Any, *, indent: int = 0) -> str:
    """Serialize Python objects to YAML."""
    return "\n".join(_emit(obj, indent)) + "\n"


def _parse_block(lines: list[str], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        return {}, start

    first = _line_content(lines[start])
    if first.startswith("- "):
        return _parse_list(lines, start, indent)

    mapping: dict[str, Any] = {}
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        current_indent = _leading_spaces(lines[i])
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation at line {i + 1}")

        content = _line_content(lines[i])
        if content.startswith("- "):
            break

        if ":" not in content:
            raise ValueError(f"Expected key at line {i + 1}: {content!r}")

        key, rest = content.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        if rest:
            mapping[key] = _parse_scalar(rest)
            i += 1
            continue

        if i + 1 < len(lines) and _leading_spaces(lines[i + 1]) > indent:
            child_indent = _leading_spaces(lines[i + 1])
            child, i = _parse_block(lines, i + 1, child_indent)
            mapping[key] = child
            continue

        mapping[key] = None
        i += 1

    return mapping, i


def _parse_list(lines: list[str], start: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        current_indent = _leading_spaces(lines[i])
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected list indentation at line {i + 1}")

        content = _line_content(lines[i])
        if not content.startswith("- "):
            break

        item_text = content[2:].strip()
        if item_text:
            items.append(_parse_scalar(item_text))
            i += 1
            continue

        if i + 1 < len(lines) and _leading_spaces(lines[i + 1]) > indent:
            child_indent = _leading_spaces(lines[i + 1])
            child, i = _parse_block(lines, i + 1, child_indent)
            items.append(child)
            continue

        items.append(None)
        i += 1

    return items, i


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _line_content(line: str) -> str:
    stripped = line.strip()
    if "#" in stripped:
        in_single = False
        in_double = False
        for idx, char in enumerate(stripped):
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            elif char == "#" and not in_single and not in_double:
                return stripped[:idx].rstrip()
    return stripped


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value or value == "~":
        return None
    if value in ("null", "Null", "NULL", "~"):
        return None
    if value in ("true", "True", "TRUE"):
        return True
    if value in ("false", "False", "FALSE"):
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
        return float(value)
    return value


def _emit(obj: Any, indent: int) -> list[str]:
    prefix = "  " * indent
    if isinstance(obj, dict):
        lines: list[str] = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_emit(value, indent + 1))
            elif value is None:
                lines.append(f"{prefix}{key}: null")
            elif isinstance(value, bool):
                lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
            elif isinstance(value, str):
                if any(c in value for c in ":#{}[]&*!|>'\"%@`"):
                    lines.append(f'{prefix}{key}: "{value}"')
                else:
                    lines.append(f"{prefix}{key}: {value}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return lines

    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_emit(item, indent + 1))
            elif item is None:
                lines.append(f"{prefix}- null")
            elif isinstance(item, bool):
                lines.append(f"{prefix}- {'true' if item else 'false'}")
            elif isinstance(item, str):
                lines.append(f'{prefix}- "{item}"' if " " in item else f"{prefix}- {item}")
            else:
                lines.append(f"{prefix}- {item}")
        return lines

    if obj is None:
        return [f"{prefix}null"]
    return [f"{prefix}{obj}"]
