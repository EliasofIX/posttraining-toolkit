"""Minimal terminal styling without third-party color libraries."""

from __future__ import annotations

import os
import sys

_STYLES = {
    "info": "\033[36m",
    "warn": "\033[33m",
    "error": "\033[1;31m",
    "reset": "\033[0m",
}


def supports_color() -> bool:
    isatty = getattr(sys.stdout, "isatty", None)
    if not callable(isatty) or not isatty():
        return False
    return os.environ.get("NO_COLOR") is None


def colorize(text: str, style: str) -> str:
    if not supports_color():
        return text
    return f"{_STYLES.get(style, '')}{text}{_STYLES['reset']}"


def print_styled(prefix: str, message: str, *, extra: str = "") -> None:
    line = f"{prefix} {message}{extra}"
    print(line, flush=True)
