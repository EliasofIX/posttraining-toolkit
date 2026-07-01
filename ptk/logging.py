"""Structured logging for human and machine (--machine) modes."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from rich.console import Console
from rich.theme import Theme

_console = Console(theme=Theme({"info": "cyan", "warn": "yellow", "error": "red bold"}))


class EventType(str, Enum):
    START = "start"
    PROGRESS = "progress"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    COMPLETE = "complete"
    METRIC = "metric"


class Logger:
    """Dual-mode logger: Rich for humans, JSONL for agents."""

    def __init__(self, machine: bool = False, verbose: bool = False) -> None:
        self.machine = machine
        self.verbose = verbose

    def _emit(self, event_type: EventType, message: str, **fields: Any) -> None:
        if self.machine:
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event_type.value,
                "message": message,
                **fields,
            }
            sys.stdout.write(json.dumps(record, default=str) + "\n")
            sys.stdout.flush()
            return

        prefix = {
            EventType.START: "[info]▶[/info]",
            EventType.PROGRESS: "[info]…[/info]",
            EventType.INFO: "[info]ℹ[/info]",
            EventType.WARN: "[warn]⚠[/warn]",
            EventType.ERROR: "[error]✗[/error]",
            EventType.COMPLETE: "[info]✓[/info]",
            EventType.METRIC: "[info]#[/info]",
        }[event_type]
        extra = f" {fields}" if fields and self.verbose else ""
        _console.print(f"{prefix} {message}{extra}")

    def start(self, message: str, **fields: Any) -> None:
        self._emit(EventType.START, message, **fields)

    def progress(self, message: str, **fields: Any) -> None:
        self._emit(EventType.PROGRESS, message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._emit(EventType.INFO, message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        self._emit(EventType.WARN, message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._emit(EventType.ERROR, message, **fields)

    def complete(self, message: str, **fields: Any) -> None:
        self._emit(EventType.COMPLETE, message, **fields)

    def metric(self, name: str, value: Any, **fields: Any) -> None:
        self._emit(EventType.METRIC, name, metric=name, value=value, **fields)
