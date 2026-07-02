"""Structured logging for human and machine (--machine) modes."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ptk.terminal import colorize, print_styled


class EventType(str, Enum):
    START = "start"
    PROGRESS = "progress"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    COMPLETE = "complete"
    METRIC = "metric"


class Logger:
    """Dual-mode logger: styled terminal for humans, JSONL for agents."""

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
            EventType.START: colorize("▶", "info"),
            EventType.PROGRESS: colorize("…", "info"),
            EventType.INFO: colorize("ℹ", "info"),
            EventType.WARN: colorize("⚠", "warn"),
            EventType.ERROR: colorize("✗", "error"),
            EventType.COMPLETE: colorize("✓", "info"),
            EventType.METRIC: colorize("#", "info"),
        }[event_type]
        extra = f" {fields}" if fields and self.verbose else ""
        print_styled(prefix, message, extra=extra)

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
