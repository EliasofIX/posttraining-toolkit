"""Structured errors for agent-legible CLI output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PTKError(Exception):
    """Base toolkit error with stable error code."""

    code: str
    message: str
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field_path:
            payload["field_path"] = self.field_path
        if self.details:
            payload["details"] = self.details
        return payload


class ValidationError(PTKError):
    """Config or input validation failure."""

    def __init__(
        self,
        message: str,
        *,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="VALIDATION_ERROR",
            message=message,
            field_path=field_path,
            details=details or {},
        )


class RuntimeError(PTKError):
    """Pipeline execution failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RUNTIME_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code=code, message=message, details=details or {})


class ResumeDriftError(PTKError):
    """Resume attempted but config drifted from original run."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="RESUME_DRIFT",
            message=message,
            details=details or {},
        )


# Stable exit codes documented for agents
EXIT_SUCCESS = 0
EXIT_VALIDATION = 1
EXIT_RUNTIME = 2
EXIT_RESUME_DRIFT = 3
