"""Run metadata, checkpoint index, and resumability."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ptk.config.schema import PTKConfig
from ptk.exceptions import ResumeDriftError
from ptk.registry.store import StoreBackend, get_store


class RunStatus(str, Enum):
    PENDING = "pending"
    DATA_GEN = "data_gen"
    TRAINING = "training"
    EVAL = "eval"
    EXPORT = "export"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class RunRecord:
    """Persisted run metadata."""

    run_id: str
    run_name: str
    status: RunStatus
    config: dict[str, Any]
    created_at: str
    updated_at: str
    git_commit: str | None = None
    data_hash: str | None = None
    checkpoint_path: str | None = None
    global_step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        data = dict(data)
        data["status"] = RunStatus(data["status"])
        return cls(**data)


class RunRegistry:
    """Manage training runs with idempotent resume support."""

    def __init__(self, store: StoreBackend | None = None) -> None:
        self.store = store or get_store()

    def create_run(self, config: PTKConfig) -> RunRecord:
        """Register a new run."""
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        record = RunRecord(
            run_id=run_id,
            run_name=config.run_name,
            status=RunStatus.PENDING,
            config=config.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
            git_commit=_git_commit(),
        )
        self.store.write(run_id, record.to_dict())
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        data = self.store.read(run_id)
        if data is None:
            return None
        return RunRecord.from_dict(data)

    def update_run(self, run_id: str, **updates: Any) -> RunRecord:
        record = self.get_run(run_id)
        if record is None:
            raise ValueError(f"Run not found: {run_id}")
        for key, value in updates.items():
            if key == "status" and isinstance(value, RunStatus):
                setattr(record, key, value)
            elif hasattr(record, key):
                setattr(record, key, value)
        record.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.write(run_id, record.to_dict())
        return record

    def list_runs(self) -> list[RunRecord]:
        records = []
        for key in self.store.list_keys():
            data = self.store.read(key)
            if data:
                records.append(RunRecord.from_dict(data))
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def find_latest_checkpoint(self, run_id: str) -> Path | None:
        record = self.get_run(run_id)
        if record is None or not record.checkpoint_path:
            return None
        path = Path(record.checkpoint_path)
        return path if path.exists() else None

    def validate_resume(self, run_id: str, config: PTKConfig) -> RunRecord:
        """Ensure resume config matches original run."""
        record = self.get_run(run_id)
        if record is None:
            raise ValueError(f"Run not found: {run_id}")

        original_hash = _config_hash(record.config)
        new_hash = _config_hash(config.model_dump(mode="json"))
        if original_hash != new_hash:
            raise ResumeDriftError(
                "Config drift detected: current config does not match original run",
                details={"run_id": run_id, "original_hash": original_hash, "new_hash": new_hash},
            )
        return record


def _config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
