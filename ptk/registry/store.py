"""Pluggable storage backend for run registry."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class StoreBackend(ABC):
    """Abstract storage backend."""

    @abstractmethod
    def read(self, key: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def write(self, key: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class LocalStore(StoreBackend):
    """Filesystem-backed store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_")
        return self.root / f"{safe}.json"

    def read(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, key: str, data: dict[str, Any]) -> None:
        path = self._path(key)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def list_keys(self, prefix: str = "") -> list[str]:
        keys = []
        for p in self.root.glob("*.json"):
            key = p.stem
            if not prefix or key.startswith(prefix.replace("/", "_")):
                keys.append(key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()


class S3Store(StoreBackend):
    """S3-compatible object store backend."""

    def __init__(self, bucket: str, prefix: str = "ptk-runs") -> None:
        self.bucket = bucket
        self.prefix = prefix
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 required for S3 store: pip install boto3") from exc
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("PTK_S3_ENDPOINT"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

    def _object_key(self, key: str) -> str:
        return f"{self.prefix}/{key}.json"

    def read(self, key: str) -> dict[str, Any] | None:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._object_key(key))
            return json.loads(obj["Body"].read().decode())
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception:
            return None

    def write(self, key: str, data: dict[str, Any]) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._object_key(key),
            Body=json.dumps(data, indent=2, default=str).encode(),
            ContentType="application/json",
        )

    def list_keys(self, prefix: str = "") -> list[str]:
        full_prefix = f"{self.prefix}/{prefix}"
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)
        keys = []
        for obj in response.get("Contents", []):
            name = obj["Key"].removeprefix(f"{self.prefix}/").removesuffix(".json")
            keys.append(name)
        return sorted(keys)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._object_key(key))


def get_store(root: str | None = None) -> StoreBackend:
    """Resolve store backend from environment or default local."""
    s3_bucket = os.environ.get("PTK_S3_BUCKET")
    if s3_bucket:
        return S3Store(s3_bucket, prefix=os.environ.get("PTK_S3_PREFIX", "ptk-runs"))
    return LocalStore(root or os.environ.get("PTK_REGISTRY_DIR", ".ptk/registry"))
