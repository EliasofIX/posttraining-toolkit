"""Minimal Hugging Face Hub REST client (no huggingface-hub dependency)."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


class HubClient:
    """Small subset of Hugging Face Hub API used by posttraining-toolkit."""

    def __init__(self, token: str | None = None, endpoint: str = "https://huggingface.co") -> None:
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.endpoint = endpoint.rstrip("/")

    def create_repo(self, repo_id: str, *, private: bool = True, exist_ok: bool = True) -> None:
        url = f"{self.endpoint}/api/repos/create"
        body = json.dumps({"name": repo_id, "private": private, "type": "model"}).encode()
        try:
            self._request("POST", url, body)
        except urllib.error.HTTPError as exc:
            if exist_ok and exc.code in (409, 400):
                return
            raise

    def upload_folder(self, folder_path: str | Path, repo_id: str, *, repo_type: str = "model") -> None:
        folder = Path(folder_path)
        operations: list[dict] = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(folder).as_posix()
            operations.append(
                {
                    "key": "file",
                    "value": {
                        "path": rel,
                        "content": base64.b64encode(path.read_bytes()).decode("ascii"),
                        "encoding": "base64",
                    },
                }
            )

        if not operations:
            return

        url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/commit/main"
        body = json.dumps(
            {
                "operations": operations,
                "commit_message": "Upload from posttraining-toolkit",
            }
        ).encode()
        self._request("POST", url, body)

    def download_repo_files(self, repo_id: str, *, filename: str | None = None) -> bytes:
        """Download a single file from a model repo."""
        if filename is None:
            filename = "train.jsonl"
        url = f"{self.endpoint}/{repo_id}/resolve/main/{filename}"
        request = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "posttraining-toolkit"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(self, method: str, url: str, body: bytes | None = None) -> bytes:
        request = urllib.request.Request(url, data=body, headers=self._headers(), method=method)
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
