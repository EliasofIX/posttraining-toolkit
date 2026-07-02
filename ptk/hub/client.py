"""Minimal Hugging Face Hub REST client (no huggingface-hub dependency)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_LFS_HEADERS = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
}
_PREUPLOAD_CHUNK = 256


@dataclass(frozen=True)
class _FileUpload:
    path: str
    abs_path: Path
    size: int
    sha256: bytes
    sample: bytes


class HubClient:
    """Small subset of Hugging Face Hub API used by posttraining-toolkit."""

    def __init__(self, token: str | None = None, endpoint: str = "https://huggingface.co") -> None:
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.endpoint = endpoint.rstrip("/")

    def create_repo(self, repo_id: str, *, private: bool = True, exist_ok: bool = True) -> None:
        organization, name = _parse_repo_id(repo_id)
        payload: dict[str, object] = {"name": name, "private": private, "type": "model"}
        if organization is not None:
            payload["organization"] = organization
        url = f"{self.endpoint}/api/repos/create"
        try:
            self._request_json("POST", url, json.dumps(payload).encode())
        except urllib.error.HTTPError as exc:
            if exist_ok and exc.code in (409, 400):
                return
            raise

    def upload_folder(self, folder_path: str | Path, repo_id: str, *, repo_type: str = "model") -> None:
        folder = Path(folder_path)
        files = [
            _file_upload(path, path.relative_to(folder).as_posix())
            for path in sorted(folder.rglob("*"))
            if path.is_file()
        ]
        if not files:
            return

        upload_modes = self._preupload(files, repo_id=repo_id, repo_type=repo_type)
        commit_ops: list[dict] = []
        lfs_files = [file for file in files if upload_modes.get(file.path) == "lfs"]
        if lfs_files:
            self._upload_lfs_files(lfs_files, repo_id=repo_id, repo_type=repo_type)

        for file in files:
            mode = upload_modes.get(file.path, "regular")
            if mode == "ignored":
                continue
            if mode == "lfs":
                commit_ops.append(
                    {
                        "key": "lfsFile",
                        "value": {
                            "path": file.path,
                            "algo": "sha256",
                            "oid": file.sha256.hex(),
                            "size": file.size,
                        },
                    }
                )
                continue
            commit_ops.append(
                {
                    "key": "file",
                    "value": {
                        "path": file.path,
                        "content": base64.b64encode(file.abs_path.read_bytes()).decode("ascii"),
                        "encoding": "base64",
                    },
                }
            )

        if not commit_ops:
            return

        url = f"{self.endpoint}/api/{_repo_api_segment(repo_type)}/{repo_id}/commit/main"
        body = _build_ndjson_commit(commit_ops, summary="Upload from posttraining-toolkit")
        self._request_ndjson("POST", url, body)

    def download_repo_files(self, repo_id: str, *, filename: str | None = None) -> bytes:
        """Download a single file from a model repo."""
        if filename is None:
            filename = "train.jsonl"
        url = f"{self.endpoint}/{repo_id}/resolve/main/{filename}"
        request = urllib.request.Request(url, headers=self._headers(content_type=None))
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    def _preupload(self, files: list[_FileUpload], *, repo_id: str, repo_type: str) -> dict[str, str]:
        modes: dict[str, str] = {}
        url = f"{self.endpoint}/api/{_repo_api_segment(repo_type)}/{repo_id}/preupload/main"
        for start in range(0, len(files), _PREUPLOAD_CHUNK):
            chunk = files[start : start + _PREUPLOAD_CHUNK]
            payload = {
                "files": [
                    {
                        "path": file.path,
                        "sample": base64.b64encode(file.sample).decode("ascii"),
                        "size": file.size,
                    }
                    for file in chunk
                ]
            }
            response = json.loads(self._request_json("POST", url, json.dumps(payload).encode()))
            for item in response.get("files", []):
                modes[item["path"]] = item.get("uploadMode", "regular")
        return modes

    def _upload_lfs_files(self, files: list[_FileUpload], *, repo_id: str, repo_type: str) -> None:
        batch_url = f"{self.endpoint}/{_repo_lfs_prefix(repo_type)}{repo_id}.git/info/lfs/objects/batch"
        payload = {
            "operation": "upload",
            "transfers": ["basic"],
            "objects": [{"oid": file.sha256.hex(), "size": file.size} for file in files],
            "hash_algo": "sha256",
            "ref": {"name": "main"},
        }
        response = json.loads(
            self._request_json(
                "POST",
                batch_url,
                json.dumps(payload).encode(),
                extra_headers=_LFS_HEADERS,
            )
        )
        oid_to_file = {file.sha256.hex(): file for file in files}
        for item in response.get("objects", []):
            if "error" in item:
                message = item["error"].get("message", "unknown LFS error")
                raise RuntimeError(f"LFS batch failed for {item.get('oid')}: {message}")
            actions = item.get("actions") or {}
            upload_action = actions.get("upload")
            if upload_action is None:
                continue
            file = oid_to_file[item["oid"]]
            upload_url = upload_action["href"]
            upload_headers = upload_action.get("header") or {}
            request = urllib.request.Request(
                upload_url,
                data=file.abs_path.read_bytes(),
                headers={**upload_headers, **self._headers(content_type=None)},
                method=upload_action.get("method", "PUT"),
            )
            with urllib.request.urlopen(request, timeout=300) as http_response:
                http_response.read()
            verify_action = actions.get("verify")
            if verify_action is not None:
                verify_payload = json.dumps({"oid": file.sha256.hex(), "size": file.size}).encode()
                verify_request = urllib.request.Request(
                    verify_action["href"],
                    data=verify_payload,
                    headers=self._headers(),
                    method=verify_action.get("method", "POST"),
                )
                with urllib.request.urlopen(verify_request, timeout=120) as verify_response:
                    verify_response.read()

    def _headers(self, *, content_type: str | None = "application/json") -> dict[str, str]:
        headers = {"User-Agent": "posttraining-toolkit"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request_json(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> bytes:
        headers = self._headers()
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    def _request_ndjson(self, method: str, url: str, body: bytes) -> bytes:
        headers = self._headers(content_type="application/x-ndjson")
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()


def _parse_repo_id(repo_id: str) -> tuple[str | None, str]:
    if "/" not in repo_id:
        return None, repo_id
    organization, name = repo_id.split("/", 1)
    return organization, name


def _repo_api_segment(repo_type: str) -> str:
    if repo_type == "dataset":
        return "datasets"
    if repo_type == "space":
        return "spaces"
    return "models"


def _repo_lfs_prefix(repo_type: str) -> str:
    if repo_type == "dataset":
        return "datasets/"
    if repo_type == "space":
        return "spaces/"
    return ""


def _file_upload(abs_path: Path, rel_path: str) -> _FileUpload:
    data = abs_path.read_bytes()
    return _FileUpload(
        path=rel_path,
        abs_path=abs_path,
        size=len(data),
        sha256=hashlib.sha256(data).digest(),
        sample=data[:512],
    )


def _build_ndjson_commit(operations: list[dict], *, summary: str) -> bytes:
    lines = [{"key": "header", "value": {"summary": summary, "description": ""}}]
    lines.extend(operations)
    return b"".join(json.dumps(item).encode() + b"\n" for item in lines)
