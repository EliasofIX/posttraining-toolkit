"""Hub client tests."""

import hashlib
import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from ptk.hub.client import (
    HubClient,
    _build_ndjson_commit,
    _is_repo_already_exists_error,
    _parse_repo_id,
)


def test_parse_repo_id_with_namespace():
    assert _parse_repo_id("user/model") == ("user", "model")


def test_parse_repo_id_without_namespace():
    assert _parse_repo_id("model") == (None, "model")


def test_create_repo_tolerates_existing_repo():
    client = HubClient(token="test-token")
    error = urllib.error.HTTPError("url", 409, "conflict", {}, None)
    with patch("urllib.request.urlopen", side_effect=error):
        client.create_repo("user/model", exist_ok=True)


def test_create_repo_tolerates_already_exists_400():
    client = HubClient(token="test-token")
    body = io.BytesIO(b'{"error":"You already created this model repo"}')
    error = urllib.error.HTTPError("url", 400, "bad request", {}, body)
    with patch("urllib.request.urlopen", side_effect=error):
        client.create_repo("user/model", exist_ok=True)


def test_create_repo_propagates_unrelated_400():
    client = HubClient(token="test-token")
    body = io.BytesIO(b'{"error":"Invalid organization name"}')
    error = urllib.error.HTTPError("url", 400, "bad request", {}, body)
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(urllib.error.HTTPError):
            client.create_repo("user/model", exist_ok=True)


def test_is_repo_already_exists_error_matches_message():
    body = io.BytesIO(b'{"error":"You already created this model repo"}')
    exc = urllib.error.HTTPError("url", 400, "bad request", {}, body)
    assert _is_repo_already_exists_error(exc) is True


def test_create_repo_sends_namespace_and_name():
    client = HubClient(token="test-token")
    captured: dict = {}

    def fake_urlopen(request, timeout=120):
        captured["body"] = json.loads(request.data.decode())
        response = MagicMock()
        response.read.return_value = b"{}"
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.create_repo("user/model", private=True)

    assert captured["body"] == {
        "name": "model",
        "organization": "user",
        "private": True,
        "type": "model",
    }


def test_download_repo_files_uses_dataset_prefix():
    client = HubClient(token="token")
    captured: dict = {}

    def fake_urlopen(request, timeout=120):
        captured["url"] = str(request.full_url)
        response = MagicMock()
        response.read.return_value = b"{}"
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.download_repo_files("user/data", filename="train.jsonl", repo_type="dataset")

    assert captured["url"] == "https://huggingface.co/datasets/user/data/resolve/main/train.jsonl"


def test_upload_folder_builds_ndjson_commit(tmp_path):
    file_path = tmp_path / "config.json"
    file_path.write_text('{"a": 1}', encoding="utf-8")
    client = HubClient(token="token")

    captured: dict = {}

    def fake_urlopen(request, timeout=120):
        if request.data is not None:
            captured["content_type"] = request.headers.get("Content-type") or request.headers.get(
                "Content-Type"
            )
            captured["lines"] = [json.loads(line) for line in request.data.decode().splitlines() if line]
        response = MagicMock()
        response.read.return_value = b"{}"
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.upload_folder(tmp_path, "user/model")

    assert captured["content_type"] == "application/x-ndjson"
    assert captured["lines"][0]["key"] == "header"
    assert captured["lines"][0]["value"]["summary"] == "Upload from posttraining-toolkit"
    assert captured["lines"][1]["value"]["path"] == "config.json"


def test_build_ndjson_commit_includes_header():
    body = _build_ndjson_commit(
        [{"key": "file", "value": {"path": "a.txt", "content": "aGk=", "encoding": "base64"}}],
        summary="test commit",
    )
    lines = [json.loads(line) for line in body.decode().splitlines()]
    assert lines[0]["value"]["summary"] == "test commit"
    assert lines[1]["value"]["path"] == "a.txt"


def test_upload_folder_uses_lfs_for_large_files(tmp_path):
    large_path = tmp_path / "model.bin"
    large_path.write_bytes(b"x" * 11)
    client = HubClient(token="token")

    captured: dict = {}

    expected_oid = hashlib.sha256(large_path.read_bytes()).hexdigest()

    def fake_urlopen_with_lfs(request, timeout=120):
        url = str(request.full_url)
        if "preupload" in url:
            response = MagicMock()
            response.read.return_value = json.dumps(
                {"files": [{"path": "model.bin", "uploadMode": "lfs"}]}
            ).encode()
            response.__enter__.return_value = response
            return response
        if "info/lfs/objects/batch" in url:
            response = MagicMock()
            response.read.return_value = json.dumps(
                {
                    "objects": [
                        {
                            "oid": expected_oid,
                            "size": 11,
                            "actions": None,
                        }
                    ]
                }
            ).encode()
            response.__enter__.return_value = response
            return response
        if request.data is not None:
            captured["lines"] = [json.loads(line) for line in request.data.decode().splitlines() if line]
        response = MagicMock()
        response.read.return_value = b"{}"
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen_with_lfs):
        client.upload_folder(tmp_path, "user/model")

    assert captured["lines"][1]["key"] == "lfsFile"
    assert captured["lines"][1]["value"]["path"] == "model.bin"
    assert captured["lines"][1]["value"]["oid"] == expected_oid


def test_upload_lfs_verify_uses_lfs_headers(tmp_path):
    large_path = tmp_path / "model.bin"
    large_path.write_bytes(b"x" * 11)
    client = HubClient(token="token")
    expected_oid = hashlib.sha256(large_path.read_bytes()).hexdigest()
    captured: dict = {}

    def fake_urlopen(request, timeout=120):
        url = str(request.full_url)
        if "preupload" in url:
            response = MagicMock()
            response.read.return_value = json.dumps(
                {"files": [{"path": "model.bin", "uploadMode": "lfs"}]}
            ).encode()
            response.__enter__.return_value = response
            return response
        if "info/lfs/objects/batch" in url:
            response = MagicMock()
            response.read.return_value = json.dumps(
                {
                    "objects": [
                        {
                            "oid": expected_oid,
                            "size": 11,
                            "actions": {
                                "upload": {
                                    "href": "https://upload.example/put",
                                    "header": {"Content-Type": "application/octet-stream"},
                                    "method": "PUT",
                                },
                                "verify": {
                                    "href": "https://verify.example/check",
                                    "header": {"X-Test": "1"},
                                    "method": "POST",
                                },
                            },
                        }
                    ]
                }
            ).encode()
            response.__enter__.return_value = response
            return response
        if "upload.example" in url:
            response = MagicMock()
            response.read.return_value = b""
            response.__enter__.return_value = response
            return response
        if "verify.example" in url:
            captured["verify_headers"] = dict(request.headers)
            captured["verify_body"] = json.loads(request.data.decode())
            response = MagicMock()
            response.read.return_value = b""
            response.__enter__.return_value = response
            return response
        response = MagicMock()
        response.read.return_value = b"{}"
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.upload_folder(tmp_path, "user/model")

    assert captured["verify_body"] == {"oid": expected_oid, "size": 11}
    assert captured["verify_headers"]["Content-type"] == "application/vnd.git-lfs+json"
    assert captured["verify_headers"]["X-test"] == "1"


def test_upload_lfs_multipart_uploads_parts_and_completes(tmp_path):
    large_path = tmp_path / "model.bin"
    large_path.write_bytes(b"abcdefghij")
    client = HubClient(token="token")
    expected_oid = hashlib.sha256(large_path.read_bytes()).hexdigest()
    captured: dict = {"parts": []}

    def fake_urlopen(request, timeout=120):
        url = str(request.full_url)
        if "preupload" in url:
            response = MagicMock()
            response.read.return_value = json.dumps(
                {"files": [{"path": "model.bin", "uploadMode": "lfs"}]}
            ).encode()
            response.__enter__.return_value = response
            return response
        if "info/lfs/objects/batch" in url:
            response = MagicMock()
            response.read.return_value = json.dumps(
                {
                    "objects": [
                        {
                            "oid": expected_oid,
                            "size": 10,
                            "actions": {
                                "upload": {
                                    "href": "https://upload.example/complete",
                                    "header": {
                                        "chunk_size": "4",
                                        "1": "https://upload.example/part/1",
                                        "2": "https://upload.example/part/2",
                                        "3": "https://upload.example/part/3",
                                    },
                                    "method": "POST",
                                }
                            },
                        }
                    ]
                }
            ).encode()
            response.__enter__.return_value = response
            return response
        if "upload.example/part" in url:
            captured["parts"].append(request.data)
            response = MagicMock()
            response.read.return_value = b""
            response.headers = {"etag": f'"part-{len(captured["parts"])}"'}
            response.__enter__.return_value = response
            return response
        if "upload.example/complete" in url:
            captured["completion"] = json.loads(request.data.decode())
            captured["completion_headers"] = dict(request.headers)
            response = MagicMock()
            response.read.return_value = b""
            response.__enter__.return_value = response
            return response
        response = MagicMock()
        response.read.return_value = b"{}"
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.upload_folder(tmp_path, "user/model")

    assert captured["parts"] == [b"abcd", b"efgh", b"ij"]
    assert captured["completion"] == {
        "oid": expected_oid,
        "parts": [
            {"partNumber": 1, "etag": "part-1"},
            {"partNumber": 2, "etag": "part-2"},
            {"partNumber": 3, "etag": "part-3"},
        ],
    }
    assert captured["completion_headers"]["Content-type"] == "application/vnd.git-lfs+json"


def test_upload_lfs_unknown_oid_raises(tmp_path):
    large_path = tmp_path / "model.bin"
    large_path.write_bytes(b"x" * 11)
    client = HubClient(token="token")

    def fake_urlopen(request, timeout=120):
        url = str(request.full_url)
        if "preupload" in url:
            response = MagicMock()
            response.read.return_value = json.dumps(
                {"files": [{"path": "model.bin", "uploadMode": "lfs"}]}
            ).encode()
            response.__enter__.return_value = response
            return response
        if "info/lfs/objects/batch" in url:
            response = MagicMock()
            response.read.return_value = json.dumps(
                {
                    "objects": [
                        {
                            "oid": "deadbeef",
                            "size": 11,
                            "actions": {
                                "upload": {"href": "https://upload.example/put", "header": {}}
                            },
                        }
                    ]
                }
            ).encode()
            response.__enter__.return_value = response
            return response
        response = MagicMock()
        response.read.return_value = b"{}"
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(RuntimeError, match="unknown oid"):
            client.upload_folder(tmp_path, "user/model")
