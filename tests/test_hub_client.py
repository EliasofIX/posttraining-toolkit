"""Hub client tests."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

from ptk.hub.client import HubClient, _build_ndjson_commit, _parse_repo_id


def test_parse_repo_id_with_namespace():
    assert _parse_repo_id("user/model") == ("user", "model")


def test_parse_repo_id_without_namespace():
    assert _parse_repo_id("model") == (None, "model")


def test_create_repo_tolerates_existing_repo():
    client = HubClient(token="test-token")
    error = urllib.error.HTTPError("url", 409, "conflict", {}, None)
    with patch("urllib.request.urlopen", side_effect=error):
        client.create_repo("user/model", exist_ok=True)


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

    import hashlib

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
