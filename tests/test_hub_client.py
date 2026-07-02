"""Hub client tests."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

from ptk.hub.client import HubClient


def test_create_repo_tolerates_existing_repo():
    client = HubClient(token="test-token")
    error = urllib.error.HTTPError("url", 409, "conflict", {}, None)
    with patch("urllib.request.urlopen", side_effect=error):
        client.create_repo("user/model", exist_ok=True)


def test_upload_folder_builds_commit_payload(tmp_path):
    file_path = tmp_path / "config.json"
    file_path.write_text('{"a": 1}', encoding="utf-8")
    client = HubClient(token="token")

    captured: dict = {}

    def fake_urlopen(request, timeout=120):
        captured["body"] = json.loads(request.data.decode())
        response = MagicMock()
        response.read.return_value = b"{}"
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.upload_folder(tmp_path, "user/model")

    assert captured["body"]["operations"][0]["value"]["path"] == "config.json"
