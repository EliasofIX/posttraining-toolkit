"""HF Hub dataset loader tests."""

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from ptk.config.schema import DatasetConfig, DatasetFormat
from ptk.data.loaders import _load_hf_hub, load_raw_dataset
from ptk.exceptions import ValidationError


def test_load_hf_hub_downloads_from_dataset_url():
    captured: dict = {}

    def fake_urlopen(request, timeout=120):
        captured["url"] = str(request.full_url)
        payload = b'{"text": "hello"}\n'
        response = MagicMock()
        response.read.return_value = payload
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        table = _load_hf_hub("user/demo")

    assert captured["url"] == "https://huggingface.co/datasets/user/demo/resolve/main/train.jsonl"
    assert len(table) == 1
    assert table[0]["text"] == "hello"


def test_load_hf_hub_404_tries_next_candidate():
    calls: list[str] = []

    def fake_urlopen(request, timeout=120):
        calls.append(str(request.full_url))
        if calls[-1].endswith("train.jsonl"):
            raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)
        payload = b'{"text": "from-data"}\n'
        response = MagicMock()
        response.read.return_value = payload
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        table = _load_hf_hub("user/demo")

    assert any(url.endswith("train.jsonl") for url in calls)
    assert any(url.endswith("data.jsonl") for url in calls)
    assert table[0]["text"] == "from-data"


def test_load_hf_hub_auth_error():
    error = urllib.error.HTTPError("url", 401, "unauthorized", {}, None)

    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ValidationError, match="Authentication required"):
            _load_hf_hub("user/private")


def test_load_hf_hub_server_error_does_not_fallback():
    calls: list[str] = []

    def fake_urlopen(request, timeout=120):
        calls.append(str(request.full_url))
        raise urllib.error.HTTPError(request.full_url, 500, "server error", {}, None)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(ValidationError, match="HTTP 500"):
            _load_hf_hub("user/demo")

    assert len(calls) == 1


def test_load_hf_hub_network_error():
    error = urllib.error.URLError("connection reset")

    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ValidationError, match="connection reset"):
            _load_hf_hub("user/demo")


def test_load_hf_hub_all_candidates_missing():
    error = urllib.error.HTTPError("url", 404, "not found", {}, None)

    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ValidationError, match="Could not load dataset"):
            _load_hf_hub("user/empty")


def test_load_raw_dataset_hf_hub_integration():
    config = DatasetConfig(path="user/demo", format=DatasetFormat.HF_HUB)
    payload = b'{"text": "row"}\n'

    def fake_urlopen(request, timeout=120):
        response = MagicMock()
        response.read.return_value = payload
        response.__enter__.return_value = response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        table = load_raw_dataset(config)

    assert len(table) == 1


def test_load_parquet_ragged_columns_raises_validation_error(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from ptk.data.loaders import _load_parquet

    class _FakeTable:
        @staticmethod
        def to_pydict():
            return {"a": [1, 2], "b": [1]}

    monkeypatch.setattr(pq, "read_table", lambda _path: _FakeTable())
    with pytest.raises(ValidationError, match="equal length"):
        _load_parquet(tmp_path / "ignored.parquet")
