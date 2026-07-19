"""YAML I/O tests."""

import json
from pathlib import Path

import pytest

from ptk.cli import CLIExit, main
from ptk.config.loader import load_config
from ptk.config.yaml_io import dump_yaml, load_yaml
from ptk.data.table import Table
from ptk.exceptions import ValidationError

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_fixture_configs():
    for path in FIXTURES.glob("*.yaml"):
        config = load_config(path)
        assert config.run_name


def test_yaml_round_trip():
    original = load_yaml((FIXTURES / "sft.yaml").read_text(encoding="utf-8"))
    dumped = dump_yaml(original)
    reloaded = load_yaml(dumped)
    assert reloaded == original


def test_yaml_empty_list_round_trip():
    dumped = dump_yaml({"eval": {"benchmarks": [], "eval_steps": 100}})
    assert "benchmarks: []" in dumped
    reloaded = load_yaml(dumped)
    assert reloaded["eval"]["benchmarks"] == []


def test_load_config_rejects_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("run_name: test\nbad indent\n  x: 1", encoding="utf-8")
    with pytest.raises(ValidationError, match="Expected key"):
        load_config(path)


def test_cli_validate_invalid_yaml_exits_validation(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("run_name: test\nbad indent\n  x: 1", encoding="utf-8")
    stdout: list[str] = []

    class _Writer:
        def write(self, text: str) -> int:
            stdout.append(text)
            return len(text)

        def flush(self) -> None:
            return None

    import sys

    original_stdout = sys.stdout
    sys.stdout = _Writer()  # type: ignore[assignment]
    try:
        with pytest.raises(CLIExit) as exc:
            main(["validate", str(path), "--json"])
        assert exc.value.code == 1
    finally:
        sys.stdout = original_stdout

    payload = json.loads("".join(stdout))
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_yaml_rejects_tab_indentation():
    with pytest.raises(ValueError, match="Tabs are not supported"):
        load_yaml("parent:\n\tchild: 1")


def test_table_single_row_split_keeps_train_data():
    table = Table.from_records([{"text": "only row"}])
    split = table.train_test_split(test_size=0.1, seed=42)
    assert len(split["train"]) == 1
    assert len(split["test"]) == 0


def test_table_from_dict_rejects_ragged_columns():
    with pytest.raises(ValueError, match="equal length"):
        Table.from_dict({"a": [1, 2], "b": [1]})
