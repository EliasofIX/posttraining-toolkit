"""YAML I/O tests."""

from pathlib import Path

from ptk.config.loader import load_config
from ptk.config.yaml_io import dump_yaml, load_yaml

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
