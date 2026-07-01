"""Config validation tests."""

from pathlib import Path

import pytest

from ptk.config.defaults import scaffold_config
from ptk.config.export_schema import export_json_schema
from ptk.config.loader import load_config
from ptk.config.schema import TrainingMethod
from ptk.exceptions import ValidationError

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_sft_fixture():
    config = load_config(FIXTURES / "sft.yaml")
    assert config.run_name == "test-sft"
    assert config.method == TrainingMethod.SFT
    assert config.base_model == "distilgpt2"


def test_scaffold_all_methods():
    for method in TrainingMethod:
        config = scaffold_config(method)
        assert config.method == method
        assert config.run_name


def test_json_schema_export():
    schema = export_json_schema()
    assert "$defs" in schema or "properties" in schema
    assert "run_name" in schema.get("properties", {})


def test_invalid_config_raises():
    with pytest.raises(ValidationError):
        load_config(FIXTURES / "nonexistent.yaml")


def test_ppo_requires_reward_model():
    from ptk.config.schema import (
        DataConfig,
        DatasetConfig,
        DatasetFormat,
        PTKConfig,
        RLConfig,
        TrainingConfig,
    )

    with pytest.raises(Exception):
        PTKConfig(
            run_name="test",
            base_model="distilgpt2",
            method=TrainingMethod.PPO,
            data=DataConfig(
                dataset=DatasetConfig(path="x", format=DatasetFormat.JSONL),
            ),
            training=TrainingConfig(rl=RLConfig(reward_model=None)),
        )
