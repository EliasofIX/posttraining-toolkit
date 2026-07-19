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


def test_grpo_rejects_incompatible_batch_size():
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
            run_name="test-grpo-bad",
            base_model="distilgpt2",
            method=TrainingMethod.GRPO,
            data=DataConfig(
                dataset=DatasetConfig(path="x", format=DatasetFormat.JSONL),
            ),
            training=TrainingConfig(
                batch_size=1,
                gradient_accumulation_steps=1,
                rl=RLConfig(num_generations=2),
            ),
        )


def test_grpo_accepts_compatible_batch_size():
    from ptk.config.schema import (
        DataConfig,
        DatasetConfig,
        DatasetFormat,
        PTKConfig,
        RLConfig,
        TrainingConfig,
    )

    config = PTKConfig(
        run_name="test-grpo-ok",
        base_model="distilgpt2",
        method=TrainingMethod.GRPO,
        data=DataConfig(
            dataset=DatasetConfig(path="x", format=DatasetFormat.JSONL),
        ),
        training=TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            rl=RLConfig(num_generations=2),
        ),
    )
    assert config.training.rl is not None
    assert config.training.rl.num_generations == 2


def test_grpo_allows_single_batch_on_multi_gpu_strategy():
    """TRL multiplies by world_size; schema must not reject multi_gpu 1×2 configs."""
    from ptk.config.schema import (
        ComputeConfig,
        ComputeStrategy,
        DataConfig,
        DatasetConfig,
        DatasetFormat,
        PTKConfig,
        RLConfig,
        TrainingConfig,
    )

    config = PTKConfig(
        run_name="test-grpo-multi",
        base_model="distilgpt2",
        method=TrainingMethod.GRPO,
        data=DataConfig(
            dataset=DatasetConfig(path="x", format=DatasetFormat.JSONL),
        ),
        training=TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=1,
            rl=RLConfig(num_generations=2),
        ),
        compute=ComputeConfig(strategy=ComputeStrategy.MULTI_GPU),
    )
    assert config.compute.strategy == ComputeStrategy.MULTI_GPU


def test_load_config_checks_dataset_path(tmp_path):
    from ptk.config.loader import load_config

    cfg = tmp_path / "missing-data.yaml"
    cfg.write_text(
        """
run_name: missing-data
base_model: distilgpt2
method: sft
data:
  source: dataset
  dataset:
    path: /nonexistent/train.jsonl
    format: jsonl
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="Dataset path not found"):
        load_config(cfg)

    config = load_config(cfg, check_dataset_path=False)
    assert config.run_name == "missing-data"


def test_load_config_resolves_path_relative_to_config_dir(tmp_path):
    from ptk.config.loader import resolve_dataset_path

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "train.jsonl"
    data_file.write_text('{"text": "hello"}\n', encoding="utf-8")
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        """
run_name: rel-path
base_model: distilgpt2
method: sft
data:
  source: dataset
  dataset:
    path: data/train.jsonl
    format: jsonl
""",
        encoding="utf-8",
    )
    config = load_config(cfg)
    # Path stays relative for portable dumps; config_dir enables resolution.
    assert config.data.dataset.path == "data/train.jsonl"
    assert config.config_dir == tmp_path.resolve()
    assert resolve_dataset_path(config.data.dataset.path, config_dir=config.config_dir) == data_file.resolve()


def test_load_config_does_not_bind_cwd_dataset(tmp_path, monkeypatch):
    """Missing config-dir data must not silently use an unrelated cwd dataset."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "data").mkdir()
    (cwd / "data" / "train.jsonl").write_text('{"text": "cwd"}\n', encoding="utf-8")

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg = cfg_dir / "run.yaml"
    cfg.write_text(
        """
run_name: no-cwd-bind
base_model: distilgpt2
method: sft
data:
  source: dataset
  dataset:
    path: data/train.jsonl
    format: jsonl
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(cwd)
    with pytest.raises(ValidationError, match="Dataset path not found"):
        load_config(cfg)


def test_load_config_rejects_directory_dataset_path(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        f"""
run_name: dir-path
base_model: distilgpt2
method: sft
data:
  source: dataset
  dataset:
    path: {data_dir.name}
    format: jsonl
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="directory"):
        load_config(cfg)


def test_registry_keeps_relative_dataset_path(tmp_path):
    from ptk.registry.runs import RunRegistry
    from ptk.registry.store import LocalStore

    cfg = FIXTURES / "sft.yaml"
    config = load_config(cfg)
    assert not Path(config.data.dataset.path).is_absolute()

    store = LocalStore(tmp_path / "registry")
    registry = RunRegistry(store=store)
    record = registry.create_run(config)
    assert not Path(record.config["data"]["dataset"]["path"]).is_absolute()
    assert record.config_dir == str(cfg.parent.resolve())


def test_load_all_method_fixtures():
    from ptk.config.loader import resolve_dataset_path

    for name in ("sft.yaml", "lora.yaml", "qlora.yaml", "dpo.yaml", "ppo.yaml", "grpo.yaml"):
        config = load_config(FIXTURES / name)
        assert config.method.value in name
        resolved = resolve_dataset_path(config.data.dataset.path, config_dir=config.config_dir)
        assert resolved.exists()
