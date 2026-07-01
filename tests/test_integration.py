"""End-to-end integration smoke tests (CPU, tiny model)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ptk.cli import app
from ptk.config.loader import load_config
from ptk.config.schema import ExportFormat
from ptk.export.formats import export_run
from ptk.logging import Logger
from ptk.pipeline import run_pipeline
from ptk.registry.runs import RunRegistry, RunStatus

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def _run_e2e(fixture_name: str, tmp_path: Path, *, min_steps: int = 1) -> str:
    config = load_config(FIXTURES / fixture_name)
    config.output.dir = str(tmp_path / "outputs")
    logger = Logger(machine=False, verbose=False)
    run_id = run_pipeline(config, logger, skip_eval=True, skip_export=True)

    registry = RunRegistry()
    record = registry.get_run(run_id)
    assert record is not None
    assert record.status == RunStatus.COMPLETED
    assert record.global_step >= min_steps
    return run_id


@pytest.mark.slow
def test_sft_e2e_cpu(tmp_path):
    """Full SFT pipeline on distilgpt2 with 2 steps."""
    _run_e2e("sft.yaml", tmp_path, min_steps=2)


@pytest.mark.slow
def test_ppo_e2e_cpu(tmp_path):
    """PPO pipeline smoke test on distilgpt2."""
    _run_e2e("ppo.yaml", tmp_path, min_steps=1)


@pytest.mark.slow
def test_grpo_e2e_cpu(tmp_path):
    """GRPO pipeline smoke test on distilgpt2."""
    _run_e2e("grpo.yaml", tmp_path, min_steps=1)


@pytest.mark.slow
def test_gguf_export_e2e(tmp_path):
    """Train SFT then export to GGUF."""
    config = load_config(FIXTURES / "sft.yaml")
    config.output.dir = str(tmp_path / "outputs")
    config.training.max_iters = 1

    logger = Logger(machine=False, verbose=False)
    run_pipeline(config, logger, skip_eval=True, skip_export=True)

    artifacts = export_run(
        config,
        formats=[ExportFormat.GGUF],
        logger=logger,
    )
    gguf_path = Path(artifacts["gguf"])
    assert gguf_path.exists()
    assert gguf_path.stat().st_size > 100_000


@pytest.mark.slow
def test_cli_run_dry_run():
    result = runner.invoke(app, ["run", str(FIXTURES / "sft.yaml"), "--dry-run"])
    assert result.exit_code == 0
