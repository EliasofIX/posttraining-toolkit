"""End-to-end integration smoke test (CPU, tiny model)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ptk.cli import app
from ptk.config.loader import load_config
from ptk.pipeline import run_pipeline
from ptk.logging import Logger
from ptk.registry.runs import RunRegistry, RunStatus

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


@pytest.mark.slow
def test_sft_e2e_cpu(tmp_path):
    """Full SFT pipeline on distilgpt2 with 2 steps."""
    config = load_config(FIXTURES / "sft.yaml")
    config.output.dir = str(tmp_path / "outputs")

    logger = Logger(machine=False, verbose=False)
    run_id = run_pipeline(config, logger, skip_eval=False, skip_export=True)

    registry = RunRegistry()
    record = registry.get_run(run_id)
    assert record is not None
    assert record.status == RunStatus.COMPLETED
    assert record.global_step >= 2


@pytest.mark.slow
def test_cli_run_dry_run():
    result = runner.invoke(app, ["run", str(FIXTURES / "sft.yaml"), "--dry-run"])
    assert result.exit_code == 0
