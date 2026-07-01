"""CLI smoke tests."""

from pathlib import Path

from typer.testing import CliRunner

from ptk.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ptk" in result.stdout


def test_cli_validate():
    result = runner.invoke(app, ["validate", str(FIXTURES / "sft.yaml")])
    assert result.exit_code == 0


def test_cli_validate_json():
    result = runner.invoke(app, ["validate", str(FIXTURES / "sft.yaml"), "--json"])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_cli_plan():
    result = runner.invoke(app, ["plan", str(FIXTURES / "sft.yaml"), "--json"])
    assert result.exit_code == 0
    assert "estimated_steps" in result.stdout


def test_cli_schema():
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    assert "run_name" in result.stdout


def test_cli_init(tmp_path):
    out = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "--method", "sft", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_cli_list():
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
