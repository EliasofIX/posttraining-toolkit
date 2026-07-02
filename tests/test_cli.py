"""CLI smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

from ptk.cli import CLIExit, main

FIXTURES = Path(__file__).parent / "fixtures"


def invoke(args: list[str]) -> tuple[int, str]:
    stdout = []
    original_stdout = sys.stdout

    class _Writer:
        def write(self, text: str) -> int:
            stdout.append(text)
            return len(text)

        def flush(self) -> None:
            return None

    sys.stdout = _Writer()  # type: ignore[assignment]
    try:
        try:
            main(args)
            code = 0
        except CLIExit as exc:
            code = exc.code
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout = original_stdout
    return code, "".join(stdout)


def test_cli_version():
    code, output = invoke(["--version"])
    assert code == 0
    assert "ptk" in output


def test_cli_validate():
    code, _ = invoke(["validate", str(FIXTURES / "sft.yaml")])
    assert code == 0


def test_cli_validate_json():
    code, output = invoke(["validate", str(FIXTURES / "sft.yaml"), "--json"])
    assert code == 0
    assert "valid" in output


def test_cli_plan():
    code, output = invoke(["plan", str(FIXTURES / "sft.yaml"), "--json"])
    assert code == 0
    assert "estimated_steps" in output


def test_cli_schema():
    code, output = invoke(["schema"])
    assert code == 0
    assert "run_name" in output


def test_cli_init(tmp_path):
    out = tmp_path / "config.yaml"
    code, _ = invoke(["init", "--method", "sft", "--output", str(out)])
    assert code == 0
    assert out.exists()


def test_cli_list():
    code, output = invoke(["list", "--json"])
    assert code == 0
    assert output.startswith("[")
