"""Logging tests."""

import json

from ptk.logging import Logger


def test_machine_mode_emits_jsonl(capsys):
    logger = Logger(machine=True)
    logger.info("hello", step=1)
    captured = capsys.readouterr().out.strip()
    record = json.loads(captured)
    assert record["event"] == "info"
    assert record["message"] == "hello"
    assert record["step"] == 1


def test_machine_mode_has_no_ansi(capsys):
    logger = Logger(machine=True)
    logger.error("boom")
    captured = capsys.readouterr().out
    assert "\033[" not in captured
