"""Posttraining Toolkit CLI."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from ptk import __version__
from ptk.config.defaults import scaffold_config
from ptk.config.export_schema import export_json_schema_string
from ptk.config.loader import config_to_yaml, load_config
from ptk.config.schema import ExportFormat, TrainingMethod
from ptk.data.generation import generate_synthetic_data
from ptk.eval.harness import eval_run
from ptk.exceptions import (
    EXIT_RESUME_DRIFT,
    EXIT_RUNTIME,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    PTKError,
    ResumeDriftError,
    ValidationError,
)
from ptk.export.formats import export_run
from ptk.logging import Logger
from ptk.pipeline import plan_run, run_pipeline
from ptk.registry.runs import RunRegistry

app = typer.Typer(
    name="ptk",
    help="Posttraining Toolkit — declarative LLM posttraining pipeline",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def _logger(machine: bool, verbose: bool) -> Logger:
    return Logger(machine=machine, verbose=verbose)


def _handle_error(exc: Exception, logger: Logger, as_json: bool) -> None:
    if isinstance(exc, PTKError):
        if as_json:
            sys.stdout.write(json.dumps({"error": exc.to_dict()}) + "\n")
        else:
            logger.error(exc.message, code=exc.code)
        code = EXIT_VALIDATION if isinstance(exc, ValidationError) else (
            EXIT_RESUME_DRIFT if isinstance(exc, ResumeDriftError) else EXIT_RUNTIME
        )
        raise typer.Exit(code) from exc
    if as_json:
        sys.stdout.write(json.dumps({"error": {"code": "UNEXPECTED", "message": str(exc)}}) + "\n")
    else:
        logger.error(str(exc))
    raise typer.Exit(EXIT_RUNTIME) from exc


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version"),
) -> None:
    if version:
        typer.echo(f"ptk {__version__}")
        raise typer.Exit(EXIT_SUCCESS)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(EXIT_SUCCESS)


@app.command("init")
def init_cmd(
    method: TrainingMethod = typer.Option(TrainingMethod.SFT, "--method", "-m", help="Training method"),
    output: Path = typer.Option(Path("config.yaml"), "--output", "-o", help="Output config path"),
    run_name: str = typer.Option("my-run", "--run-name", help="Run name"),
    base_model: str = typer.Option("distilgpt2", "--base-model", help="Base model ID"),
) -> None:
    """Scaffold a commented starter config."""
    config = scaffold_config(method, run_name=run_name, base_model=base_model)
    yaml_content = (
        f"# Posttraining Toolkit config — method: {method.value}\n"
        f"# Generate schema: ptk schema --json\n\n"
        + config_to_yaml(config)
    )
    output.write_text(yaml_content, encoding="utf-8")
    typer.echo(f"Config written to {output}")


@app.command("validate")
def validate_cmd(
    config_path: Path = typer.Argument(..., help="Path to config YAML/JSON"),
    as_json: bool = typer.Option(False, "--json", help="Structured JSON output"),
    machine: bool = typer.Option(False, "--machine", help="Machine-readable JSONL logs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Validate a config file."""
    logger = _logger(machine, verbose)
    try:
        config = load_config(config_path)
        if as_json or machine:
            sys.stdout.write(
                json.dumps({"valid": True, "run_name": config.run_name, "method": config.method.value}) + "\n"
            )
        else:
            logger.complete("Config valid", run_name=config.run_name, method=config.method.value)
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, logger, as_json or machine)


@app.command("plan")
def plan_cmd(
    config_path: Path = typer.Argument(..., help="Path to config YAML/JSON"),
    as_json: bool = typer.Option(False, "--json", help="Structured JSON output"),
    machine: bool = typer.Option(False, "--machine", help="Machine-readable JSONL logs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Dry-run: show data volume, GPU-hours, disk footprint."""
    logger = _logger(machine, verbose)
    try:
        config = load_config(config_path)
        plan = plan_run(config)
        if as_json or machine:
            sys.stdout.write(json.dumps(plan) + "\n")
        else:
            for key, value in plan.items():
                typer.echo(f"{key}: {value}")
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, logger, as_json or machine)


@app.command("run")
def run_cmd(
    config_path: Path = typer.Argument(..., help="Path to config YAML/JSON"),
    machine: bool = typer.Option(False, "--machine", help="Machine-readable JSONL logs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only, no execution"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    skip_eval: bool = typer.Option(False, "--skip-eval"),
    skip_export: bool = typer.Option(False, "--skip-export"),
) -> None:
    """Execute the full pipeline (data → train → eval → export)."""
    logger = _logger(machine, verbose)
    try:
        config = load_config(config_path)
        run_id = run_pipeline(
            config,
            logger,
            dry_run=dry_run,
            skip_eval=skip_eval,
            skip_export=skip_export,
        )
        if machine:
            logger.complete("run_finished", run_id=run_id)
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, logger, machine)


@app.command("resume")
def resume_cmd(
    run_id: str = typer.Argument(..., help="Run ID to resume"),
    machine: bool = typer.Option(False, "--machine"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    skip_eval: bool = typer.Option(False, "--skip-eval"),
    skip_export: bool = typer.Option(False, "--skip-export"),
) -> None:
    """Resume from last checkpoint in the registry."""
    logger = _logger(machine, verbose)
    try:
        registry = RunRegistry()
        record = registry.get_run(run_id)
        if record is None:
            raise ValidationError(f"Run not found: {run_id}")
        config = load_config_from_dict(record.config)
        checkpoint = registry.find_latest_checkpoint(run_id)
        run_pipeline(
            config,
            logger,
            run_id=run_id,
            resume_from=checkpoint,
            skip_eval=skip_eval,
            skip_export=skip_export,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, logger, machine)


def load_config_from_dict(data: dict):
    from ptk.config.schema import PTKConfig

    return PTKConfig.model_validate(data)


data_app = typer.Typer(help="Data operations")
app.add_typer(data_app, name="data")


@data_app.command("gen")
def data_gen_cmd(
    config_path: Path = typer.Argument(..., help="Path to config YAML/JSON"),
    machine: bool = typer.Option(False, "--machine"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run only synthetic data generation."""
    logger = _logger(machine, verbose)
    try:
        config = load_config(config_path)
        dataset = generate_synthetic_data(config, logger)
        out = config.output_path() / "synthetic_data"
        out.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(out))
        logger.complete("Data saved", path=str(out), rows=len(dataset))
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, logger, machine)


@app.command("eval")
def eval_cmd(
    run_id: str = typer.Argument(..., help="Run ID to evaluate"),
    benchmarks: Optional[str] = typer.Option(None, "--benchmarks", help="Comma-separated benchmark names"),
    machine: bool = typer.Option(False, "--machine"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run eval harness against a completed run."""
    logger = _logger(machine, verbose)
    try:
        bench_list = benchmarks.split(",") if benchmarks else None
        results = eval_run(run_id, logger, benchmarks=bench_list)
        if machine:
            sys.stdout.write(json.dumps(results) + "\n")
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, logger, machine)


@app.command("export")
def export_cmd(
    run_id: str = typer.Argument(..., help="Run ID to export"),
    format: str = typer.Option("adapter_only", "--format", "-f", help="Export format"),
    machine: bool = typer.Option(False, "--machine"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Export a trained run to a target format."""
    logger = _logger(machine, verbose)
    try:
        registry = RunRegistry()
        record = registry.get_run(run_id)
        if record is None:
            raise ValidationError(f"Run not found: {run_id}")
        config = load_config_from_dict(record.config)
        fmt = ExportFormat(format)
        checkpoint = registry.find_latest_checkpoint(run_id)
        artifacts = export_run(config, formats=[fmt], checkpoint_path=checkpoint, logger=logger)
        if machine:
            sys.stdout.write(json.dumps(artifacts) + "\n")
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_error(exc, logger, machine)


@app.command("schema")
def schema_cmd(
    as_json: bool = typer.Option(True, "--json", help="Emit JSON Schema"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write to file"),
) -> None:
    """Emit config JSON Schema for agent consumption."""
    schema_str = export_json_schema_string()
    if output:
        output.write_text(schema_str, encoding="utf-8")
        typer.echo(f"Schema written to {output}")
    else:
        typer.echo(schema_str)


@app.command("list")
def list_cmd(
    as_json: bool = typer.Option(False, "--json"),
    machine: bool = typer.Option(False, "--machine"),
) -> None:
    """List runs in the registry."""
    registry = RunRegistry()
    runs = registry.list_runs()
    if as_json or machine:
        payload = [r.to_dict() for r in runs]
        sys.stdout.write(json.dumps(payload, default=str) + "\n")
    else:
        if not runs:
            typer.echo("No runs found.")
        for r in runs:
            typer.echo(f"{r.run_id}  {r.run_name}  [{r.status.value}]  step={r.global_step}")


if __name__ == "__main__":
    app()
