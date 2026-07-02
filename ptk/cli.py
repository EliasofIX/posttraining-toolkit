"""Posttraining Toolkit CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ptk import __version__
from ptk.config.defaults import scaffold_config
from ptk.config.export_schema import export_json_schema_string
from ptk.config.loader import config_to_yaml, load_config
from ptk.config.schema import ExportFormat, PTKConfig, TrainingMethod
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


class CLIExit(SystemExit):
    """CLI exit with explicit status code."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


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
        raise CLIExit(code) from exc
    if as_json:
        sys.stdout.write(json.dumps({"error": {"code": "UNEXPECTED", "message": str(exc)}}) + "\n")
    else:
        logger.error(str(exc))
    raise CLIExit(EXIT_RUNTIME) from exc


def load_config_from_dict(data: dict) -> PTKConfig:
    return PTKConfig.model_validate(data)


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--machine", action="store_true", help="Machine-readable JSONL logs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptk",
        description="Posttraining Toolkit — declarative LLM posttraining pipeline",
    )
    parser.add_argument("--version", "-V", action="store_true", help="Show version")

    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Scaffold a starter config")
    init_parser.add_argument(
        "--method",
        "-m",
        choices=[m.value for m in TrainingMethod],
        default=TrainingMethod.SFT.value,
    )
    init_parser.add_argument("--output", "-o", type=Path, default=Path("config.yaml"))
    init_parser.add_argument("--run-name", default="my-run")
    init_parser.add_argument("--base-model", default="distilgpt2")

    validate_parser = subparsers.add_parser("validate", help="Validate a config file")
    validate_parser.add_argument("config_path", type=Path)
    validate_parser.add_argument("--json", action="store_true", help="Structured JSON output")
    _add_common_flags(validate_parser)

    plan_parser = subparsers.add_parser("plan", help="Dry-run resource estimates")
    plan_parser.add_argument("config_path", type=Path)
    plan_parser.add_argument("--json", action="store_true")
    _add_common_flags(plan_parser)

    run_parser = subparsers.add_parser("run", help="Execute the full pipeline")
    run_parser.add_argument("config_path", type=Path)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--skip-eval", action="store_true")
    run_parser.add_argument("--skip-export", action="store_true")
    _add_common_flags(run_parser)

    resume_parser = subparsers.add_parser("resume", help="Resume from last checkpoint")
    resume_parser.add_argument("run_id")
    resume_parser.add_argument("--skip-eval", action="store_true")
    resume_parser.add_argument("--skip-export", action="store_true")
    _add_common_flags(resume_parser)

    data_parser = subparsers.add_parser("data", help="Data operations")
    data_sub = data_parser.add_subparsers(dest="data_command")
    data_gen = data_sub.add_parser("gen", help="Run synthetic data generation")
    data_gen.add_argument("config_path", type=Path)
    _add_common_flags(data_gen)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation harness")
    eval_parser.add_argument("run_id")
    eval_parser.add_argument("--benchmarks", default=None)
    _add_common_flags(eval_parser)

    export_parser = subparsers.add_parser("export", help="Export a trained run")
    export_parser.add_argument("run_id")
    export_parser.add_argument("--format", "-f", default="adapter_only")
    _add_common_flags(export_parser)

    schema_parser = subparsers.add_parser("schema", help="Emit config JSON Schema")
    schema_parser.add_argument("--json", action="store_true", default=True)
    schema_parser.add_argument("--output", "-o", type=Path, default=None)

    list_parser = subparsers.add_parser("list", help="List runs in the registry")
    list_parser.add_argument("--json", action="store_true")
    list_parser.add_argument("--machine", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        print(f"ptk {__version__}")
        raise CLIExit(EXIT_SUCCESS)

    if not args.command:
        parser.print_help()
        raise CLIExit(EXIT_SUCCESS)

    try:
        if args.command == "init":
            _cmd_init(args)
        elif args.command == "validate":
            _cmd_validate(args)
        elif args.command == "plan":
            _cmd_plan(args)
        elif args.command == "run":
            _cmd_run(args)
        elif args.command == "resume":
            _cmd_resume(args)
        elif args.command == "data":
            if args.data_command != "gen":
                parser.parse_args(["data", "--help"])
            _cmd_data_gen(args)
        elif args.command == "eval":
            _cmd_eval(args)
        elif args.command == "export":
            _cmd_export(args)
        elif args.command == "schema":
            _cmd_schema(args)
        elif args.command == "list":
            _cmd_list(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except CLIExit:
        raise
    except Exception as exc:
        machine = getattr(args, "machine", False)
        as_json = getattr(args, "json", False) or machine
        logger = _logger(machine, getattr(args, "verbose", False))
        _handle_error(exc, logger, as_json)


def _cmd_init(args: argparse.Namespace) -> None:
    method = TrainingMethod(args.method)
    config = scaffold_config(method, run_name=args.run_name, base_model=args.base_model)
    yaml_content = (
        f"# Posttraining Toolkit config — method: {method.value}\n"
        f"# Generate schema: ptk schema --json\n\n"
        + config_to_yaml(config)
    )
    args.output.write_text(yaml_content, encoding="utf-8")
    print(f"Config written to {args.output}")


def _cmd_validate(args: argparse.Namespace) -> None:
    logger = _logger(args.machine, args.verbose)
    config = load_config(args.config_path)
    if args.json or args.machine:
        sys.stdout.write(
            json.dumps({"valid": True, "run_name": config.run_name, "method": config.method.value}) + "\n"
        )
    else:
        logger.complete("Config valid", run_name=config.run_name, method=config.method.value)


def _cmd_plan(args: argparse.Namespace) -> None:
    config = load_config(args.config_path)
    plan = plan_run(config)
    if args.json or args.machine:
        sys.stdout.write(json.dumps(plan) + "\n")
    else:
        for key, value in plan.items():
            print(f"{key}: {value}")


def _cmd_run(args: argparse.Namespace) -> None:
    logger = _logger(args.machine, args.verbose)
    config = load_config(args.config_path)
    run_id = run_pipeline(
        config,
        logger,
        dry_run=args.dry_run,
        skip_eval=args.skip_eval,
        skip_export=args.skip_export,
    )
    if args.machine:
        logger.complete("run_finished", run_id=run_id)


def _cmd_resume(args: argparse.Namespace) -> None:
    logger = _logger(args.machine, args.verbose)
    registry = RunRegistry()
    record = registry.get_run(args.run_id)
    if record is None:
        raise ValidationError(f"Run not found: {args.run_id}")
    config = load_config_from_dict(record.config)
    checkpoint = registry.find_latest_checkpoint(args.run_id)
    run_pipeline(
        config,
        logger,
        run_id=args.run_id,
        resume_from=checkpoint,
        skip_eval=args.skip_eval,
        skip_export=args.skip_export,
    )


def _cmd_data_gen(args: argparse.Namespace) -> None:
    logger = _logger(args.machine, args.verbose)
    config = load_config(args.config_path)
    dataset = generate_synthetic_data(config, logger)
    out = config.output_path() / "synthetic_data"
    out.mkdir(parents=True, exist_ok=True)
    dataset.save_jsonl(out)
    logger.complete("Data saved", path=str(out), rows=len(dataset))


def _cmd_eval(args: argparse.Namespace) -> None:
    logger = _logger(args.machine, args.verbose)
    bench_list = args.benchmarks.split(",") if args.benchmarks else None
    results = eval_run(args.run_id, logger, benchmarks=bench_list)
    if args.machine:
        sys.stdout.write(json.dumps(results) + "\n")


def _cmd_export(args: argparse.Namespace) -> None:
    logger = _logger(args.machine, args.verbose)
    registry = RunRegistry()
    record = registry.get_run(args.run_id)
    if record is None:
        raise ValidationError(f"Run not found: {args.run_id}")
    config = load_config_from_dict(record.config)
    fmt = ExportFormat(args.format)
    checkpoint = registry.find_latest_checkpoint(args.run_id)
    artifacts = export_run(config, formats=[fmt], checkpoint_path=checkpoint, logger=logger)
    if args.machine:
        sys.stdout.write(json.dumps(artifacts) + "\n")


def _cmd_schema(args: argparse.Namespace) -> None:
    schema_str = export_json_schema_string()
    if args.output:
        args.output.write_text(schema_str, encoding="utf-8")
        print(f"Schema written to {args.output}")
    else:
        print(schema_str)


def _cmd_list(args: argparse.Namespace) -> None:
    registry = RunRegistry()
    runs = registry.list_runs()
    if args.json or args.machine:
        payload = [r.to_dict() for r in runs]
        sys.stdout.write(json.dumps(payload, default=str) + "\n")
    else:
        if not runs:
            print("No runs found.")
        for r in runs:
            print(f"{r.run_id}  {r.run_name}  [{r.status.value}]  step={r.global_step}")


if __name__ == "__main__":
    try:
        main()
    except CLIExit as exc:
        raise SystemExit(exc.code) from exc
