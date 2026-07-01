"""Render distributed launch scripts from Jinja templates."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ptk.config.schema import PTKConfig
from ptk.distributed.detect import ComputeEnvironment

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_torchrun_script(config: PTKConfig, output_path: Path) -> Path:
    """Render torchrun launch script."""
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=False)
    template = env.get_template("torchrun.sh.j2")
    content = template.render(run_name=config.run_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    output_path.chmod(0o755)
    return output_path


def render_slurm_script(
    config: PTKConfig,
    env: ComputeEnvironment,
    output_path: Path,
    *,
    master_port: int = 29500,
) -> Path:
    """Render SLURM sbatch script for multi-node training."""
    jinja = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=False)
    template = jinja.get_template("slurm.sbatch.j2")
    content = template.render(
        run_name=config.run_name,
        nodes=env.num_nodes,
        gpus_per_node=env.num_gpus,
        config_path=str(output_path.parent / "config.yaml"),
        output_dir=str(config.output_path()),
        master_port=master_port,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    output_path.chmod(0o755)
    return output_path
