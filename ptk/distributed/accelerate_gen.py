"""Generate Accelerate configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ptk.config.schema import ComputeStrategy
from ptk.distributed.detect import ComputeEnvironment


def generate_accelerate_config(
    env: ComputeEnvironment,
    output_path: Path,
    *,
    deepspeed_config: Path | None = None,
) -> Path:
    """Write an accelerate config YAML for the detected environment."""
    if deepspeed_config is not None:
        config = _deepspeed_config(env, deepspeed_config)
    elif env.strategy == ComputeStrategy.SINGLE_GPU:
        config = _single_gpu_config(env)
    elif env.strategy == ComputeStrategy.MULTI_GPU:
        config = _multi_gpu_config(env)
    else:
        config = _multi_node_config(env)

    lines = _dict_to_yaml_lines(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _deepspeed_config(env: ComputeEnvironment, deepspeed_path: Path) -> dict[str, Any]:
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "DEEPSPEED",
        "mixed_precision": "bf16",
        "num_processes": max(env.num_gpus, 1),
        "deepspeed_config": str(deepspeed_path.resolve()),
    }


def _single_gpu_config(env: ComputeEnvironment) -> dict[str, Any]:
    use_cpu = env.device.value == "cpu"
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "NO",
        "mixed_precision": "fp16" if not use_cpu else "no",
        "use_cpu": use_cpu,
        "num_processes": 1,
    }


def _multi_gpu_config(env: ComputeEnvironment) -> dict[str, Any]:
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "mixed_precision": "bf16",
        "num_processes": env.num_gpus,
        "gpu_ids": ",".join(str(i) for i in range(env.num_gpus)),
    }


def _multi_node_config(env: ComputeEnvironment) -> dict[str, Any]:
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "mixed_precision": "bf16",
        "num_machines": env.num_nodes,
        "num_processes": env.num_gpus * env.num_nodes,
        "main_process_ip": "127.0.0.1",
        "main_process_port": 29500,
    }


def _dict_to_yaml_lines(d: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_dict_to_yaml_lines(value, indent + 1))
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return lines
