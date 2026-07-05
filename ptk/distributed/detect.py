"""Environment and hardware topology detection."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ptk.config.schema import ComputeStrategy, DeviceType
from ptk.exceptions import ValidationError


@dataclass
class ComputeEnvironment:
    """Detected compute environment."""

    device: DeviceType
    strategy: ComputeStrategy
    num_gpus: int = 0
    num_nodes: int = 1
    cuda_available: bool = False
    mps_available: bool = False
    slurm: bool = False
    world_size: int = 1
    local_rank: int = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_device(requested: DeviceType = DeviceType.AUTO) -> DeviceType:
    """Resolve device from auto or explicit request."""
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if requested == DeviceType.AUTO:
        if cuda:
            return DeviceType.CUDA
        if mps:
            return DeviceType.MPS
        return DeviceType.CPU

    if requested == DeviceType.CUDA and not cuda:
        raise ValidationError(
            "CUDA requested but not available",
            field_path="compute.device",
        )
    if requested == DeviceType.MPS and not mps:
        raise ValidationError(
            "MPS requested but not available (requires Apple Silicon)",
            field_path="compute.device",
        )
    return requested


def detect_strategy(
    requested: ComputeStrategy,
    device: DeviceType,
    num_gpus: int,
    num_nodes: int,
) -> ComputeStrategy:
    """Resolve compute strategy with MPS limitations."""
    if device == DeviceType.MPS:
        if requested in (ComputeStrategy.MULTI_GPU, ComputeStrategy.MULTI_NODE):
            raise ValidationError(
                f"compute.strategy '{requested.value}' is not supported on Apple Silicon (MPS). "
                "Use 'single_gpu' or 'auto'.",
                field_path="compute.strategy",
            )
        return ComputeStrategy.SINGLE_GPU

    if device == DeviceType.CPU:
        return ComputeStrategy.SINGLE_GPU

    if requested != ComputeStrategy.AUTO:
        return requested

    if num_nodes > 1:
        return ComputeStrategy.MULTI_NODE
    if num_gpus > 1:
        return ComputeStrategy.MULTI_GPU
    return ComputeStrategy.SINGLE_GPU


def count_gpus() -> int:
    """Count available CUDA GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def detect_slurm() -> dict[str, Any] | None:
    """Return SLURM environment info if present."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    return {
        "job_id": job_id,
        "nodes": int(os.environ.get("SLURM_NNODES", "1")),
        "ntasks": int(os.environ.get("SLURM_NTASKS", "1")),
        "gpus_per_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
    }


def detect_environment(
    *,
    device: DeviceType = DeviceType.AUTO,
    strategy: ComputeStrategy = ComputeStrategy.AUTO,
    nodes: int | None = None,
    gpus_per_node: int | None = None,
) -> ComputeEnvironment:
    """Full environment detection for planning and validation."""
    resolved_device = detect_device(device)
    num_gpus = count_gpus() if resolved_device == DeviceType.CUDA else (1 if resolved_device == DeviceType.MPS else 0)
    slurm_info = detect_slurm()
    num_nodes = nodes or (slurm_info["nodes"] if slurm_info else 1)
    world_size = int(os.environ.get("WORLD_SIZE", str(max(num_gpus, 1) * num_nodes)))

    env = ComputeEnvironment(
        device=resolved_device,
        strategy=ComputeStrategy.SINGLE_GPU,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        cuda_available=torch.cuda.is_available(),
        mps_available=hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        slurm=slurm_info is not None,
        world_size=world_size,
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        metadata={"slurm": slurm_info} if slurm_info else {},
    )

    if gpus_per_node:
        env.num_gpus = gpus_per_node

    env.strategy = detect_strategy(strategy, resolved_device, env.num_gpus, num_nodes)

    if env.strategy in (ComputeStrategy.MULTI_GPU, ComputeStrategy.MULTI_NODE):
        if resolved_device != DeviceType.CUDA:
            raise ValidationError(
                "Multi-GPU/multi-node requires CUDA",
                field_path="compute.strategy",
            )
        if not shutil.which("accelerate"):
            env.warnings.append("accelerate CLI not found in PATH")

    if resolved_device == DeviceType.MPS and strategy != ComputeStrategy.SINGLE_GPU:
        if strategy == ComputeStrategy.AUTO:
            env.warnings.append("MPS detected: distributed training limited to single device")

    return env


def torch_device_string(device: DeviceType) -> str:
    """Map DeviceType to torch device string."""
    mapping = {
        DeviceType.CUDA: "cuda",
        DeviceType.MPS: "mps",
        DeviceType.CPU: "cpu",
        DeviceType.AUTO: "cpu",
    }
    return mapping[device]


def resolve_mixed_precision(device: DeviceType, requested: str) -> str:
    """Resolve mixed precision setting per device."""
    if requested != "auto":
        return requested
    if device == DeviceType.CUDA:
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    if device == DeviceType.MPS:
        return "fp16"
    return "fp32"


def is_main_process() -> bool:
    """Return True on global rank 0 in distributed runs."""
    if os.environ.get("RANK") is not None:
        return int(os.environ["RANK"]) == 0
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return True


def is_distributed_run() -> bool:
    """Return True when running under torchrun/accelerate."""
    if os.environ.get("LOCAL_RANK") is not None:
        return True
    if os.environ.get("PTK_DISTRIBUTED_ACTIVE") == "1":
        return True
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def distributed_barrier() -> None:
    """Synchronize all distributed ranks when torch.distributed is active."""
    if not is_distributed_run():
        return
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def wait_for_cache_manifest(cache_dir: Path, *, timeout_seconds: float = 600.0) -> None:
    """Block until rank 0 finishes writing processed cache."""
    import time

    timeout_seconds = float(os.environ.get("PTK_CACHE_WAIT_SECONDS", timeout_seconds))
    ready_marker = cache_dir / ".ready"
    manifest = cache_dir / "manifest.json"

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if ready_marker.exists() and manifest.exists():
            return
        time.sleep(0.5)

    raise TimeoutError(f"Timed out waiting for processed cache ready marker: {ready_marker}")


def nvidia_smi_info() -> dict[str, Any] | None:
    """Query nvidia-smi if available."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        gpus = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        return {"gpus": gpus, "count": len(gpus)}
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
