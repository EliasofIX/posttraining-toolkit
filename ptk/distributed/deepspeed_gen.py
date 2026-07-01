"""Generate DeepSpeed configuration files (CUDA-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ptk.distributed.detect import ComputeEnvironment


def generate_deepspeed_config(
    env: ComputeEnvironment,
    output_path: Path,
    *,
    zero_stage: int = 2,
    gradient_accumulation_steps: int = 1,
    train_batch_size: int = 4,
) -> Path:
    """Write a DeepSpeed ZeRO config JSON."""
    if env.device.value != "cuda":
        raise ValueError("DeepSpeed is CUDA-only")

    config: dict[str, Any] = {
        "train_batch_size": train_batch_size * gradient_accumulation_steps * max(env.num_gpus, 1),
        "train_micro_batch_size_per_gpu": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_path
