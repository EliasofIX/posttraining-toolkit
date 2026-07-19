"""Evaluation benchmarks on MLX adapters."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from ptk.config.schema import PTKConfig
from ptk.exceptions import RuntimeError as PTKRuntimeError
from ptk.logging import Logger
from ptk.mlx.availability import require_mlx
from ptk.mlx.export import read_mlx_runtime
from ptk.mlx.trainer import is_mlx_checkpoint


def run_mlx_eval(
    config: PTKConfig,
    checkpoint_path: Path,
    *,
    benchmarks: list[str],
    logger: Logger | None = None,
    eval_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Run stock-shaped eval benchmarks using mlx-lm."""
    require_mlx()
    log = logger or Logger()
    if not is_mlx_checkpoint(checkpoint_path):
        raise PTKRuntimeError(
            f"Checkpoint is not an MLX adapter directory: {checkpoint_path}",
            code="MLX_EVAL_FAILED",
        )

    runtime = read_mlx_runtime(checkpoint_path)
    mlx_model = runtime.get("mlx_model_path") or config.base_model
    log.start("Evaluation", benchmarks=benchmarks, model=str(checkpoint_path), backend="mlx_quant")

    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.generate import generate
        from mlx_lm.utils import load
    except ImportError as exc:
        raise PTKRuntimeError(f"MLX eval imports failed: {exc}", code="MLX_IMPORT_FAILED") from exc

    try:
        model, tokenizer = load(
            str(mlx_model),
            adapter_path=str(checkpoint_path),
            tokenizer_config={"trust_remote_code": True},
        )
        model.eval()
    except Exception as exc:
        raise PTKRuntimeError(
            f"Failed to load MLX model/adapters for eval: {exc}",
            code="MLX_EVAL_FAILED",
        ) from exc

    texts = eval_texts or []
    max_samples = config.eval.max_samples
    results: dict[str, Any] = {
        "benchmarks": {},
        "model": str(checkpoint_path),
        "backend": "mlx_quant",
        "timestamp": time.time(),
    }

    for bench in benchmarks:
        if bench == "perplexity":
            n = max_samples or 10
            results["benchmarks"][bench] = _perplexity(model, tokenizer, texts, n, mx=mx, nn=nn)
        elif bench == "generation_quality":
            n = max_samples or 5
            results["benchmarks"][bench] = _generation_quality(
                model, tokenizer, texts, n, generate=generate
            )
        elif bench == "instruction_following":
            results["benchmarks"][bench] = _instruction_following(model, tokenizer, generate=generate)
        else:
            results["benchmarks"][bench] = {"error": f"Unknown benchmark: {bench}"}

    log.complete("Evaluation complete", benchmarks=list(results["benchmarks"].keys()))
    return results


def _default_texts() -> list[str]:
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn patterns from data.",
        "Fine-tuning adapts pretrained weights to downstream tasks.",
    ]


def _perplexity(model, tokenizer, eval_texts: list[str], n: int, *, mx, nn) -> dict[str, float]:
    texts = (eval_texts or _default_texts())[:n]
    if not texts:
        texts = _default_texts()[:n]

    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        arr = mx.array(tokens)[None, :]
        logits = model(arr)
        # logits: [1, T, V]; predict next token
        targets = arr[:, 1:]
        pred = logits[:, :-1, :]
        # cross entropy
        ce = nn.losses.cross_entropy(pred.reshape(-1, pred.shape[-1]), targets.reshape(-1))
        loss = ce.mean().item()
        ntoks = int(targets.size)
        total_loss += loss * ntoks
        total_tokens += ntoks
        mx.eval(pred)

    avg_loss = total_loss / max(total_tokens, 1)
    return {"perplexity": float(math.exp(avg_loss)), "avg_loss": float(avg_loss)}


def _generation_quality(model, tokenizer, eval_texts: list[str], n: int, *, generate) -> dict[str, float]:
    prompts = (eval_texts or ["Explain AI in simple terms.", "What is LoRA?"])[:n]
    lengths: list[int] = []
    for prompt in prompts:
        try:
            out = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
            text = out if isinstance(out, str) else str(out)
            lengths.append(len(text.split()))
        except Exception:
            lengths.append(0)
    return {
        "avg_output_length": sum(lengths) / max(len(lengths), 1),
        "num_samples": len(lengths),
    }


def _instruction_following(model, tokenizer, *, generate) -> dict[str, Any]:
    prompt = "Instruction: List three colors.\nResponse:"
    try:
        out = generate(model, tokenizer, prompt=prompt, max_tokens=30, verbose=False)
        response = out if isinstance(out, str) else str(out)
    except Exception as exc:
        return {"passed": False, "sample_output": str(exc)[:200]}
    return {"passed": "Response:" in response or len(response) > 0, "sample_output": response[:200]}
