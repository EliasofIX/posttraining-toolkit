"""Evaluation harness for post-training benchmarks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptk.config.schema import PTKConfig
from ptk.distributed.detect import detect_environment, torch_device_string
from ptk.logging import Logger
from ptk.registry.runs import RunRegistry


BENCHMARKS: dict[str, dict[str, Any]] = {
    "perplexity": {
        "description": "Compute perplexity on validation samples",
        "default_samples": 10,
    },
    "generation_quality": {
        "description": "Basic generation length and diversity metrics",
        "default_samples": 5,
    },
    "instruction_following": {
        "description": "Smoke test for instruction-response format",
        "default_samples": 3,
    },
}


class EvalHarness:
    """Run configurable benchmark suites."""

    def __init__(self, logger: Logger | None = None) -> None:
        self.logger = logger or Logger()

    def run(
        self,
        config: PTKConfig,
        *,
        checkpoint_path: Path | None = None,
        benchmarks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute benchmarks and return structured results."""
        bench_list = benchmarks or config.eval.benchmarks or ["perplexity"]
        env = detect_environment(device=config.compute.device, strategy=config.compute.strategy)
        device = torch_device_string(env.device)

        model_path = str(checkpoint_path or config.output_path() / "checkpoints" / "final")
        if not Path(model_path).exists():
            model_path = config.base_model

        self.logger.start("Evaluation", benchmarks=bench_list, model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        if device != "cpu":
            model.to(device)
        model.eval()

        results: dict[str, Any] = {"benchmarks": {}, "model": model_path, "timestamp": time.time()}
        max_samples = config.eval.max_samples

        for bench in bench_list:
            if bench not in BENCHMARKS:
                results["benchmarks"][bench] = {"error": f"Unknown benchmark: {bench}"}
                continue
            fn = getattr(self, f"_bench_{bench}", None)
            if fn is None:
                results["benchmarks"][bench] = {"error": f"Not implemented: {bench}"}
                continue
            samples = max_samples or BENCHMARKS[bench]["default_samples"]
            results["benchmarks"][bench] = fn(model, tokenizer, device, samples)

        self.logger.complete("Evaluation complete", benchmarks=list(results["benchmarks"].keys()))
        return results

    def _bench_perplexity(self, model, tokenizer, device: str, n: int) -> dict[str, float]:
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models learn patterns from data.",
            "Fine-tuning adapts pretrained weights to downstream tasks.",
        ][:n]
        total_loss = 0.0
        total_tokens = 0
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()
        avg_loss = total_loss / max(total_tokens, 1)
        return {"perplexity": float(torch.exp(torch.tensor(avg_loss))), "avg_loss": avg_loss}

    def _bench_generation_quality(self, model, tokenizer, device: str, n: int) -> dict[str, float]:
        prompts = ["Explain AI in simple terms.", "What is LoRA?"][:n]
        lengths = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            lengths.append(len(text.split()))
        return {
            "avg_output_length": sum(lengths) / max(len(lengths), 1),
            "num_samples": len(lengths),
        }

    def _bench_instruction_following(self, model, tokenizer, device: str, n: int) -> dict[str, Any]:
        prompt = "Instruction: List three colors.\nResponse:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        has_response = "Response:" in response
        return {"passed": has_response, "sample_output": response[:200]}


def eval_run(run_id: str, logger: Logger, benchmarks: list[str] | None = None) -> dict[str, Any]:
    """Evaluate a registered run by ID."""
    registry = RunRegistry()
    record = registry.get_run(run_id)
    if record is None:
        raise ValueError(f"Run not found: {run_id}")

    config = PTKConfig.model_validate(record.config)
    checkpoint = registry.find_latest_checkpoint(run_id)
    harness = EvalHarness(logger)
    results = harness.run(config, checkpoint_path=checkpoint, benchmarks=benchmarks)

    out_path = config.output_path() / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    registry.update_run(run_id, metrics={**record.metrics, **{f"eval_{k}": v for k, v in _flatten_metrics(results).items()}})
    return results


def _flatten_metrics(results: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for bench, data in results.get("benchmarks", {}).items():
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    flat[f"{bench}_{k}"] = float(v)
    return flat
