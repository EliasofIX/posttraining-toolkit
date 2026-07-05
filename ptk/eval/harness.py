"""Evaluation harness for post-training benchmarks."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ptk.config.schema import PTKConfig
from ptk.data.pipeline import load_processed_cache
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

        return self._run_benchmarks(config, model, tokenizer, device, bench_list, model_path=model_path)

    def run_with_model(
        self,
        config: PTKConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        benchmarks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run benchmarks on an in-memory model without reloading weights."""
        bench_list = benchmarks or config.eval.benchmarks
        if not bench_list:
            return {"benchmarks": {}, "model": "in-memory", "timestamp": time.time()}

        env = detect_environment(device=config.compute.device, strategy=config.compute.strategy)
        device = torch_device_string(env.device)
        self.logger.start("Evaluation", benchmarks=bench_list, model="in-memory")
        model.eval()
        return self._run_benchmarks(config, model, tokenizer, device, bench_list, model_path="in-memory")

    def _run_benchmarks(
        self,
        config: PTKConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        bench_list: list[str],
        *,
        model_path: str,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {"benchmarks": {}, "model": model_path, "timestamp": time.time()}
        max_samples = config.eval.max_samples
        eval_texts = self._load_eval_texts(config, max_samples or 32)

        for bench in bench_list:
            if bench not in BENCHMARKS:
                results["benchmarks"][bench] = {"error": f"Unknown benchmark: {bench}"}
                continue
            fn = getattr(self, f"_bench_{bench}", None)
            if fn is None:
                results["benchmarks"][bench] = {"error": f"Not implemented: {bench}"}
                continue
            samples = max_samples or BENCHMARKS[bench]["default_samples"]
            if bench in ("perplexity", "generation_quality"):
                results["benchmarks"][bench] = fn(model, tokenizer, device, samples, eval_texts)
            else:
                results["benchmarks"][bench] = fn(model, tokenizer, device, samples)

        self.logger.complete("Evaluation complete", benchmarks=list(results["benchmarks"].keys()))
        return results

    def _load_eval_texts(self, config: PTKConfig, limit: int) -> list[str]:
        """Sample evaluation texts from cached validation split when available."""
        cache_root = config.output_path() / "data_cache" / "processed"
        if not cache_root.exists():
            return []

        candidates = sorted(cache_root.glob("*/manifest.json"))
        for manifest_path in reversed(candidates):
            cached = load_processed_cache(manifest_path.parent)
            if cached is None or "validation" not in cached:
                continue
            validation = cached["validation"]
            text_key = "text" if "text" in validation.column_names else "prompt"
            texts = [str(validation[i].get(text_key, "")) for i in range(len(validation))]
            texts = [text for text in texts if text.strip()]
            if texts:
                random.seed(config.data.seed)
                random.shuffle(texts)
                return texts[:limit]
        return []

    def _bench_perplexity(
        self,
        model,
        tokenizer,
        device: str,
        n: int,
        eval_texts: list[str],
    ) -> dict[str, float]:
        texts = eval_texts[:n] if eval_texts else [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models learn patterns from data.",
            "Fine-tuning adapts pretrained weights to downstream tasks.",
        ][:n]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            token_count = inputs["input_ids"].numel()
            total_loss = outputs.loss.item() * token_count

        avg_loss = total_loss / max(token_count, 1)
        return {"perplexity": float(torch.exp(torch.tensor(avg_loss))), "avg_loss": avg_loss}

    def _bench_generation_quality(
        self,
        model,
        tokenizer,
        device: str,
        n: int,
        eval_texts: list[str],
    ) -> dict[str, float]:
        prompts = eval_texts[:n] if eval_texts else ["Explain AI in simple terms.", "What is LoRA?"][:n]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)

        lengths = []
        for row in out:
            text = tokenizer.decode(row, skip_special_tokens=True)
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
