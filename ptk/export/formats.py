"""Model export utilities."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptk.config.schema import ExportFormat, PTKConfig
from ptk.logging import Logger


def export_run(
    config: PTKConfig,
    *,
    formats: list[ExportFormat] | None = None,
    checkpoint_path: Path | None = None,
    logger: Logger | None = None,
) -> dict[str, str]:
    """Export trained model to requested formats."""
    log = logger or Logger()
    export_formats = formats or config.output.export
    ckpt = checkpoint_path or config.output_path() / "checkpoints" / "final"
    export_dir = config.output_path() / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}

    for fmt in export_formats:
        log.start(f"Exporting {fmt.value}")
        if fmt == ExportFormat.ADAPTER_ONLY:
            artifacts["adapter_only"] = _export_adapter_only(ckpt, export_dir / "adapter")
        elif fmt == ExportFormat.MERGED_FP16:
            artifacts["merged_fp16"] = _export_merged_fp16(config, ckpt, export_dir / "merged_fp16")
        elif fmt == ExportFormat.GGUF:
            artifacts["gguf"] = _export_gguf(config, ckpt, export_dir / "model.gguf")
        log.complete(f"Exported {fmt.value}", path=artifacts.get(fmt.value))

    if config.output.push_to_hub:
        _push_to_hub(config, artifacts, log)

    manifest = export_dir / "manifest.json"
    manifest.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    return artifacts


def _export_adapter_only(checkpoint: Path, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        if (checkpoint / "adapter_config.json").exists() or list(checkpoint.glob("adapter_*")):
            shutil.copytree(checkpoint, out_dir, dirs_exist_ok=True)
        else:
            shutil.copytree(checkpoint, out_dir, dirs_exist_ok=True)
    return str(out_dir)


def _export_merged_fp16(config: PTKConfig, checkpoint: Path, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_config = checkpoint / "adapter_config.json"

    if adapter_config.exists():
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(checkpoint))
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out_dir))
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        tokenizer.save_pretrained(str(out_dir))
    elif checkpoint.exists():
        shutil.copytree(checkpoint, out_dir, dirs_exist_ok=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.save_pretrained(str(out_dir))
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        tokenizer.save_pretrained(str(out_dir))

    return str(out_dir)


def _export_gguf(config: PTKConfig, checkpoint: Path, out_path: Path) -> str:
    """Export to GGUF using native converter or llama.cpp fallback."""
    merged_dir = out_path.parent / "merged_for_gguf"
    _export_merged_fp16(config, checkpoint, merged_dir)

    try:
        from ptk.export.gguf_convert import convert_hf_to_gguf

        result = convert_hf_to_gguf(merged_dir, out_path, outtype="f16")
        return str(result)
    except Exception as exc:
        from ptk.exceptions import RuntimeError as PTKRuntimeError

        if isinstance(exc, PTKRuntimeError):
            raise
        raise PTKRuntimeError(
            f"GGUF export failed: {exc}",
            code="GGUF_CONVERT_FAILED",
        ) from exc


def _push_to_hub(config: PTKConfig, artifacts: dict[str, str], logger: Logger) -> None:
    model_id = config.output.hub_model_id or config.run_name
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(model_id, private=config.output.hub_private, exist_ok=True)
        path = artifacts.get("merged_fp16") or artifacts.get("adapter_only")
        if path:
            api.upload_folder(folder_path=path, repo_id=model_id, repo_type="model")
            logger.complete("Pushed to Hub", repo_id=model_id)
    except Exception as exc:
        logger.warn(f"Hub push failed: {exc}")
