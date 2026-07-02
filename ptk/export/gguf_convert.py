"""HuggingFace to GGUF conversion with native GPT-2 support and llama.cpp fallback."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import torch
from gguf import MODEL_ARCH, GGUFWriter, LlamaFileType, TensorNameMap
from gguf.vocab import BpeVocab, SpecialVocab

from ptk.exceptions import RuntimeError as PTKRuntimeError
from ptk.io.safetensors import load_file as load_safetensors

logger = logging.getLogger(__name__)

OutType = Literal["f32", "f16", "bf16", "q8_0"]

# Self-contained llama.cpp convert script (pinned commit) for general architectures.
_LLAMA_CPP_CONVERT_URL = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/b3999/convert_hf_to_gguf.py"
)
_CACHE_DIR = Path(os.environ.get("PTK_CACHE_DIR", Path.home() / ".cache" / "ptk"))


def convert_hf_to_gguf(
    model_dir: str | Path,
    output_path: str | Path,
    *,
    outtype: OutType = "f16",
) -> Path:
    """Convert a HuggingFace model directory to GGUF.

    Conversion backends (tried in order):
    1. Native GPT-2 converter (distilgpt2, gpt2, etc.)
    2. llama.cpp ``convert_hf_to_gguf.py`` on PATH or ``LLAMA_CPP_PATH``
    3. Cached download of llama.cpp convert script
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arch = _detect_architecture(model_dir)
    if arch == "gpt2":
        logger.info("Using native GPT-2 GGUF converter")
        return _convert_gpt2_native(model_dir, output_path, outtype=outtype)

    script = _find_convert_script()
    if not script:
        script = _download_convert_script()
    if script:
        logger.info("Using llama.cpp convert script: %s", script)
        return _convert_via_script(script, model_dir, output_path, outtype=outtype)

    raise PTKRuntimeError(
        f"No GGUF converter available for architecture '{arch}'. "
        "For GPT-2 models, install gguf: pip install posttraining-toolkit[gguf]. "
        "For other architectures, set LLAMA_CPP_PATH to a llama.cpp checkout or "
        "place convert_hf_to_gguf.py on PATH.",
        code="GGUF_CONVERT_FAILED",
        details={"architecture": arch, "model_dir": str(model_dir)},
    )


def _detect_architecture(model_dir: Path) -> str:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise PTKRuntimeError(f"config.json not found in {model_dir}", code="GGUF_CONVERT_FAILED")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type", "").lower()
    architectures = config.get("architectures", [])

    if model_type == "gpt2" or any("GPT2" in a for a in architectures):
        return "gpt2"
    if model_type:
        return model_type
    if architectures:
        return architectures[0]
    return "unknown"


def _find_convert_script() -> Path | None:
    """Locate llama.cpp convert_hf_to_gguf.py."""
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path:
        candidate = Path(env_path) / "convert_hf_to_gguf.py"
        if candidate.exists():
            return candidate

    which = shutil.which("convert_hf_to_gguf.py")
    if which:
        return Path(which)

    cached = _CACHE_DIR / "convert_hf_to_gguf.py"
    if cached.exists():
        return cached

    return None


def _download_convert_script() -> Path:
    """Download and cache llama.cpp convert script."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / "convert_hf_to_gguf.py"
    if dest.exists():
        return dest

    try:
        import urllib.request

        logger.info("Downloading llama.cpp convert script to %s", dest)
        urllib.request.urlretrieve(_LLAMA_CPP_CONVERT_URL, dest)  # noqa: S310
    except Exception as exc:
        raise PTKRuntimeError(
            f"Failed to download convert script: {exc}",
            code="GGUF_CONVERT_FAILED",
        ) from exc
    return dest


def _convert_via_script(
    script: Path,
    model_dir: Path,
    output_path: Path,
    *,
    outtype: OutType,
) -> Path:
    cmd = [
        sys.executable,
        str(script),
        str(model_dir),
        "--outfile",
        str(output_path),
        "--outtype",
        outtype,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise PTKRuntimeError(
            f"llama.cpp conversion failed: {exc.stderr or exc.stdout or exc}",
            code="GGUF_CONVERT_FAILED",
        ) from exc

    if not output_path.exists():
        raise PTKRuntimeError(
            f"Conversion completed but output not found: {output_path}",
            code="GGUF_CONVERT_FAILED",
        )
    return output_path


def _file_type(outtype: OutType) -> LlamaFileType:
    mapping = {
        "f32": LlamaFileType.ALL_F32,
        "f16": LlamaFileType.MOSTLY_F16,
        "bf16": LlamaFileType.MOSTLY_BF16,
        "q8_0": LlamaFileType.MOSTLY_Q8_0,
    }
    return mapping.get(outtype, LlamaFileType.MOSTLY_F16)


def _torch_dtype(outtype: OutType) -> torch.dtype:
    if outtype == "f32":
        return torch.float32
    if outtype == "bf16":
        return torch.bfloat16
    return torch.float16


def _convert_gpt2_native(model_dir: Path, output_path: Path, *, outtype: OutType) -> Path:
    """Native GPT-2/distilgpt2 converter using the gguf library."""
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    n_layer = config["n_layer"]
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    n_ctx = config.get("n_ctx", config.get("n_positions", 1024))
    layer_norm_eps = config.get("layer_norm_epsilon", 1e-5)

    writer = GGUFWriter(str(output_path), arch="gpt2")
    tensor_map = TensorNameMap(MODEL_ARCH.GPT2, n_layer)
    ftype = _file_type(outtype)
    torch_dtype = _torch_dtype(outtype)

    writer.add_block_count(n_layer)
    writer.add_context_length(n_ctx)
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(4 * n_embd)
    writer.add_head_count(n_head)
    writer.add_layer_norm_eps(layer_norm_eps)
    writer.add_file_type(ftype)

    _write_gpt2_vocab(model_dir, writer)

    state_dict = _load_state_dict(model_dir)
    for name, tensor in state_dict.items():
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            continue

        data = tensor.detach().cpu()
        if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight")):
            data = data.transpose(0, 1).contiguous()

        gguf_name = tensor_map.get_name(name)
        if gguf_name is None:
            logger.debug("Skipping unmapped tensor: %s", name)
            continue

        arr = data.to(dtype=torch_dtype).contiguous().numpy()
        writer.add_tensor(gguf_name, arr)

        # GPT-2 ties output weights to token embeddings
        if gguf_name == "token_embd.weight":
            writer.add_tensor("output.weight", arr.copy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if not output_path.exists():
        raise PTKRuntimeError(
            f"Native GPT-2 conversion failed to produce {output_path}",
            code="GGUF_CONVERT_FAILED",
        )
    return output_path


def _write_gpt2_vocab(model_dir: Path, writer: GGUFWriter) -> None:
    bpe = BpeVocab(model_dir)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")

    tokens: list[str] = []
    scores: list[float] = []
    toktypes: list[int] = []
    for token_bytes, score, toktype in bpe.all_tokens():
        if isinstance(token_bytes, str):
            token_str = token_bytes
        else:
            token_str = token_bytes.decode("utf-8", errors="replace")
        tokens.append(token_str)
        scores.append(score)
        toktypes.append(int(toktype))

    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    writer.add_vocab_size(len(tokens))

    special = SpecialVocab(model_dir, load_merges=True)
    special.add_to_gguf(writer, quiet=True)


def _load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        return load_safetensors(str(safetensors_path))

    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=True)

    raise PTKRuntimeError(
        f"No model weights found in {model_dir}",
        code="GGUF_CONVERT_FAILED",
    )
