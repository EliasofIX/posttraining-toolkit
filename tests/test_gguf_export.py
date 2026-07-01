"""GGUF export tests."""


import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptk.export.gguf_convert import convert_hf_to_gguf


@pytest.mark.slow
def test_convert_distilgpt2_to_gguf(tmp_path):
    model_dir = tmp_path / "hf_model"
    out_path = tmp_path / "model.gguf"

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    result = convert_hf_to_gguf(model_dir, out_path, outtype="f16")
    assert result.exists()
    assert result.stat().st_size > 100_000
    assert result.suffix == ".gguf"


def test_detect_gpt2_architecture(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"model_type": "gpt2", "n_layer": 6}', encoding="utf-8")

    from ptk.export.gguf_convert import _detect_architecture

    assert _detect_architecture(tmp_path) == "gpt2"
