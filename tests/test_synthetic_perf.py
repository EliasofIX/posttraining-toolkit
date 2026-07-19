"""Synthetic generation efficiency tests."""

from unittest.mock import patch

from ptk.config.schema import DataSource, PTKConfig, SyntheticConfig, TrainingMethod
from ptk.data.generation import generate_synthetic_data
from ptk.data.table import Table
from ptk.logging import Logger


def test_synthetic_raw_cache(tmp_path):
    config = PTKConfig(
        run_name="syn-cache",
        base_model="distilgpt2",
        method=TrainingMethod.SFT,
        data=__import__("ptk.config.schema", fromlist=["DataConfig"]).DataConfig(
            source=DataSource.SYNTHETIC,
            synthetic=SyntheticConfig(n_samples=2, backend=__import__("ptk.config.schema", fromlist=["SyntheticBackend"]).SyntheticBackend.SELF_INSTRUCT),
        ),
        output=__import__("ptk.config.schema", fromlist=["OutputConfig"]).OutputConfig(dir=str(tmp_path / "outputs")),
    )
    logger = Logger(machine=False, verbose=False)
    generated = Table.from_dict({"text": ["cached sample one", "cached sample two"]})

    with patch("ptk.data.generation.get_generator") as mock_get:
        mock_get.return_value.generate.return_value = generated
        first = generate_synthetic_data(config, logger)
        second = generate_synthetic_data(config, logger)

    assert len(first) == 2
    assert len(second) == 2
    mock_get.return_value.generate.assert_called_once()


def test_self_instruct_batching():
    from ptk.data.generators.self_instruct import SelfInstructGenerator

    generator = SelfInstructGenerator(model="distilgpt2", batch_size=2)
    with patch.object(generator, "_generate_batch", side_effect=[["q1", "q2"], ["a1", "a2"]]) as mock_batch:
        result = generator.generate(["seed"], n_samples=2)

    assert mock_batch.call_count == 2
    assert len(result) >= 1
