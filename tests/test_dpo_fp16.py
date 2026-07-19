"""DPO mixed precision wiring tests."""

from unittest.mock import MagicMock, patch

from ptk.config.schema import (
    DataConfig,
    DatasetConfig,
    DatasetFormat,
    DeviceType,
    PTKConfig,
    RLConfig,
    TrainingConfig,
    TrainingMethod,
)
from ptk.distributed.detect import ComputeEnvironment
from ptk.logging import Logger


def test_dpo_config_receives_fp16_on_cuda():
    config = PTKConfig(
        run_name="dpo-fp16",
        base_model="distilgpt2",
        method=TrainingMethod.DPO,
        data=DataConfig(dataset=DatasetConfig(path="x.jsonl", format=DatasetFormat.JSONL)),
        training=TrainingConfig(rl=RLConfig(), max_iters=1),
    )
    env = ComputeEnvironment(device=DeviceType.CUDA, strategy=__import__("ptk.config.schema", fromlist=["ComputeStrategy"]).ComputeStrategy.SINGLE_GPU)
    logger = Logger(machine=False, verbose=False)

    from ptk.data.table import Table, TableDict
    from ptk.training.rl.dpo import DPOTrainerWrapper

    dataset = TableDict({"train": Table.from_dict({"prompt": ["p"], "chosen": ["c"], "rejected": ["r"]})})

    wrapper = DPOTrainerWrapper(config, env, logger)
    captured: dict = {}

    def fake_dpo_config(**kwargs):
        captured.update(kwargs)
        mock = MagicMock()
        mock.output_dir = str(wrapper.output_dir)
        return mock

    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 1
    mock_trainer.model = MagicMock()

    with patch("ptk.training.rl.dpo.DPOConfig", side_effect=fake_dpo_config), patch(
        "ptk.training.rl.dpo.DPOTrainer", return_value=mock_trainer
    ), patch("ptk.training.rl.dpo.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()), patch(
        "ptk.training.rl.dpo.prepare_tokenizer", return_value=MagicMock()
    ), patch("ptk.training.rl.dpo.resolve_mixed_precision", return_value="fp16"):
        wrapper.train(dataset)

    assert captured.get("fp16") is True
    assert captured.get("use_cpu") is False
