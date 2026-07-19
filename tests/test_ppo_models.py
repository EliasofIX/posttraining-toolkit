"""PPO trainer model-graph unit tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ptk.config.schema import (
    ComputeConfig,
    ComputeStrategy,
    DataConfig,
    DatasetConfig,
    DatasetFormat,
    DeviceType,
    PTKConfig,
    RLConfig,
    TrainingConfig,
    TrainingMethod,
)
from ptk.data.table import Table, TableDict
from ptk.distributed.detect import ComputeEnvironment
from ptk.logging import Logger
from ptk.training.rl.ppo import PPOTrainerWrapper


class _FakeBackbone:
    def __init__(self) -> None:
        self.config = SimpleNamespace(hidden_size=16, n_embd=16)
        self.transformer = object()
        self.save_pretrained = MagicMock()


def test_ppo_uses_separate_policy_reward_value_backbones():
    config = PTKConfig(
        run_name="ppo-separate",
        base_model="distilgpt2",
        method=TrainingMethod.PPO,
        data=DataConfig(dataset=DatasetConfig(path="x.jsonl", format=DatasetFormat.JSONL)),
        training=TrainingConfig(
            max_iters=1,
            batch_size=1,
            rl=RLConfig(reward_model="distilgpt2", max_completion_length=8),
        ),
        compute=ComputeConfig(
            device=DeviceType.CPU,
            strategy=ComputeStrategy.SINGLE_GPU,
            mixed_precision="fp32",
        ),
    )
    env = ComputeEnvironment(device=DeviceType.CPU, strategy=ComputeStrategy.SINGLE_GPU)
    logger = Logger(machine=False, verbose=False)
    wrapper = PPOTrainerWrapper(config, env, logger)
    dataset = TableDict({"train": Table.from_dict({"text": ["hello world", "another prompt"]})})

    models: list[_FakeBackbone] = []

    def fake_from_pretrained(*args, **kwargs):
        model = _FakeBackbone()
        models.append(model)
        return model

    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 1

    class FakeValueHead:
        def __init__(self, backbone):
            self.pretrained_model = backbone
            self.v_head = MagicMock()

    with patch(
        "ptk.training.rl.ppo.AutoModelForCausalLM.from_pretrained",
        side_effect=fake_from_pretrained,
    ), patch(
        "ptk.training.rl.ppo.AutoModelForCausalLMWithValueHead",
        side_effect=FakeValueHead,
    ), patch("ptk.training.rl.ppo.PPOTrainer", return_value=mock_trainer) as ppo_cls, patch(
        "ptk.training.rl.ppo.prepare_tokenizer"
    ) as tok, patch("ptk.training.rl.ppo.to_hf_dataset_dict") as to_hf, patch.object(
        wrapper, "save_training_metadata"
    ):
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda text, **kw: {"input_ids": [1, 2, 3]}
        tokenizer.save_pretrained = MagicMock()
        tok.return_value = tokenizer

        train_ds = MagicMock()
        train_ds.column_names = ["text"]
        train_ds.map.return_value = train_ds
        to_hf.return_value = {"train": train_ds}

        wrapper.train(dataset)

    assert len(models) == 3
    policy, reward_backbone, value_backbone = models
    assert policy is not reward_backbone
    assert policy is not value_backbone
    assert reward_backbone is not value_backbone

    kwargs = ppo_cls.call_args.kwargs
    assert kwargs["model"] is policy
    assert kwargs["reward_model"].transformer is reward_backbone.transformer
    assert kwargs["value_model"].pretrained_model is value_backbone
