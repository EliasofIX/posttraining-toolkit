"""Default configuration cascade per training method."""

from __future__ import annotations

from ptk.config.schema import (
    ComputeConfig,
    DataConfig,
    DataSource,
    DatasetConfig,
    DatasetFormat,
    EvalConfig,
    LoRAConfig,
    OutputConfig,
    PTKConfig,
    QuantizationConfig,
    RLConfig,
    TrainingConfig,
    TrainingMethod,
)


def defaults_for_method(method: TrainingMethod) -> dict:
    """Return method-specific default overrides as a nested dict."""
    base: dict = {
        "method": method.value,
        "training": {},
        "compute": {},
        "eval": {},
        "output": {},
    }

    if method == TrainingMethod.SFT:
        base["training"].update(
            {
                "learning_rate": 2e-5,
                "batch_size": 4,
                "epochs": 3,
            }
        )
        base["output"]["export"] = ["adapter_only"]

    elif method == TrainingMethod.LORA:
        base["training"].update(
            {
                "learning_rate": 2e-4,
                "batch_size": 4,
                "epochs": 3,
                "lora": LoRAConfig().model_dump(),
            }
        )
        base["output"]["export"] = ["adapter_only", "merged_fp16"]

    elif method == TrainingMethod.QLORA:
        base["training"].update(
            {
                "learning_rate": 2e-4,
                "batch_size": 2,
                "epochs": 3,
                "lora": LoRAConfig().model_dump(),
                "quantization": QuantizationConfig().model_dump(),
            }
        )
        base["output"]["export"] = ["adapter_only"]

    elif method == TrainingMethod.DPO:
        base["training"].update(
            {
                "learning_rate": 5e-7,
                "batch_size": 2,
                "epochs": 1,
                "rl": RLConfig(beta=0.1).model_dump(),
            }
        )

    elif method == TrainingMethod.PPO:
        base["training"].update(
            {
                "learning_rate": 1e-5,
                "batch_size": 2,
                "epochs": 1,
                "rl": RLConfig(beta=0.1, reward_model="gpt2", kl_coef=0.05).model_dump(),
            }
        )

    elif method == TrainingMethod.GRPO:
        base["training"].update(
            {
                "learning_rate": 5e-6,
                "batch_size": 2,
                "epochs": 1,
                "rl": RLConfig(beta=0.04, num_generations=4).model_dump(),
            }
        )

    return base


def scaffold_config(
    method: TrainingMethod,
    *,
    run_name: str = "my-run",
    base_model: str = "distilgpt2",
) -> PTKConfig:
    """Build a commented-ready starter config for a given method."""
    overrides = defaults_for_method(method)
    data = DataConfig(
        source=DataSource.DATASET,
        dataset=DatasetConfig(path="./data/train.jsonl", format=DatasetFormat.JSONL),
    )
    if method == TrainingMethod.DPO:
        data = DataConfig(
            source=DataSource.DATASET,
            dataset=DatasetConfig(
                path="./data/preferences.jsonl",
                format=DatasetFormat.JSONL,
                prompt_column="prompt",
                chosen_column="chosen",
                rejected_column="rejected",
            ),
        )

    config = PTKConfig(
        run_name=run_name,
        base_model=base_model,
        method=method,
        data=data,
        training=TrainingConfig(**overrides.get("training", {})),
        compute=ComputeConfig(**overrides.get("compute", {})),
        eval=EvalConfig(**overrides.get("eval", {})),
        output=OutputConfig(**overrides.get("output", {})),
    )
    return config
