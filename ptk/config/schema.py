"""Pydantic configuration models with full validation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingMethod(str, Enum):
    SFT = "sft"
    LORA = "lora"
    QLORA = "qlora"
    DPO = "dpo"
    PPO = "ppo"
    GRPO = "grpo"


class DataSource(str, Enum):
    SYNTHETIC = "synthetic"
    DATASET = "dataset"


class SyntheticBackend(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    SELF_INSTRUCT = "self_instruct"


class DatasetFormat(str, Enum):
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HF_HUB = "hf_hub"


class ComputeStrategy(str, Enum):
    AUTO = "auto"
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    MULTI_NODE = "multi_node"


class DeviceType(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class ExportFormat(str, Enum):
    MERGED_FP16 = "merged_fp16"
    GGUF = "gguf"
    ADAPTER_ONLY = "adapter_only"


class QuantBits(int, Enum):
    FOUR = 4
    EIGHT = 8


class QuantType(str, Enum):
    NF4 = "nf4"
    FP4 = "fp4"


class QuantBackend(str, Enum):
    BNB = "bnb"
    MLX_QUANT = "mlx_quant"
    AUTO = "auto"


class SyntheticFilters(BaseModel):
    min_length: int = Field(default=10, ge=1)
    max_length: int = Field(default=4096, ge=1)
    dedup_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    min_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SyntheticConfig(BaseModel):
    backend: SyntheticBackend = SyntheticBackend.SELF_INSTRUCT
    seed_prompts: str | None = None
    n_samples: int = Field(default=100, ge=1)
    model: str | None = None
    filters: SyntheticFilters = Field(default_factory=SyntheticFilters)


class DatasetConfig(BaseModel):
    path: str
    format: DatasetFormat = DatasetFormat.JSONL
    text_column: str = "text"
    prompt_column: str | None = None
    response_column: str | None = None
    chosen_column: str | None = None
    rejected_column: str | None = None
    streaming: bool = False


class DataConfig(BaseModel):
    source: DataSource = DataSource.DATASET
    synthetic: SyntheticConfig | None = None
    dataset: DatasetConfig | None = None
    train_split: float = Field(default=0.9, gt=0.0, lt=1.0)
    seed: int = 42

    @model_validator(mode="after")
    def validate_source(self) -> DataConfig:
        if self.source == DataSource.SYNTHETIC and self.synthetic is None:
            self.synthetic = SyntheticConfig()
        if self.source == DataSource.DATASET and self.dataset is None:
            raise ValueError("data.dataset is required when data.source is 'dataset'")
        return self


class LoRAConfig(BaseModel):
    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: list[str] | None = None
    bias: str = "none"


class QuantizationConfig(BaseModel):
    bits: QuantBits = QuantBits.FOUR
    quant_type: QuantType = QuantType.NF4
    backend: QuantBackend = QuantBackend.AUTO
    double_quant: bool = True


class RLConfig(BaseModel):
    beta: float = Field(default=0.1, ge=0.0)
    reward_model: str | None = None
    kl_coef: float = Field(default=0.05, ge=0.0)
    num_generations: int = Field(default=4, ge=1)
    max_prompt_length: int = Field(default=512, ge=1)
    max_completion_length: int = Field(default=256, ge=1)
    ref_model_quantization: Literal["none", "4bit"] = "none"
    precompute_ref_log_probs: bool = False


class TrainingConfig(BaseModel):
    epochs: int = Field(default=1, ge=1)
    max_iters: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_seq_length: int = Field(default=512, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    save_strategy: Literal["steps", "epoch", "no"] = "steps"
    dataloader_num_workers: int | None = Field(default=None, ge=0)
    dataloader_pin_memory: bool | None = None
    dataloader_prefetch_factor: int | None = Field(default=None, ge=1)
    dataloader_persistent_workers: bool | None = None
    dataset_num_proc: int | None = Field(default=None, ge=1)
    gradient_checkpointing: bool = False
    lora: LoRAConfig | None = None
    quantization: QuantizationConfig | None = None
    rl: RLConfig | None = None


class ComputeConfig(BaseModel):
    strategy: ComputeStrategy = ComputeStrategy.AUTO
    device: DeviceType = DeviceType.AUTO
    nodes: int | None = Field(default=None, ge=1)
    gpus_per_node: int | None = Field(default=None, ge=1)
    accelerate_config: str | None = None
    deepspeed_config: str | None = None
    mixed_precision: Literal["auto", "fp16", "bf16", "fp32"] = "auto"


class EvalConfig(BaseModel):
    benchmarks: list[str] = Field(default_factory=list)
    eval_steps: int = Field(default=100, ge=1)
    max_samples: int | None = Field(default=None, ge=1)


class OutputConfig(BaseModel):
    dir: str = "./outputs"
    export: list[ExportFormat] = Field(default_factory=lambda: [ExportFormat.ADAPTER_ONLY])
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_private: bool = True


class PTKConfig(BaseModel):
    """Top-level posttraining toolkit configuration."""

    run_name: str = Field(..., min_length=1)
    base_model: str = Field(..., min_length=1)
    method: TrainingMethod
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("run_name")
    @classmethod
    def sanitize_run_name(cls, v: str) -> str:
        if any(c in v for c in "/\\:*?\"<>|"):
            raise ValueError("run_name must not contain path separators or special chars")
        return v

    @model_validator(mode="after")
    def validate_method_requirements(self) -> PTKConfig:
        method = self.method
        if method in (TrainingMethod.LORA, TrainingMethod.QLORA):
            if self.training.lora is None:
                self.training.lora = LoRAConfig()
        if method == TrainingMethod.QLORA:
            if self.training.quantization is None:
                self.training.quantization = QuantizationConfig()
        if method in (TrainingMethod.DPO, TrainingMethod.PPO, TrainingMethod.GRPO):
            if self.training.rl is None:
                self.training.rl = RLConfig()
        if method == TrainingMethod.PPO and not self.training.rl.reward_model:
            raise ValueError("training.rl.reward_model is required for PPO")
        if method == TrainingMethod.GRPO:
            assert self.training.rl is not None
            num_generations = self.training.rl.num_generations
            if num_generations < 2:
                raise ValueError("training.rl.num_generations must be >= 2 for GRPO")
            generation_batch_size = (
                self.training.batch_size * self.training.gradient_accumulation_steps
            )
            if generation_batch_size % num_generations != 0:
                raise ValueError(
                    "For GRPO, training.batch_size * training.gradient_accumulation_steps "
                    f"({generation_batch_size}) must be divisible by training.rl.num_generations "
                    f"({num_generations})"
                )
        return self

    def output_path(self) -> Path:
        return Path(self.output.dir) / self.run_name
