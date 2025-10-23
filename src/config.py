"""Configuration dataclasses and loader utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RunConfig:
    """Top-level run metadata and reproducibility controls."""

    name: Optional[str] = None
    seed: Optional[int] = 42
    output_dir: str = "./outputs"
    snapshot_dir: Optional[str] = None
    result_dir: Optional[str] = None
    print_model: bool = False


@dataclass
class ModelConfig:
    """Model loading and tokenizer options."""

    id: str = "gpt2"
    type: str = "CausalLM"
    dtype: str = "bf16"
    tokenizer_id: Optional[str] = None
    trust_remote_code: bool = True
    flash_attention: bool = False
    use_safetensors: bool = True
    revision: Optional[str] = None
    device_map: Optional[str] = None
    padding_side: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset selection and preprocessing controls."""

    name: str = "cola"
    subset: Optional[str] = None
    split_train: str = "train"
    split_eval: str = "validation"
    split_test: Optional[str] = None
    task: Optional[str] = None
    instruction_template: Optional[str] = None
    max_eval_samples: Optional[int] = None
    max_train_samples: Optional[int] = None


@dataclass
class LoragaGradientConfig:
    """Settings for gradient estimation used during LoRA-GA initialisation."""

    subset_size: Optional[int] = None
    quant_flag: bool = False
    save_path: Optional[str] = None
    origin_type: str = "bf16"
    quant_type: str = "nf4"
    no_split_module_classes: Optional[List[str]] = None


@dataclass
class LoragaConfig:
    """PEFT LoRA + GA configuration."""

    lora_alpha: int = 32
    r: int = 8
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: Optional[List[str]] = None
    modules_to_save: Optional[List[str]] = None
    init_lora_weights: Optional[str] = None
    task_type: Optional[str] = None
    inference_mode: bool = False
    bsz: Optional[int] = None
    sample_size: Optional[int] = None
    gradient: LoragaGradientConfig = field(default_factory=LoragaGradientConfig)


@dataclass
class TrainingConfig:
    """Training arguments wrapper compatible with :class:`transformers.TrainingArguments`."""

    num_train_epochs: float = 3.0
    per_device_batch_size: int = 1
    per_device_eval_batch_size: Optional[int] = None
    real_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = None
    logging_steps: int = 10
    eval_strategy: str = "steps"
    save_strategy: Optional[str] = None
    logging_strategy: Optional[str] = None
    max_length: int = 1024
    generation_max_length: Optional[int] = None
    generation_num_beams: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    predict_with_generate: bool = True
    gradient_checkpointing: bool = False
    early_stopping_patience: Optional[int] = None
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None
    load_best_model_at_end: bool = False
    use_loraplus: bool = False
    loraplus_lr_ratio: Optional[float] = None
    dataloader_num_workers: int = 0
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    logging_first_step: bool = True
    warmup_steps: Optional[int] = None
    optim: Optional[str] = None
    deepspeed: Optional[str] = None
    training_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WandbConfig:
    """Weights & Biases logging settings."""

    enabled: bool = False
    project: str = "peft-training"
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    mode: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Root configuration object."""

    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    loraga: Optional[LoragaConfig] = field(default_factory=LoragaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _deep_update_dataclass(dataclass_obj, data: Dict[str, Any]) -> Any:
    """Recursively update dataclass fields with values coming from raw dictionaries."""

    for field_name in data:
        if not hasattr(dataclass_obj, field_name):
            continue
        value = getattr(dataclass_obj, field_name)
        new_val = data[field_name]
        if hasattr(value, "__dataclass_fields__") and isinstance(new_val, dict):
            _deep_update_dataclass(value, new_val)
        elif isinstance(value, list) and isinstance(new_val, list):
            setattr(dataclass_obj, field_name, new_val)
        else:
            setattr(dataclass_obj, field_name, new_val)
    return dataclass_obj


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate configuration from a YAML file."""

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_cfg: Dict[str, Any] = yaml.safe_load(handle) or {}

    experiment = ExperimentConfig()
    for section_name, section_data in raw_cfg.items():
        if not hasattr(experiment, section_name):
            continue
        section_obj = getattr(experiment, section_name)
        if hasattr(section_obj, "__dataclass_fields__") and isinstance(section_data, dict):
            _deep_update_dataclass(section_obj, section_data)
        else:
            setattr(experiment, section_name, section_data)

    # Default per_device_eval_batch_size mirrors training batch if unspecified.
    if (
        experiment.training.per_device_eval_batch_size is None
        and experiment.training.per_device_batch_size is not None
    ):
        experiment.training.per_device_eval_batch_size = experiment.training.per_device_batch_size

    if experiment.training.generation_max_length is None:
        experiment.training.generation_max_length = experiment.training.max_length

    if experiment.wandb.name is None and experiment.run.name:
        experiment.wandb.name = experiment.run.name

    return experiment


__all__ = [
    "ExperimentConfig",
    "RunConfig",
    "ModelConfig",
    "DatasetConfig",
    "LoragaConfig",
    "LoragaGradientConfig",
    "TrainingConfig",
    "WandbConfig",
    "load_config",
]
