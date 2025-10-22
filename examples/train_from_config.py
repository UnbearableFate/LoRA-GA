import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import wandb
import yaml
from accelerate import Accelerator
from fire import Fire

from peft import PeftModel, LoraGAConfig, get_peft_model
from peft.utils.lora_ga_utils import (
    LoraGAContext,
    estimate_gradient,
    save_loraga_model_final,
    save_loraga_model_init,
)
from data import DATASET_MAP
from utils import (
    find_all_linear_modules,
    initialize_text_to_text_model,
    preprocess_dataset,
    seed_everything,
    train_text_to_text_model,
    transform_dataset,
)


def _load_yaml(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _get_nested(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return cfg[key] if key in cfg else default


def _select_subset(dataset: Any, limit: Optional[int]):
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("subset size must be positive")
    if hasattr(dataset, "__len__"):
        limit = min(limit, len(dataset))
    if isinstance(dataset, list):
        return dataset[:limit]
    indices = list(range(limit))
    return dataset.select(indices)  # type: ignore[attr-defined]


def _infer_experiment_name(
    model_id: str,
    dataset_name: str,
    loraga_cfg: Mapping[str, Any],
    seed: Optional[int],
) -> str:
    alias = model_id.split("/")[-1]
    alpha = loraga_cfg.get("lora_alpha")
    rank = loraga_cfg.get("r")
    sample_size = loraga_cfg.get("sample_size")
    components: Iterable[str] = (
        f"model={alias}",
        f"d={dataset_name}",
        f"a={alpha}" if alpha is not None else None,
        f"r={rank}" if rank is not None else None,
        f"s={sample_size}" if sample_size is not None else None,
        f"sd={seed}" if seed is not None else None,
    )
    return "_".join(filter(None, components))


def _ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main(config: str):
    cfg = _load_yaml(config)

    run_cfg = cfg.get("run", {})
    seed = _get_nested(run_cfg, "seed")
    if seed is not None:
        seed_everything(seed)

    accelerator = Accelerator()

    model_cfg = cfg.get("model", {})
    if "id" not in model_cfg:
        raise ValueError("model.id must be provided in the config file")
    model_id = model_cfg["id"]
    model_type = model_cfg.get("type", "CausalLM")
    model_dtype = model_cfg.get("dtype", "bf16")
    flash_attention = model_cfg.get("flash_attention", False)

    dataset_cfg = cfg.get("dataset", {})
    dataset_name = dataset_cfg.get("name")
    if dataset_name is None:
        raise ValueError("dataset.name must be provided in the config file")
    dataset_args = dict(dataset_cfg.get("args", {}))
    tokenizer_aware_datasets = {
        "meta_math",
        "meta_math_full",
        "meta_math_5k",
        "flan_v2",
        "codefeedback",
        "wizard_lm",
    }
    if dataset_name in tokenizer_aware_datasets:
        dataset_args.setdefault("tokenizer_name", model_id)

    loraga_cfg = cfg.get("loraga", {})
    loraga_config_kwargs = {
        k: v for k, v in loraga_cfg.items() if k not in {"sample_size", "gradient"}
    }
    target_modules = loraga_cfg.get("target_modules")
    # allow users to request automatic target discovery
    auto_target = target_modules in (None, "auto")

    if auto_target:
        # initialize the model before discovering target modules
        model, tokenizer = initialize_text_to_text_model(
            model_id,
            model_type,
            model_dtype,
            flash_attention=flash_attention,
        )
        target_modules = find_all_linear_modules(model)
    else:
        model, tokenizer = initialize_text_to_text_model(
            model_id,
            model_type,
            model_dtype,
            flash_attention=flash_attention,
        )

    loraga_config_kwargs["target_modules"] = target_modules

    sample_size = loraga_cfg.get("sample_size")
    if "iters" not in loraga_config_kwargs and sample_size is not None:
        # sample_size here denotes twice the number of gradient accumulation steps
        loraga_config_kwargs["iters"] = max(1, sample_size // 2)

    peft_config = LoraGAConfig(**loraga_config_kwargs)

    dataset_func = DATASET_MAP.get(dataset_name)
    if dataset_func is None:
        raise ValueError(f"Unknown dataset {dataset_name}. Available: {list(DATASET_MAP)}")
    train_set, val_set, _ = dataset_func(**dataset_args)

    gradient_cfg = loraga_cfg.get("gradient", {})
    subset_size = gradient_cfg.get(
        "subset_size", peft_config.bsz * peft_config.iters
    )
    temp_set = _select_subset(train_set, subset_size)
    if isinstance(temp_set, list):
        temp_set = preprocess_dataset(temp_set)
    transform_dataset(
        model_type=model_type,
        dataset=temp_set,
        tokenizer=tokenizer,
        max_length=peft_config.max_length,
    )
    dataloader = torch.utils.data.DataLoader(
        temp_set,
        batch_size=peft_config.bsz,
    )

    experiment_name = run_cfg.get(
        "name",
        _infer_experiment_name(model_id, dataset_name, loraga_cfg, seed),
    )
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", True)
    wandb_name = wandb_cfg.get("name", experiment_name)
    if wandb_enabled and accelerator.is_local_main_process:
        wandb.init(
            name=wandb_name,
            project=wandb_cfg.get("project", "LoRA-GA in PEFT"),
            group=wandb_cfg.get("group"),
            mode=wandb_cfg.get("mode", "offline"),
        )

    grad_save_path = gradient_cfg.get("save_path")
    if grad_save_path:
        grad_save_path = Path(grad_save_path , f"grad_save_{model_id.replace('/', '-')}_{dataset_name}.pt")
        _ensure_directory(grad_save_path.parent)
    else:
        grad_save_path = Path("data_cache") / f"grad_save_{model_id.replace('/', '-')}_{dataset_name}.pt"
        _ensure_directory(grad_save_path.parent)

    named_grad = estimate_gradient(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
        quant_flag=gradient_cfg.get("quant_flag", False),
        grad_save_path=str(grad_save_path),
    )

    if accelerator.is_local_main_process and run_cfg.get("print_model", True):
        print(model)

    with LoraGAContext(model=model, named_grad=named_grad):
        model = get_peft_model(model=model, peft_config=peft_config)

    snapshot_root = Path(run_cfg.get("snapshot_dir", "./snapshot"))
    save_dir = snapshot_root / experiment_name
    if accelerator.is_local_main_process:
        _ensure_directory(save_dir)
        save_loraga_model_init(model=model, save_dir=str(save_dir))
    if accelerator.is_local_main_process:
        print("finish get_peft_model=================================================")

    training_cfg = cfg.get("training", {})
    per_device_batch_size = training_cfg.get("per_device_batch_size", 1)
    real_batch_size = training_cfg.get(
        "real_batch_size",
        per_device_batch_size * accelerator.num_processes,
    )
    max_length = training_cfg.get("max_length", peft_config.max_length)
    training_kwargs = {
        k: v
        for k, v in training_cfg.items()
        if k not in {"per_device_batch_size", "real_batch_size", "max_length"}
    }
    training_kwargs.setdefault("num_process", accelerator.num_processes)
    training_kwargs.setdefault("seed", seed)
    training_kwargs.setdefault("bf16", model_dtype == "bf16")

    model.to(accelerator.device)

    model = train_text_to_text_model(
        run_name=os.path.join(run_cfg.get("result_dir", "peft_test"), experiment_name),
        train_dataset=train_set,
        valid_dataset=val_set,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        per_device_batch_size=per_device_batch_size,
        real_batch_size=real_batch_size,
        max_length=max_length,
        **training_kwargs,
    )

    if accelerator.is_main_process:
        save_loraga_model_final(model=model, save_dir=str(save_dir))
        base_model, _ = initialize_text_to_text_model(
            model_id,
            model_type,
            model_dtype,
            flash_attention=flash_attention,
        )
        loaded = PeftModel.from_pretrained(base_model, str(save_dir))
        print(loaded)
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    Fire(main)
