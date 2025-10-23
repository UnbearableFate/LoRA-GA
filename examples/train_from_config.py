import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping, Optional
from collections import Counter
import numpy as np
import re
import string
import math

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


CLASSIFICATION_DATASETS = {
    "sst2",
    "cola",
    "qqp",
    "mrpc",
    "mnli",
    "emo",
    "qnli",
}
QA_DATASETS = {"squad"}
MATH_DATASETS = {"gsm8k", "meta_math", "meta_math_full", "meta_math_5k"}
INSTRUCTION_DATASETS = {
    "alpaca",
    "alpaca_gpt4",
    "flan",
    "flan_v2",
    "codefeedback",
    "wizard_lm",
}


CLASSIFICATION_METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
    "cola": {
        "metrics": ("accuracy", "matthews_corrcoef"),
        "positive_label": "acceptable",
    },
    "sst2": {"metrics": ("accuracy",), "positive_label": "positive"},
    "qqp": {"metrics": ("accuracy", "f1"), "positive_label": "duplicate"},
    "mrpc": {"metrics": ("accuracy", "f1"), "positive_label": "equivalent"},
    "mnli": {"metrics": ("accuracy",)},
    "emo": {"metrics": ("accuracy",)},
    "qnli": {"metrics": ("accuracy",), "positive_label": "entailment"},
}


def _infer_dataset_category(dataset_name: str) -> str:
    name = dataset_name.lower()
    if name in CLASSIFICATION_DATASETS:
        return "classification"
    if name in QA_DATASETS:
        return "qa"
    if name in MATH_DATASETS:
        return "math"
    if name in INSTRUCTION_DATASETS:
        return "instruction"
    return "generic"


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _simple_normalize(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_label_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return _simple_normalize(text)


def _map_prediction_to_label(
    prediction: str, known_labels: Iterable[str]
) -> Optional[str]:
    if not prediction:
        return None
    normalized = prediction.replace("\n", " ").strip()
    known_list = list(known_labels)
    ordered_labels = sorted(known_list, key=len, reverse=True)
    if normalized in ordered_labels:
        return normalized
    collapsed = normalized.replace(" ", "")
    if collapsed in ordered_labels:
        return collapsed
    tokens = normalized.split()
    if not tokens:
        return None
    for token in tokens:
        if token in ordered_labels:
            return token
        token_collapsed = token.replace(" ", "")
        if token_collapsed in ordered_labels:
            return token_collapsed
    for label in ordered_labels:
        if normalized.startswith(label) or collapsed.startswith(label):
            return label
    padded_normalized = f" {normalized} "
    for label in ordered_labels:
        if f" {label} " in padded_normalized:
            return label
        if label in collapsed:
            return label
    return None


def _compute_accuracy(true_ids: np.ndarray, pred_ids: np.ndarray) -> float:
    if true_ids.size == 0:
        return 0.0
    return float(np.mean(pred_ids == true_ids))


def _binary_confusion(
    true_positive: np.ndarray, pred_positive: np.ndarray
) -> tuple[int, int, int, int]:
    tp = int(np.logical_and(true_positive, pred_positive).sum())
    fp = int(np.logical_and(~true_positive, pred_positive).sum())
    fn = int(np.logical_and(true_positive, ~pred_positive).sum())
    tn = int(np.logical_and(~true_positive, ~pred_positive).sum())
    return tp, tn, fp, fn


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def _binary_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom else 0.0


def _squad_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _decode_predictions(
    pred_ids: np.ndarray, label_ids: np.ndarray, mask: np.ndarray, tokenizer
) -> tuple[list[str], list[str]]:
    pred_array = np.asarray(pred_ids)
    label_array = np.asarray(label_ids)
    mask_array = np.asarray(mask, dtype=bool)
    pred_texts: list[str] = []
    label_texts: list[str] = []
    for pred_row, label_row, mask_row in zip(pred_array, label_array, mask_array):
        if not mask_row.any():
            continue
        pred_tokens = pred_row[mask_row].astype(np.int64).tolist()
        label_tokens = label_row[mask_row].astype(np.int64).tolist()
        pred_texts.append(
            tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
        )
        label_texts.append(
            tokenizer.decode(label_tokens, skip_special_tokens=True).strip()
        )
    return pred_texts, label_texts


def _build_compute_metrics(tokenizer, dataset_name: str):
    category = _infer_dataset_category(dataset_name)

    def compute_metrics(eval_pred):
        predictions_raw = eval_pred.predictions
        if isinstance(predictions_raw, tuple):
            predictions_raw = predictions_raw[0]
        predictions = np.asarray(predictions_raw)
        label_ids = np.asarray(eval_pred.label_ids)
        if predictions.ndim == label_ids.ndim + 1:
            predictions = predictions.argmax(axis=-1)
        mask = label_ids != -100
        total_tokens = int(mask.sum())
        token_matches = int(((predictions == label_ids) & mask).sum())
        token_accuracy = (
            float(token_matches / total_tokens) if total_tokens else 0.0
        )

        metrics: Dict[str, float] = {}
        pred_texts, label_texts = _decode_predictions(
            predictions, label_ids, mask, tokenizer
        )

        if category == "classification":
            metric_cfg = CLASSIFICATION_METRIC_CONFIG.get(
                dataset_name.lower(), {}
            )
            metrics_to_compute = metric_cfg.get("metrics", ("accuracy",))
            positive_label_norm = None
            if metric_cfg.get("positive_label"):
                positive_label_norm = _normalize_label_text(
                    metric_cfg["positive_label"]
                )

            normalized_labels = [
                _normalize_label_text(label) for label in label_texts
            ]
            normalized_predictions = [
                _normalize_label_text(pred) for pred in pred_texts
            ]

            known_labels: list[str] = []
            for label in normalized_labels:
                if label and label not in known_labels:
                    known_labels.append(label)
            if positive_label_norm and positive_label_norm not in known_labels:
                known_labels.append(positive_label_norm)

            label_to_id: Dict[str, int] = {
                label: idx for idx, label in enumerate(known_labels)
            }

            true_ids = np.array(
                [label_to_id.get(label, -1) for label in normalized_labels],
                dtype=np.int64,
            )
            mapped_predictions = [
                _map_prediction_to_label(pred, known_labels)
                if pred
                else None
                for pred in normalized_predictions
            ]
            pred_ids = np.array(
                [
                    label_to_id.get(mapped, -1)
                    if mapped is not None
                    else -1
                    for mapped in mapped_predictions
                ],
                dtype=np.int64,
            )
            valid_mask = true_ids != -1
            true_ids = true_ids[valid_mask]
            pred_ids = pred_ids[valid_mask]
            if true_ids.size and pred_ids.size:
                unknown_pred_mask = pred_ids == -1
                if unknown_pred_mask.any():
                    pred_ids = pred_ids.copy()
                    pred_ids[unknown_pred_mask] = -999

            if "accuracy" in metrics_to_compute or not metrics_to_compute:
                metrics["accuracy"] = _compute_accuracy(true_ids, pred_ids)

            positive_id = None
            if positive_label_norm is not None:
                positive_id = label_to_id.get(positive_label_norm)

            needs_binary_metrics = any(
                metric in metrics_to_compute
                for metric in ("f1", "matthews_corrcoef")
            )
            binary_counts = None
            if (
                needs_binary_metrics
                and positive_id is not None
                and positive_id >= 0
                and true_ids.size
                and pred_ids.size
            ):
                true_positive_mask = true_ids == positive_id
                pred_positive_mask = pred_ids == positive_id
                binary_counts = _binary_confusion(
                    true_positive_mask, pred_positive_mask
                )

            if "f1" in metrics_to_compute:
                if binary_counts is not None:
                    tp, tn, fp, fn = binary_counts
                    metrics["f1"] = _binary_f1(tp, fp, fn)
                else:
                    metrics["f1"] = 0.0

            if "matthews_corrcoef" in metrics_to_compute:
                if binary_counts is not None:
                    tp, tn, fp, fn = binary_counts
                    metrics["matthews_corrcoef"] = _binary_mcc(
                        tp, tn, fp, fn
                    )
                else:
                    metrics["matthews_corrcoef"] = 0.0

            metrics["token_accuracy"] = token_accuracy
        else:
            if category == "qa":
                exact = [
                    _normalize_answer(pred) == _normalize_answer(label)
                    for pred, label in zip(pred_texts, label_texts)
                ]
                metrics["exact_match"] = (
                    float(np.mean(exact)) if exact else 0.0
                )
                f1_scores = [
                    _squad_f1(pred, label)
                    for pred, label in zip(pred_texts, label_texts)
                ]
                metrics["f1"] = float(np.mean(f1_scores)) if f1_scores else 0.0
            elif category == "math":
                exact = [
                    _simple_normalize(pred) == _simple_normalize(label)
                    for pred, label in zip(pred_texts, label_texts)
                ]
                metrics["exact_match"] = (
                    float(np.mean(exact)) if exact else 0.0
                )
                metrics["token_accuracy"] = token_accuracy
            else:
                metrics["token_accuracy"] = token_accuracy
        return metrics

    return compute_metrics


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

    print(f"Using {loraga_config_kwargs} to initialize LoRA")
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
    report_to = None
    if wandb_enabled and accelerator.is_main_process:
        wandb.init(
            name=wandb_name,
            project=wandb_cfg.get("project", "LoRA-GA in PEFT"),
            group=wandb_cfg.get("group"),
            mode=wandb_cfg.get("mode", "offline"),
        )
        report_to = "wandb"

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

    #if accelerator.is_local_main_process and run_cfg.get("print_model", True):
    #    print(model)
    lora_init_start_time = time.time()
    model.to("cuda:0")
    with LoraGAContext(model=model, named_grad=named_grad):
        model = get_peft_model(model=model, peft_config=peft_config)
    print(f"{loraga_config_kwargs['init_lora_weights']} initialization for model {model_id} took {time.time() - lora_init_start_time:.2f} seconds")
    print(f"max cuda memory allocated after LoRA-GA init: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

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

    compute_metrics_fn = _build_compute_metrics(tokenizer, dataset_name)

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
        report_to=report_to,
        compute_metrics_fn=compute_metrics_fn,
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
