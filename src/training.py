"""High-level training orchestration for PEFT fine-tuning with Hugging Face Trainer."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from peft import LoraGAConfig, TaskType, get_peft_model
from peft.utils.lora_ga_utils import LoraGAContext, estimate_gradient, save_loraga_model_final, save_loraga_model_init

from config import ExperimentConfig, ModelConfig, TrainingConfig
from data import DatasetBundle, build_dataset
from data.builders import compute_classification_metrics, compute_math_metrics
from utils import (
    count_trainable_parameters,
    ensure_dir,
    find_all_linear_module_names,
    seed_everything,
    setup_logging,
    unwrap_lora_target_modules,
)

LOGGER = logging.getLogger(__name__)


def run_training(config: ExperimentConfig) -> Dict[str, float]:
    """Entry point to train a PEFT model based on the supplied configuration."""

    setup_logging()
    if config.run.seed is not None:
        seed_everything(config.run.seed)

    accelerator = Accelerator()
    tokenizer = _load_tokenizer(config.model)
    dataset_bundle = build_dataset(config.dataset, tokenizer, config.training)

    model = _load_model(config.model, tokenizer)
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    peft_applied = False
    if config.loraga is not None:
        model = _prepare_loraga(
            model=model,
            tokenizer=tokenizer,
            dataset_bundle=dataset_bundle,
            accelerator=accelerator,
            config=config,
        )
        peft_applied = True

    LOGGER.info("Trainable parameters: %s", f"{count_trainable_parameters(model):,}")

    training_args = _create_training_arguments(config, tokenizer)
    callbacks = _build_callbacks(config.training)

    compute_metrics_fn = _build_compute_metrics(tokenizer, dataset_bundle)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_bundle.train_dataset,
        eval_dataset=dataset_bundle.eval_dataset,
        data_collator=dataset_bundle.data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )

    if config.wandb.enabled:
        _log_wandb_config(config)

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_state()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if dataset_bundle.eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        metrics.update(eval_metrics)

    if peft_applied and config.run.snapshot_dir:
        snapshot_dir = ensure_dir(config.run.snapshot_dir)
        save_loraga_model_final(trainer.model, str(snapshot_dir))
        LOGGER.info("Saved LoRA final weights to %s", snapshot_dir)

    if config.wandb.enabled:
        try:
            import wandb

            wandb.finish()
        except ImportError:
            pass

    return metrics


def _load_tokenizer(model_cfg: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer_id = model_cfg.id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=model_cfg.revision,
        use_fast=True,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    return tokenizer


def _load_model(model_cfg: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(model_cfg.dtype.lower(), torch.float32)

    common_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_cfg.id,
        "revision": model_cfg.revision,
        "trust_remote_code": model_cfg.trust_remote_code,
        "use_safetensors": model_cfg.use_safetensors,
        "torch_dtype": torch_dtype,
    }

    if model_cfg.flash_attention:
        common_kwargs["attn_implementation"] = "flash_attention_2"

    model_type = model_cfg.type.lower()
    if model_type == "causallm":
        model = AutoModelForCausalLM.from_pretrained(**common_kwargs)
    elif model_type in {"seq2seqlm", "conditionalgeneration"}:
        model = AutoModelForSeq2SeqLM.from_pretrained(**common_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_cfg.type}")

    if tokenizer.pad_token and getattr(model, "config", None):
        model.resize_token_embeddings(len(tokenizer))

    return model


def _prepare_loraga(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset_bundle: DatasetBundle,
    accelerator: Accelerator,
    config: ExperimentConfig,
) -> PreTrainedModel:
    loraga_cfg = config.loraga
    if loraga_cfg is None:
        return model

    subset_size = loraga_cfg.sample_size or loraga_cfg.gradient.subset_size
    gradient_dataset = dataset_bundle.gradient_subset(subset_size)
    gradient_batch_size = loraga_cfg.bsz or config.training.per_device_batch_size
    gradient_loader = DataLoader(
        gradient_dataset,
        batch_size=gradient_batch_size,
        shuffle=False,
        collate_fn=dataset_bundle.data_collator,
    )

    if loraga_cfg.gradient.save_path and accelerator.is_main_process:
        ensure_dir(Path(loraga_cfg.gradient.save_path))

    grad_save_filename = f"grad_save_{config.model.id.replace('/', '-')}_{config.dataset.name}.pt"
    named_grad = estimate_gradient(
        model=model,
        dataloader=gradient_loader,
        accelerator=accelerator,
        quant_flag=loraga_cfg.gradient.quant_flag,
        origin_type=loraga_cfg.gradient.origin_type,
        quant_type=loraga_cfg.gradient.quant_type,
        no_split_module_classes=None,
        grad_save_path=Path(loraga_cfg.gradient.save_path, grad_save_filename) if loraga_cfg.gradient.save_path else None,
    )

    task_type = _infer_task_type(config.model.type)
    target_modules = unwrap_lora_target_modules(loraga_cfg.target_modules)
    if not target_modules:
        target_modules = find_all_linear_module_names(model)
        LOGGER.info("Auto-detected target modules for LoRA: %s", target_modules)

    peft_config = LoraGAConfig(
        r=loraga_cfg.r,
        lora_alpha=loraga_cfg.lora_alpha,
        lora_dropout=loraga_cfg.lora_dropout,
        bias=loraga_cfg.bias,
        target_modules=target_modules,
        modules_to_save=loraga_cfg.modules_to_save,
        init_lora_weights=loraga_cfg.init_lora_weights,
        task_type=task_type,
        inference_mode=loraga_cfg.inference_mode,
    )

    with LoraGAContext(model=model, named_grad=named_grad):
        model = get_peft_model(model=model, peft_config=peft_config)

    model.print_trainable_parameters()

    if config.run.snapshot_dir:
        save_loraga_model_init(model, config.run.snapshot_dir)

    return model


def _infer_task_type(model_type: str) -> TaskType:
    model_type = model_type.lower()
    if model_type == "causallm":
        return TaskType.CAUSAL_LM
    if model_type in {"seq2seqlm", "conditionalgeneration"}:
        return TaskType.SEQ_2_SEQ_LM
    raise ValueError(f"Cannot infer PEFT task type from model type: {model_type}")


def _create_training_arguments(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> Seq2SeqTrainingArguments:
    training_cfg = config.training
    output_root = config.run.result_dir or config.run.output_dir
    run_name = config.run.name or "peft-run"
    output_dir = ensure_dir(Path(output_root) / run_name)

    gradient_accumulation = training_cfg.gradient_accumulation_steps
    if gradient_accumulation is None and training_cfg.real_batch_size:
        gradient_accumulation = max(
            training_cfg.real_batch_size // max(training_cfg.per_device_batch_size, 1),
            1,
        )

    eval_strategy = training_cfg.eval_strategy
    if training_cfg.eval_steps:
        eval_strategy = "steps"
    elif eval_strategy is None:
        eval_strategy = "no"

    save_strategy = training_cfg.save_strategy
    if training_cfg.save_steps:
        save_strategy = "steps"
    elif save_strategy is None:
        save_strategy = "epoch"

    logging_strategy = training_cfg.logging_strategy or "steps"

    args_dict = dict(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg.num_train_epochs,
        per_device_train_batch_size=training_cfg.per_device_batch_size,
        per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        warmup_ratio=training_cfg.warmup_ratio,
        logging_steps=training_cfg.logging_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        save_steps=training_cfg.save_steps,
        eval_steps=training_cfg.eval_steps,
        save_total_limit=training_cfg.save_total_limit,
        gradient_accumulation_steps=gradient_accumulation,
        max_grad_norm=training_cfg.max_grad_norm,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        predict_with_generate=training_cfg.predict_with_generate,
        generation_max_length=training_cfg.generation_max_length,
        generation_num_beams=training_cfg.generation_num_beams,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
        fp16=training_cfg.fp16,
        bf16=training_cfg.bf16,
        #tf32=training_cfg.tf32,
        dataloader_num_workers=training_cfg.dataloader_num_workers,
        optim=training_cfg.optim,
        warmup_steps=training_cfg.warmup_steps,
        logging_first_step=training_cfg.logging_first_step,
        load_best_model_at_end=training_cfg.load_best_model_at_end,
        metric_for_best_model=training_cfg.metric_for_best_model,
        greater_is_better=training_cfg.greater_is_better,
        report_to=["wandb"] if config.wandb.enabled else [],
    )

    args_dict.update({k: v for k, v in training_cfg.training_args.items() if v is not None})

    if not config.wandb.enabled:
        args_dict["report_to"] = []

    training_args = Seq2SeqTrainingArguments(**{k: v for k, v in args_dict.items() if v is not None})

    return training_args


def _build_callbacks(training_cfg: TrainingConfig) -> Optional[List[Any]]:
    callbacks: List[Any] = []
    if training_cfg.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_cfg.early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        )
    return callbacks or None


def _build_compute_metrics(tokenizer: PreTrainedTokenizerBase, dataset_bundle: DatasetBundle):
    prompts = dataset_bundle.eval_prompts
    targets = dataset_bundle.eval_targets
    label_list = dataset_bundle.label_list or []
    task_category = dataset_bundle.task_category
    metric_config = dataset_bundle.metric_config or {}

    def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        predictions, label_ids = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        reference_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cleaned_predictions = [_strip_prompt(pred, prompt) for pred, prompt in zip(pred_texts, prompts)]
        cleaned_references = [target.strip() for target in targets]
        if len(reference_texts) == len(cleaned_references):
            cleaned_references = [ref.strip() for ref in cleaned_references]

        if task_category in {"glue", "classification"}:
            metrics = compute_classification_metrics(
                predicted=cleaned_predictions,
                references=cleaned_references,
                label_list=label_list or None,
                metrics=metric_config.get("metrics", ()),
            )
        elif task_category == "math":
            metrics = compute_math_metrics(
                predicted=cleaned_predictions,
                references=cleaned_references,
            )
        else:
            metrics = {"exact_match": float(np.mean([pred == ref for pred, ref in zip(cleaned_predictions, cleaned_references)]))}

        LOGGER.info("Eval metrics: %s", metrics)
        return metrics

    return compute_metrics


def _strip_prompt(prediction: str, prompt: str) -> str:
    prediction = prediction.strip()
    prompt = prompt.strip()
    if prediction.startswith(prompt):
        return prediction[len(prompt) :].strip()
    return prediction


def _log_wandb_config(config: ExperimentConfig) -> None:
    try:
        import wandb
    except ImportError:
        LOGGER.warning("wandb not installed; skipping logging despite configuration.")
        return

    if config.wandb.mode:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            tags=config.wandb.tags,
            config=_wandb_config_dict(config),
            mode=config.wandb.mode,
            reinit=True,
        )
    else:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            tags=config.wandb.tags,
            config=_wandb_config_dict(config),
            reinit=True,
        )


def _wandb_config_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return {
        "run": asdict(config.run),
        "model": asdict(config.model),
        "dataset": asdict(config.dataset),
        "loraga": asdict(config.loraga) if config.loraga else None,
        "training": asdict(config.training),
    }

__all__ = ["run_training"]
