"""Dataset loading and preprocessing helpers."""

from __future__ import annotations

import logging
import math
import re
import string
from dataclasses import dataclass
import sys
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)

sys.path.append("../")
from config import DatasetConfig, TrainingConfig

LOGGER = logging.getLogger(__name__)


GLUE_TASK_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "cola": {
        "hf_path": ("glue", "cola"),
        "instruction": "Classify the grammatical acceptability of the sentence.",
        "format": lambda ex: ex["sentence"],
        "labels": {0: "unacceptable", 1: "acceptable"},
        "metrics": ("accuracy", "matthews_corrcoef"),
    },
    "sst2": {
        "hf_path": ("glue", "sst2"),
        "instruction": "Classify the sentiment of the sentence.",
        "format": lambda ex: ex["sentence"],
        "labels": {0: "negative", 1: "positive"},
        "metrics": ("accuracy",),
    },
    "mrpc": {
        "hf_path": ("glue", "mrpc"),
        "instruction": "Determine if the two sentences are semantically equivalent.",
        "format": lambda ex: "\n".join([ex["sentence1"], ex["sentence2"]]),
        "labels": {0: "different", 1: "equivalent"},
        "metrics": ("accuracy", "f1"),
    },
    "qqp": {
        "hf_path": ("glue", "qqp"),
        "instruction": "Decide if the two questions are duplicates.",
        "format": lambda ex: "\n".join([ex["question1"], ex["question2"]]),
        "labels": {0: "different", 1: "duplicate"},
        "metrics": ("accuracy", "f1"),
    },
    "mnli": {
        "hf_path": ("glue", "mnli"),
        "instruction": "Classify the relationship between the premise and hypothesis.",
        "format": lambda ex: "\n".join([f"Premise: {ex['premise']}", f"Hypothesis: {ex['hypothesis']}"]),
        "labels": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "metrics": ("accuracy",),
        "validation_split": "validation_matched",
    },
    "qnli": {
        "hf_path": ("glue", "qnli"),
        "instruction": "Determine whether the answer sentence entails the question.",
        "format": lambda ex: "\n".join([f"Question: {ex['question']}", f"Sentence: {ex['sentence']}"]),
        "labels": {0: "entailment", 1: "not_entailment"},
        "metrics": ("accuracy",),
    },
    "rte": {
        "hf_path": ("glue", "rte"),
        "instruction": "Classify whether the hypothesis is entailed by the premise.",
        "format": lambda ex: "\n".join([f"Premise: {ex['premise']}", f"Hypothesis: {ex['hypothesis']}"]),
        "labels": {0: "not_entails", 1: "entails"},
        "metrics": ("accuracy",),
    },
    "wnli": {
        "hf_path": ("glue", "wnli"),
        "instruction": "Determine whether the hypothesis is entailed by the premise.",
        "format": lambda ex: "\n".join([f"Premise: {ex['sentence1']}", f"Hypothesis: {ex['sentence2']}"]),
        "labels": {0: "not_entailment", 1: "entailment"},
        "metrics": ("accuracy",),
    },
    "stsb": {
        "hf_path": ("glue", "stsb"),
        "instruction": "Rate the semantic similarity of the sentence pair on a scale between 0 and 5.",
        "format": lambda ex: "\n".join([f"Sentence 1: {ex['sentence1']}", f"Sentence 2: {ex['sentence2']}"]),
        "labels": None,
        "metrics": ("pearson", "spearmanr"),
    },
}


MATH_DATASETS: Dict[str, Dict[str, object]] = {
    "gsm8k": {
        "hf_path": ("gsm8k", "main"),
        "question_field": "question",
        "answer_field": "answer",
        "instruction": "Solve the following math problem and provide the final numerical answer.",
    },
    "meta_math": {
        "hf_path": ("meta-math/MetaMathQA", None),
        "question_field": "problem",
        "answer_field": "solution",
        "instruction": "Solve the following math problem and provide the final answer.",
    },
    "meta_math_full": {
        "hf_path": ("meta-math/MetaMathQA", "full"),
        "question_field": "problem",
        "answer_field": "solution",
        "instruction": "Solve the following math problem and provide the final answer.",
    },
    "meta_math_5k": {
        "hf_path": ("meta-math/MetaMathQA", "5k"),
        "question_field": "problem",
        "answer_field": "solution",
        "instruction": "Solve the following math problem and provide the final answer.",
    },
}


DEFAULT_CLASSIFICATION_METRICS = ("accuracy",)


@dataclass
class DatasetBundle:
    """Container aggregating processed dataset artifacts."""

    train_dataset: Dataset
    eval_dataset: Dataset
    data_collator: Callable
    eval_prompts: List[str]
    eval_targets: List[str]
    task_category: str
    label_list: Optional[List[str]] = None
    metric_config: Optional[Dict[str, object]] = None

    def gradient_subset(self, subset_size: Optional[int]) -> Dataset:
        """Return a dataset subset (or the full train set)."""

        if subset_size is None or subset_size <= 0:
            return self.train_dataset
        subset_size = min(subset_size, len(self.train_dataset))
        return self.train_dataset.select(range(subset_size))


class CausalLMCollator(DataCollatorWithPadding):
    """Collator that pads inputs and labels for causal LM finetuning."""

    def __call__(self, features):
        labels = [feature["labels"] for feature in features]
        for feature in features:
            feature.pop("labels", None)
            feature.pop("prompt", None)
            feature.pop("target", None)
        batch = super().__call__(features)
        labels_padded = self._pad_labels(labels, batch["input_ids"].shape[-1])
        batch["labels"] = labels_padded
        return batch

    def _pad_labels(self, labels: List[List[int]], max_length: int):
        import torch

        padding_side = getattr(self.tokenizer, "padding_side", "right")
        truncation_side = getattr(self.tokenizer, "truncation_side", "right")
        padded = []
        for label in labels:
            if len(label) < max_length:
                pad_length = max_length - len(label)
                padding = [-100] * pad_length
                if padding_side == "left":
                    label = padding + label
                else:
                    label = label + padding
            else:
                if truncation_side == "left":
                    label = label[-max_length:]
                else:
                    label = label[:max_length]
            padded.append(label)
        return torch.tensor(padded, dtype=torch.long)


def build_dataset(
    dataset_cfg: DatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    training_cfg: TrainingConfig,
) -> DatasetBundle:
    """Load dataset, apply prompt templates, and tokenize examples."""

    task_name = dataset_cfg.name.lower()
    LOGGER.info("Loading dataset: %s", task_name)

    if dataset_cfg.task:
        task_category = dataset_cfg.task.lower()
    elif task_name in GLUE_TASK_DEFINITIONS:
        task_category = "glue"
    elif task_name in MATH_DATASETS:
        task_category = "math"
    else:
        task_category = "generic"

    if task_category == "glue":
        dataset, eval_meta = _load_glue_dataset(task_name, dataset_cfg)
    elif task_category == "math":
        dataset, eval_meta = _load_math_dataset(task_name, dataset_cfg)
    else:
        dataset, eval_meta = _load_generic_dataset(dataset_cfg)

    formatter = _build_prompt_formatter(task_category, dataset_cfg, eval_meta)
    train_processed = dataset[dataset_cfg.split_train].map(formatter)
    eval_processed = dataset[dataset_cfg.split_eval].map(formatter)

    if dataset_cfg.max_train_samples:
        train_processed = train_processed.select(range(min(dataset_cfg.max_train_samples, len(train_processed))))
    if dataset_cfg.max_eval_samples:
        eval_processed = eval_processed.select(range(min(dataset_cfg.max_eval_samples, len(eval_processed))))

    eval_prompts = eval_processed["prompt"]
    eval_targets = eval_processed["target"]

    tokeniser_fn = _build_tokenizer_function(tokenizer, training_cfg.max_length)
    remove_cols = [col for col in train_processed.column_names if col not in {"prompt", "target"}]
    train_tokenised = train_processed.map(
        tokeniser_fn,
        batched=True,
        remove_columns=remove_cols,
    )
    eval_tokenised = eval_processed.map(
        tokeniser_fn,
        batched=True,
        remove_columns=remove_cols,
    )

    collator = CausalLMCollator(tokenizer=tokenizer, padding="longest")

    label_list = eval_meta.get("label_list")
    metric_config = {
        "metrics": eval_meta.get("metrics", DEFAULT_CLASSIFICATION_METRICS),
        "positive_label": eval_meta.get("positive_label"),
    } if eval_meta else None

    return DatasetBundle(
        train_dataset=train_tokenised,
        eval_dataset=eval_tokenised,
        data_collator=collator,
        eval_prompts=eval_prompts,
        eval_targets=eval_targets,
        task_category=task_category,
        label_list=label_list,
        metric_config=metric_config,
    )


def _load_glue_dataset(task_name: str, dataset_cfg: DatasetConfig) -> Tuple[DatasetDict, Dict[str, object]]:
    definition = GLUE_TASK_DEFINITIONS[task_name]
    hf_path = definition["hf_path"]
    subset = dataset_cfg.subset or hf_path[1]
    dataset = load_dataset(hf_path[0], subset)
    validation_split = definition.get("validation_split")
    if validation_split:
        dataset_cfg.split_eval = validation_split  # update in place for downstream usage
    label_map = definition.get("labels")
    label_list = None
    if isinstance(label_map, dict):
        label_list = [label_map[key] for key in sorted(label_map)]
    metrics = definition.get("metrics", DEFAULT_CLASSIFICATION_METRICS)
    return dataset, {"definition": definition, "label_list": label_list, "metrics": metrics}


def _load_math_dataset(task_name: str, dataset_cfg: DatasetConfig) -> Tuple[DatasetDict, Dict[str, object]]:
    meta = MATH_DATASETS[task_name]
    hf_path, subset = meta["hf_path"]
    dataset = load_dataset(hf_path, subset) if subset else load_dataset(hf_path)
    return dataset, meta


def _load_generic_dataset(dataset_cfg: DatasetConfig) -> Tuple[DatasetDict, Dict[str, object]]:
    dataset = load_dataset(dataset_cfg.name, dataset_cfg.subset) if dataset_cfg.subset else load_dataset(dataset_cfg.name)
    return dataset, {}


def _build_prompt_formatter(
    task_category: str,
    dataset_cfg: DatasetConfig,
    meta: Dict[str, object],
) -> Callable[[Dict[str, object]], Dict[str, str]]:
    if task_category == "glue":
        definition = meta["definition"]
        instruction = dataset_cfg.instruction_template or definition["instruction"]
        label_map = definition.get("labels")
        formatter = definition["format"]

        def glue_formatter(example: Dict[str, object]) -> Dict[str, str]:
            prompt = f"{instruction}\n{formatter(example)}\nAnswer:"
            label = example["label"]
            target = label_map[label] if isinstance(label_map, dict) and label in label_map else str(label)
            return {"prompt": prompt, "target": f" {target}"}

        return glue_formatter

    if task_category == "math":
        instruction = dataset_cfg.instruction_template or meta["instruction"]
        question_field = meta["question_field"]
        answer_field = meta["answer_field"]

        def math_formatter(example: Dict[str, object]) -> Dict[str, str]:
            prompt = f"{instruction}\nProblem: {example[question_field]}\nAnswer:"
            target_text = example[answer_field]
            final_answer = _extract_final_answer(target_text)
            return {"prompt": prompt, "target": f" {final_answer}"}

        return math_formatter

    instruction = dataset_cfg.instruction_template or "Respond to the following input."

    def default_formatter(example: Dict[str, object]) -> Dict[str, str]:
        text_fields = [str(value) for value in example.values() if isinstance(value, str)]
        source = "\n".join(text_fields) if text_fields else str(example)
        prompt = f"{instruction}\nInput:\n{source}\nAnswer:"
        target = str(example.get("label", example.get("answer", "")))
        return {"prompt": prompt, "target": f" {target}"}

    return default_formatter


def _build_tokenizer_function(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Callable[[Dict[str, List[str]]], Dict[str, List[List[int]]]]:
    eos = tokenizer.eos_token or tokenizer.sep_token or ""

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        prompts = batch["prompt"]
        targets = batch["target"]
        concatenated = [prompt + target + eos for prompt, target in zip(prompts, targets)]
        model_inputs = tokenizer(
            concatenated,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        prompt_tokenised = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        labels: List[List[int]] = []
        for idx, input_ids in enumerate(model_inputs["input_ids"]):
            prompt_ids = prompt_tokenised["input_ids"][idx]
            prompt_len = len(prompt_ids)
            label_ids = input_ids.copy()
            label_ids[:prompt_len] = [-100] * min(prompt_len, len(label_ids))
            labels.append(label_ids)
        model_inputs["labels"] = labels
        return model_inputs

    return tokenize_batch


def _extract_final_answer(solution: str) -> str:
    """Extract final numeric answer when possible."""

    if not solution:
        return ""
    match = re.findall(r"(-?\d+(?:\.\d+)?)", solution)
    if match:
        return match[-1]
    return solution.strip()


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for robust comparisons."""

    text = text.lower()
    text = text.strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.sub(r"\s+", " ", text)


def compute_classification_metrics(
    predicted: Iterable[str],
    references: Iterable[str],
    label_list: Optional[List[str]] = None,
    metrics: Tuple[str, ...] = DEFAULT_CLASSIFICATION_METRICS,
) -> Dict[str, float]:
    """Compute standard classification metrics from decoded predictions."""

    predicted = list(predicted)
    references = list(references)
    resolved_predictions = list(predicted)
    if label_list:
        resolved_predictions = [_resolve_label(pred, label_list) for pred in resolved_predictions]
        label_map = {normalize_text(name): idx for idx, name in enumerate(label_list)}
    else:
        label_map = {}

    def map_to_ids(items: List[str]) -> np.ndarray:
        ids = []
        for item in items:
            normalized = normalize_text(item)
            if label_map:
                ids.append(label_map.get(normalized, -1))
            else:
                ids.append(item)
        return np.array(ids)

    preds_ids = map_to_ids(resolved_predictions)
    refs_ids = map_to_ids(references)

    metrics_out: Dict[str, float] = {}
    if "accuracy" in metrics:
        mask = refs_ids != -1 if label_map else np.ones_like(refs_ids, dtype=bool)
        correct = (preds_ids == refs_ids) & mask
        accuracy = correct.sum() / max(mask.sum(), 1)
        metrics_out["accuracy"] = float(accuracy)

    if "f1" in metrics and label_list and len(label_list) == 2:
        positive_id = 1
        precision, recall, f1 = _binary_metrics(refs_ids, preds_ids, positive_id)
        metrics_out.update({"precision": precision, "recall": recall, "f1": f1})

    if "matthews_corrcoef" in metrics and label_map:
        try:
            from sklearn.metrics import matthews_corrcoef
        except ImportError:
            LOGGER.warning("scikit-learn not installed, skipping matthews_corrcoef metric.")
        else:
            valid = (refs_ids != -1) & (preds_ids != -1)
            if valid.any():
                metrics_out["matthews_corrcoef"] = float(matthews_corrcoef(refs_ids[valid], preds_ids[valid]))
            else:
                metrics_out["matthews_corrcoef"] = 0.0

    if "pearson" in metrics or "spearmanr" in metrics:
        try:
            from scipy.stats import pearsonr, spearmanr
        except ImportError:
            LOGGER.warning("scipy not installed, skipping correlation metrics.")
        else:
            preds_float = np.array([_safe_float(x) for x in resolved_predictions], dtype=float)
            refs_float = np.array([_safe_float(x) for x in references], dtype=float)
            if "pearson" in metrics:
                metrics_out["pearson"] = float(pearsonr(preds_float, refs_float)[0])
            if "spearmanr" in metrics:
                metrics_out["spearmanr"] = float(spearmanr(preds_float, refs_float)[0])

    return metrics_out


def compute_math_metrics(predicted: Iterable[str], references: Iterable[str]) -> Dict[str, float]:
    """Exact match style scoring for math datasets."""

    predicted_norm = [normalize_text(pred) for pred in predicted]
    references_norm = [normalize_text(ref) for ref in references]
    total = len(references_norm)
    correct = sum(int(p == r) for p, r in zip(predicted_norm, references_norm))
    return {"exact_match": float(correct / max(total, 1))}


def _binary_metrics(true_ids: np.ndarray, pred_ids: np.ndarray, positive_id: int) -> Tuple[float, float, float]:
    true_pos = (true_ids == positive_id)
    pred_pos = (pred_ids == positive_id)
    tp = float(np.logical_and(true_pos, pred_pos).sum())
    fp = float(np.logical_and(~true_pos, pred_pos).sum())
    fn = float(np.logical_and(true_pos, ~pred_pos).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def _resolve_label(prediction: str, label_list: List[str]) -> str:
    normalized_prediction = normalize_text(prediction)
    normalized_map = {normalize_text(label): label for label in label_list}
    if normalized_prediction in normalized_map:
        return normalized_map[normalized_prediction]
    for label in label_list:
        normalized_label = normalize_text(label)
        if normalized_label and normalized_label in normalized_prediction:
            return label
    return prediction.strip()


__all__ = [
    "DatasetBundle",
    "build_dataset",
    "compute_classification_metrics",
    "compute_math_metrics",
    "normalize_text",
]
