"""Utility helpers shared across the training stack."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


def setup_logging(verbosity: int = logging.INFO) -> None:
    """Configure basic logging suitable for CLI usage."""

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=verbosity,
    )


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def find_all_linear_module_names(
    model: torch.nn.Module,
    forbidden_modules: Sequence[str] = ("lm_head", "embed_tokens"),
) -> List[str]:
    """Collect module names that are instances of :class:`torch.nn.Linear` excluding forbidden suffixes."""

    linear_cls = torch.nn.Linear
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            suffix = name.split(".")[-1]
            if any(suffix.endswith(forbidden) or forbidden in name for forbidden in forbidden_modules):
                continue
            module_names.add(suffix)
    return sorted(module_names)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unwrap_lora_target_modules(modules: Iterable[str] | None) -> List[str] | None:
    """Return unique target module names preserving user provided order."""

    if modules is None:
        return None
    seen = set()
    ordered: List[str] = []
    for item in modules:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


__all__ = [
    "count_trainable_parameters",
    "ensure_dir",
    "find_all_linear_module_names",
    "seed_everything",
    "setup_logging",
    "unwrap_lora_target_modules",
]

