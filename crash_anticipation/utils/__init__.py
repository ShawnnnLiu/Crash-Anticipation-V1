"""Utility helpers for crash anticipation experiments."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


class AverageMeter:
    """Keep track of the running average of a value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=level,
    )
    return logging.getLogger("crash_anticipation")


def count_parameters(model: torch.nn.Module, only_trainable: bool = True) -> int:
    params: Iterable[torch.nn.Parameter]
    if only_trainable:
        params = (p for p in model.parameters() if p.requires_grad)
    else:
        params = model.parameters()
    return sum(p.numel() for p in params)


__all__ = [
    "set_seed",
    "ensure_dir",
    "move_to_device",
    "save_checkpoint",
    "load_checkpoint",
    "AverageMeter",
    "setup_logging",
    "count_parameters",
]

