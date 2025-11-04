"""Data utilities for crash anticipation."""

from __future__ import annotations

from typing import Any, Tuple

from torch.utils.data import DataLoader

from .datasets.ccd import CCDDataset, build_dataloaders as build_ccd_dataloaders
from .datasets.dad import DADDataset, build_dataloaders as build_dad_dataloaders


def build_dataloaders(
    data_config: Any,
    batch_size: int,
    is_distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    dataset_type = getattr(data_config, "dataset_type", "dad").lower()
    if dataset_type == "ccd":
        return build_ccd_dataloaders(data_config, batch_size, is_distributed)
    return build_dad_dataloaders(data_config, batch_size, is_distributed)


__all__ = [
    "build_dataloaders",
    "DADDataset",
    "CCDDataset",
]

