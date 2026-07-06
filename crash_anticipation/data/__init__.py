"""Data utilities for crash anticipation."""

from __future__ import annotations

from typing import Any, Tuple

from torch.utils.data import DataLoader

from .datasets.windows import (
    AnticipationWindowDataset,
    VideoRecord,
    build_dataloaders as build_window_dataloaders,
    load_ccd_records,
    load_dad_negative_records,
)


def build_dataloaders(
    data_config: Any,
    batch_size: int,
    is_distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    dataset_type = getattr(data_config, "dataset_type", "windows").lower()
    if dataset_type != "windows":
        raise ValueError(
            f"Unsupported dataset_type '{dataset_type}'. The project uses the "
            "windowed anticipation formulation (dataset_type: windows)."
        )
    return build_window_dataloaders(data_config, batch_size, is_distributed)


__all__ = [
    "build_dataloaders",
    "AnticipationWindowDataset",
    "VideoRecord",
    "load_ccd_records",
    "load_dad_negative_records",
]
