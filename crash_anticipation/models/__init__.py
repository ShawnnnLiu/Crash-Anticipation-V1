"""Model factory for crash anticipation."""

from __future__ import annotations

from typing import Any

from torch import nn

from .mobilenet_temporal import MobileNetTemporal
from .videomae import VideoMAEAnticipation


def build_model(model_cfg: Any) -> nn.Module:
    model_type = getattr(model_cfg, "type", "videomae").lower()
    if model_type == "videomae":
        return VideoMAEAnticipation.from_config(model_cfg)
    if model_type in ("mobilenet_gru", "mobilenet_temporal"):
        return MobileNetTemporal.from_config(model_cfg)
    raise ValueError(f"Unknown model type: {model_type}")


__all__ = ["VideoMAEAnticipation", "MobileNetTemporal", "build_model"]
