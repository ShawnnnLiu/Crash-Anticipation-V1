"""Lightweight streaming crash-anticipation model for embedded deployment.

MobileNetV2 encodes each frame independently; a single-layer GRU integrates
the per-frame embeddings causally. Because the recurrence is causal, the
model supports true streaming inference: at deployment each new frame costs
one MobileNetV2 forward (~1-2 ms on desktop GPU, feasible on automotive SoCs)
instead of re-encoding a 16-frame window as the transformer baseline does.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


class MobileNetTemporal(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        temporal_hidden: int = 256,
        dropout: float = 0.2,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v2(weights=weights, width_mult=width_mult)
        self.features = backbone.features
        feat_dim = backbone.last_channel  # 1280 at width_mult=1.0

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, temporal_hidden),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(temporal_hidden),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden, 1),
        )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, D) frame embeddings."""

        feats = self.features(frames)
        feats = self.pool(feats).flatten(1)
        return self.proj(feats)

    def forward(self, video: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        if video.dim() != 5:
            raise ValueError(f"Video tensor must be 5D but got shape {video.shape}")
        b, t, c, h, w = video.shape
        embeddings = self.encode_frames(video.reshape(b * t, c, h, w)).reshape(b, t, -1)
        seq, _ = self.gru(embeddings)
        features = seq[:, -1]
        logits = self.head(features).squeeze(-1)
        output = {"logits": logits}
        if return_features:
            output["features"] = features
        return output

    @torch.no_grad()
    def forward_stream(
        self,
        frame: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Streaming step: one frame (1, C, H, W) plus GRU state -> (logit, state)."""

        embedding = self.encode_frames(frame).unsqueeze(1)  # (1, 1, D)
        seq, state = self.gru(embedding, state)
        logit = self.head(seq[:, -1]).squeeze(-1)
        return logit, state

    @classmethod
    def from_config(cls, cfg: Any) -> "MobileNetTemporal":
        return cls(
            pretrained=cfg.pretrained,
            temporal_hidden=getattr(cfg, "temporal_hidden", 256),
            dropout=cfg.dropout,
        )
