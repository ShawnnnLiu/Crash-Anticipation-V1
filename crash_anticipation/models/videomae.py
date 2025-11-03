"""VideoMAE baseline model for crash anticipation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import timm
import torch
from torch import nn


class VideoMAEAnticipation(nn.Module):
    """VideoMAE backbone with a lightweight binary anticipation head."""

    def __init__(
        self,
        backbone: str = "videomae_small_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone_layers: int = 0,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )

        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"Backbone {backbone} does not expose num_features")

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

        if freeze_backbone_layers > 0:
            self.freeze_backbone(freeze_backbone_layers)

    def forward(
        self,
        video: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Input expected as (B, T, C, H, W); convert to (B, C, T, H, W)
        if video.dim() != 5:
            raise ValueError(f"Video tensor must be 5D but got shape {video.shape}")

        backbone_input = video.permute(0, 2, 1, 3, 4).contiguous()
        features = self.backbone(backbone_input)
        logits = self.head(features).squeeze(-1)
        output = {"logits": logits}
        if return_features:
            output["features"] = features
        return output

    def freeze_backbone(self, num_blocks: int) -> None:
        """Freeze the earliest transformer blocks of the backbone."""

        if not hasattr(self.backbone, "blocks"):
            return

        blocks = getattr(self.backbone, "blocks")
        if not isinstance(blocks, (list, nn.ModuleList)):
            return

        num_blocks = min(num_blocks, len(blocks))
        for block in blocks[:num_blocks]:
            for param in block.parameters():
                param.requires_grad = False

    @classmethod
    def from_config(cls, cfg: Any) -> "VideoMAEAnticipation":
        return cls(
            backbone=cfg.backbone,
            pretrained=cfg.pretrained,
            dropout=cfg.dropout,
            freeze_backbone_layers=cfg.freeze_backbone_layers,
        )

    def configure_optim_parameters(self) -> list[Dict[str, Any]]:
        """Separate parameter groups for backbone and head if needed."""

        return [
            {"params": self.backbone.parameters(), "name": "backbone"},
            {"params": self.head.parameters(), "name": "head"},
        ]

