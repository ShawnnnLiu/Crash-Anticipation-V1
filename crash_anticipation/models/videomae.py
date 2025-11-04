"""VideoMAE baseline model for crash anticipation."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class VideoMAEHuggingFaceBackbone(nn.Module):
    """Wrap the Hugging Face VideoMAE encoder to mimic timm's interface."""

    def __init__(self, hf_name: str) -> None:
        super().__init__()
        try:
            from transformers import VideoMAEModel
        except ImportError as err:  # pragma: no cover - dependency guard
            raise ImportError(
                "transformers is required when model.provider='huggingface'. "
                "Install the optional dependencies or switch the provider back to 'timm'."
            ) from err

        self.model = VideoMAEModel.from_pretrained(hf_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=False)
        # Use CLS token representation as global video descriptor
        return outputs.last_hidden_state[:, 0]


class VideoMAEAnticipation(nn.Module):
    """VideoMAE backbone with a lightweight binary anticipation head."""

    def __init__(
        self,
        backbone: str = "videomae_small_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone_layers: int = 0,
        provider: str = "timm",
        hf_name: str | None = None,
    ) -> None:
        super().__init__()
        self.provider = provider.lower()
        self.backbone_name = backbone

        if self.provider == "huggingface":
            hf_model_name = hf_name or "MCG-NJU/videomae-small"
            self.backbone = VideoMAEHuggingFaceBackbone(hf_model_name)
            feature_dim = self.backbone.hidden_size
        else:
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
        if video.dim() != 5:
            raise ValueError(f"Video tensor must be 5D but got shape {video.shape}")

        if self.provider == "huggingface":
            features = self.backbone(video)
        else:
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
            provider=cfg.provider,
            hf_name=cfg.hf_name,
        )

    def configure_optim_parameters(self) -> list[Dict[str, Any]]:
        """Separate parameter groups for backbone and head if needed."""

        return [
            {"params": self.backbone.parameters(), "name": "backbone"},
            {"params": self.head.parameters(), "name": "head"},
        ]

