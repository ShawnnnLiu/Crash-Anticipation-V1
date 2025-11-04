"""Lightweight video transform utilities."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


class VideoTransform:
    """Apply basic spatial augmentations and normalization to video clips."""

    def __init__(
        self,
        frame_size: int = 224,
        is_train: bool = True,
        augmentation: Dict[str, Any] | None = None,
    ) -> None:
        self.frame_size = frame_size
        self.is_train = is_train
        self.augmentation = augmentation or {}
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # Expect clip shape (T, C, H, W) with values in [0, 255]
        clip = clip.float() / 255.0

        clip = F.interpolate(
            clip,
            size=(self.frame_size, self.frame_size),
            mode="bilinear",
            align_corners=False,
        )

        if self.is_train and self.augmentation.get("random_horizontal_flip", True):
            if torch.rand(1) < 0.5:
                clip = torch.flip(clip, dims=[-1])

        if self.is_train and self.augmentation.get("temporal_jitter", False):
            clip = self._temporal_jitter(clip)

        clip = (clip - self.mean) / self.std
        return clip

    def _temporal_jitter(self, clip: torch.Tensor) -> torch.Tensor:
        max_jitter = int(self.augmentation.get("temporal_jitter", 2))
        if max_jitter <= 0:
            return clip
        t, _, _, _ = clip.shape
        shift = torch.randint(-max_jitter, max_jitter + 1, (1,)).item()
        if shift == 0:
            return clip
        padded = torch.zeros_like(clip)
        if shift > 0:
            padded[shift:] = clip[: t - shift]
        else:
            padded[: shift] = clip[-shift:]
        return padded


def build_video_transform(
    frame_size: int = 224,
    is_train: bool = True,
    augmentation: Dict[str, Any] | None = None,
) -> VideoTransform:
    return VideoTransform(frame_size=frame_size, is_train=is_train, augmentation=augmentation)

