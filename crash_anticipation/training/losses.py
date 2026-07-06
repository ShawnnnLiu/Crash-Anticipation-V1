"""Loss functions for crash anticipation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AnticipationLoss(nn.Module):
    """Earliness-weighted binary cross-entropy.

    Per-sample weights come from the dataset: positives near the accident
    onset carry weight exp(-tte/tau), so the model is rewarded for firing
    early but never *forced* to fire when there is nothing to see yet.
    Guard-band windows arrive with weight 0 and contribute no gradient.
    """

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if weights is None:
            return bce.mean()
        return (bce * weights).sum() / weights.sum().clamp_min(1e-6)


class DistillationLoss(nn.Module):
    """Soft-target distillation for a binary head.

    total = alpha * BCE(student/T, sigmoid(teacher/T)) * T^2
          + (1 - alpha) * AnticipationLoss(student, labels, weights)
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.hard_loss = AnticipationLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.temperature
        soft_targets = torch.sigmoid(teacher_logits.detach() / t)
        soft = F.binary_cross_entropy_with_logits(student_logits / t, soft_targets) * (t * t)
        hard = self.hard_loss(student_logits, labels, weights)
        return self.alpha * soft + (1.0 - self.alpha) * hard
