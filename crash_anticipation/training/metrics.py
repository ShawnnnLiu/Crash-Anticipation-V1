"""Evaluation metrics for crash anticipation models."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: Iterable[float] = (0.5,),
) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    metrics: Dict[str, float] = {}
    try:
        metrics["ap"] = float(average_precision_score(labels_np, probs))
    except ValueError:
        metrics["ap"] = float("nan")

    for threshold in thresholds:
        preds = (probs >= threshold).astype(np.int32)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds, average="binary", zero_division=0
        )
        metrics[f"precision@{threshold:.2f}"] = float(precision)
        metrics[f"recall@{threshold:.2f}"] = float(recall)
        metrics[f"f1@{threshold:.2f}"] = float(f1)

    return metrics


def compute_lead_time_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    time_to_event: torch.Tensor,
    lead_times: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    tte_np = time_to_event.detach().cpu().numpy()

    metrics: Dict[str, float] = {}
    preds = (probs >= threshold).astype(np.int32)

    for lead_time in lead_times:
        # true positive if crash sample predicted positive and detected before lead time
        positive_idx = labels_np == 1
        if positive_idx.sum() == 0:
            metrics[f"lead_recall@{lead_time:.1f}s"] = float("nan")
            continue

        detected = (preds == 1) & positive_idx & (tte_np <= lead_time)
        metrics[f"lead_recall@{lead_time:.1f}s"] = float(detected.sum() / positive_idx.sum())

    return metrics

