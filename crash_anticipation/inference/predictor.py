"""Streaming sliding-window risk prediction.

Mirrors the training-time window construction exactly: a fixed-length window
ending at the newest frame, left-padded by repeating the first frame while
the buffer warms up. This lets the predictor emit a risk estimate from the
very first frame of a stream, as required for deployment.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
import torch

from ..config import ExperimentConfig, load_config
from ..data.transforms import build_video_transform
from ..models import build_model


@dataclass
class RiskEstimate:
    prob: float  # raw model probability
    prob_smooth: float  # EMA-smoothed probability (for display/alerting)
    latency_ms: float  # model forward time for this step


class OnlineAnticipator:
    def __init__(
        self,
        model: torch.nn.Module,
        clip_len: int = 16,
        frame_size: int = 224,
        device: Optional[torch.device] = None,
        ema_alpha: float = 0.4,
    ) -> None:
        self.model = model
        self.clip_len = clip_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ema_alpha = ema_alpha
        self.transform = build_video_transform(frame_size=frame_size, is_train=False, augmentation={})
        self.model.to(self.device).eval()
        self.reset()

    def reset(self) -> None:
        self.buffer: Deque[torch.Tensor] = deque(maxlen=self.clip_len)
        self.prob_smooth = 0.0

    @torch.no_grad()
    def step(self, frame_bgr: np.ndarray) -> RiskEstimate:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.buffer.append(torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32))

        frames = list(self.buffer)
        if len(frames) < self.clip_len:
            frames = [frames[0]] * (self.clip_len - len(frames)) + frames

        clip = self.transform(torch.stack(frames, dim=0)).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.model(clip)["logits"]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        prob = torch.sigmoid(logits).item()
        self.prob_smooth = self.ema_alpha * prob + (1.0 - self.ema_alpha) * self.prob_smooth
        return RiskEstimate(prob=prob, prob_smooth=self.prob_smooth, latency_ms=latency_ms)


def load_anticipator(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: Optional[str] = None,
    ema_alpha: float = 0.4,
) -> Tuple[OnlineAnticipator, ExperimentConfig]:
    cfg = load_config(config_path, overrides={})
    model = build_model(cfg.model)
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(checkpoint_path), map_location=dev)
    model.load_state_dict(ckpt["model_state"])
    anticipator = OnlineAnticipator(
        model,
        clip_len=cfg.data.clip_len,
        frame_size=cfg.data.frame_size,
        device=dev,
        ema_alpha=ema_alpha,
    )
    return anticipator, cfg
