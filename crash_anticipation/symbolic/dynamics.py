"""Infer vehicle dynamics from tracked bounding boxes.

Monocular time-to-collision uses the classical visual-looming result: for an
object approaching at constant speed, TTC = s / (ds/dt) where s is any image
size of the object. We estimate d(log s)/dt by least-squares over a short
history, which is robust to per-frame detector jitter.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .perception import Detection


@dataclass
class ObjectFacts:
    """Symbolic facts about one tracked road agent at one instant."""

    track_id: int
    cls_name: str
    xyxy: np.ndarray
    zone: str  # "left" | "center" | "right"
    in_path: bool  # inside the ego corridor
    ttc_s: float  # time-to-collision estimate, +inf if receding
    looming: float  # relative expansion rate, 1/s (positive = approaching)
    lateral_v: float  # normalized horizontal speed, frame-widths/s
    size_frac: float  # bbox area / frame area
    heading: str  # "approaching" | "crossing" | "stable" | "receding"


class TrackDynamics:
    """Maintain per-track histories and derive symbolic facts."""

    def __init__(
        self,
        fps: float = 10.0,
        history: int = 6,
        corridor: Tuple[float, float] = (0.30, 0.70),
        min_history: int = 3,
    ) -> None:
        self.fps = fps
        self.history = history
        self.corridor = corridor
        self.min_history = min_history
        self.tracks: Dict[int, Deque[np.ndarray]] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, detections: List[Detection], frame_shape: Tuple[int, int]) -> List[ObjectFacts]:
        height, width = frame_shape[:2]
        seen = set()
        facts: List[ObjectFacts] = []

        for det in detections:
            seen.add(det.track_id)
            hist = self.tracks.setdefault(det.track_id, deque(maxlen=self.history))
            hist.append(det.xyxy.copy())
            fact = self._facts_for(det, hist, width, height)
            if fact is not None:
                facts.append(fact)

        # Drop tracks that disappeared.
        for track_id in list(self.tracks):
            if track_id not in seen:
                del self.tracks[track_id]

        return facts

    # -- helpers -------------------------------------------------------------

    def _facts_for(
        self,
        det: Detection,
        hist: Deque[np.ndarray],
        width: int,
        height: int,
    ) -> Optional[ObjectFacts]:
        x1, y1, x2, y2 = det.xyxy
        cx = (x1 + x2) / 2.0 / width
        size_frac = max((x2 - x1) * (y2 - y1), 1.0) / (width * height)

        zone = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
        lo, hi = self.corridor
        in_path = lo <= cx <= hi and (y2 / height) > 0.45

        looming = 0.0
        lateral_v = 0.0
        if len(hist) >= self.min_history:
            looming = self._looming_rate(hist)
            lateral_v = self._lateral_velocity(hist, width)

        # Collision-course test (constant bearing, decreasing range): an object
        # we are merely passing looms too, but it also drifts across the field
        # of view. Only assign a finite TTC when the bearing is near-constant
        # or the object sits in the ego corridor.
        on_collision_course = in_path or abs(lateral_v) < 0.15
        if looming > 1e-3 and on_collision_course:
            ttc = 1.0 / looming
        else:
            ttc = float("inf")

        if looming > 0.08:
            heading = "approaching"
        elif abs(lateral_v) > 0.25:
            heading = "crossing"
        elif looming < -0.08:
            heading = "receding"
        else:
            heading = "stable"

        return ObjectFacts(
            track_id=det.track_id,
            cls_name=det.cls_name,
            xyxy=det.xyxy,
            zone=zone,
            in_path=in_path,
            ttc_s=ttc,
            looming=looming,
            lateral_v=lateral_v,
            size_frac=size_frac,
            heading=heading,
        )

    def _looming_rate(self, hist: Deque[np.ndarray]) -> float:
        """Least-squares slope of log(size) over time -> relative expansion 1/s."""

        sizes = []
        for box in hist:
            w = max(box[2] - box[0], 1.0)
            h = max(box[3] - box[1], 1.0)
            sizes.append(math.sqrt(w * h))
        log_s = np.log(np.asarray(sizes))
        t = np.arange(len(log_s)) / self.fps
        t_centered = t - t.mean()
        denom = float((t_centered**2).sum())
        if denom <= 0:
            return 0.0
        slope = float((t_centered * (log_s - log_s.mean())).sum() / denom)
        return slope

    def _lateral_velocity(self, hist: Deque[np.ndarray], width: int) -> float:
        centers = np.asarray([(b[0] + b[2]) / 2.0 for b in hist]) / width
        t = np.arange(len(centers)) / self.fps
        t_centered = t - t.mean()
        denom = float((t_centered**2).sum())
        if denom <= 0:
            return 0.0
        return float((t_centered * (centers - centers.mean())).sum() / denom)
