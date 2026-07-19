"""Infer vehicle dynamics from tracked bounding boxes.

Monocular time-to-collision uses the classical visual-looming result: for an
object approaching at constant speed, TTC = s / (ds/dt) where s is any image
size of the object. We estimate d(log s)/dt by least-squares over a short
history, which is robust to per-frame detector jitter.

Whether the object is coming *at us* — rather than being overtaken in a
neighboring lane — is decided by projecting its track to the moment of closest
approach. For a pinhole camera and constant relative velocity, the lateral
offset from the camera at closest approach, expressed in current-frame pixels,
is simply x_dot * TTC (the current image position cancels out of the algebra).
The object's own box width is a ruler at its depth (w = f * W / Z), so
dividing by it converts that offset to physical vehicle-widths with no depth
estimate: a car projected to pass more than ~1.5 of its own widths off-axis
clears the ego vehicle; one near zero is on a collision course. This is the
formal version of the sailor's constant-bearing rule, but scale-invariant —
unlike a fixed bearing-rate threshold, it neither ignores distant cars that
drift slowly in the image nor dismisses fast cut-ins that will converge on us.

Known limitation: x_dot * TTC assumes negligible ego rotation. During hard ego
yaw (turns, lane changes) the whole field acquires uniform lateral flow;
correcting for that needs background optical flow or an IMU, neither of which
is derivable from agent boxes alone.
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
    in_path: bool  # within the ego corridor at the object's own depth
    ttc_s: float  # time-to-collision estimate, +inf unless on collision course
    looming: float  # relative expansion rate, 1/s (positive = closing)
    lateral_v: float  # normalized horizontal speed, frame-widths/s
    size_frac: float  # bbox area / frame area
    miss_widths: float  # projected closest-approach offset, in object widths
    converging: bool  # drifting toward the ego line (image center)
    heading: str  # "approaching" | "passing" | "crossing" | "stable" | "receding"


class TrackDynamics:
    """Maintain per-track histories and derive symbolic facts."""

    def __init__(
        self,
        fps: float = 10.0,
        history: int = 6,
        min_history: int = 3,
        miss_widths_max: float = 1.5,
        path_halfwidth: float = 1.2,
        persist: int = 2,
    ) -> None:
        self.fps = fps
        self.history = history
        self.min_history = min_history
        # Collision-course threshold on the projected miss, in object widths.
        # 1.5 approximates (object width + ego width) / 2 for car-sized agents;
        # generous for trucks, tight for motorcycles, a workable compromise.
        self.miss_widths_max = miss_widths_max
        # Ego-corridor half-width, in object widths at the object's depth.
        self.path_halfwidth = path_halfwidth
        # Consecutive frames the collision-course test must hold before a
        # finite TTC is emitted (the miss is a product of two noisy slopes).
        self.persist = persist
        self.tracks: Dict[int, Deque[np.ndarray]] = {}
        self._course_streak: Dict[int, int] = {}

    def reset(self) -> None:
        self.tracks.clear()
        self._course_streak.clear()

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
                self._course_streak.pop(track_id, None)

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
        cx_px = (x1 + x2) / 2.0
        cx = cx_px / width
        box_w = max(x2 - x1, 1.0)
        size_frac = max(box_w * (y2 - y1), 1.0) / (width * height)

        zone = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
        # Perspective-aware corridor: the corridor is ~path_halfwidth object
        # widths either side of straight-ahead *at the object's depth*, so it
        # narrows toward the vanishing point instead of swallowing every
        # distant car regardless of lane.
        in_path = abs(cx_px - width / 2.0) < self.path_halfwidth * box_w

        looming = 0.0
        lateral_px = 0.0
        if len(hist) >= self.min_history:
            looming = self._looming_rate(hist)
            lateral_px = self._lateral_velocity_px(hist)
        lateral_v = lateral_px / width

        # Projected miss at closest approach: |x_dot| * TTC in pixels, i.e.
        # |x_dot| / looming, then divided by the object's width to become
        # depth-free. Only defined while the object is closing.
        if looming > 1e-3:
            miss_widths = abs(lateral_px) / (looming * box_w)
        else:
            miss_widths = float("inf")

        converging = (cx_px - width / 2.0) * lateral_px < 0

        on_course = looming > 1e-3 and miss_widths <= self.miss_widths_max
        streak = self._course_streak.get(det.track_id, 0) + 1 if on_course else 0
        self._course_streak[det.track_id] = streak

        ttc = 1.0 / looming if streak >= self.persist else float("inf")

        if looming > 0.08:
            heading = "approaching" if on_course else "passing"
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
            miss_widths=miss_widths,
            converging=converging,
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

    def _lateral_velocity_px(self, hist: Deque[np.ndarray]) -> float:
        """Least-squares slope of the box-center x over time, px/s."""

        centers = np.asarray([(b[0] + b[2]) / 2.0 for b in hist])
        t = np.arange(len(centers)) / self.fps
        t_centered = t - t.mean()
        denom = float((t_centered**2).sum())
        if denom <= 0:
            return 0.0
        return float((t_centered * (centers - centers.mean())).sum() / denom)
