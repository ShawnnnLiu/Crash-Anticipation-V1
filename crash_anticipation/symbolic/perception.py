"""Object detection and tracking for the symbolic layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# COCO ids of road agents we care about.
ROAD_AGENT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    track_id: int
    cls_name: str
    xyxy: np.ndarray  # (4,) float32, pixel coords
    conf: float


class ObjectPerception:
    """YOLO detection with persistent ByteTrack IDs, one call per frame."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: Optional[str] = None,
        conf: float = 0.30,
        iou: float = 0.5,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.iou = iou

    def reset(self) -> None:
        """Reset tracker state between independent video streams."""

        if getattr(self.model, "predictor", None) is not None:
            trackers = getattr(self.model.predictor, "trackers", None)
            if trackers:
                for tracker in trackers:
                    tracker.reset()

    def track(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.track(
            frame_bgr,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            classes=list(ROAD_AGENT_CLASSES),
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        boxes = results.boxes
        if boxes is None or boxes.id is None:
            return detections
        for xyxy, track_id, cls_id, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        ):
            detections.append(
                Detection(
                    track_id=int(track_id),
                    cls_name=ROAD_AGENT_CLASSES.get(int(cls_id), str(cls_id)),
                    xyxy=xyxy.astype(np.float32),
                    conf=float(conf),
                )
            )
        return detections
