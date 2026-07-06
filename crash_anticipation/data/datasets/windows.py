"""Unified windowed dataset for online crash anticipation.

This module replaces the earlier clip-classification formulation (which drew
every sample from crash videos and therefore had no negatives) with a
window-based formulation suitable for *online* anticipation:

- Crash videos (CCD): jpg frame sequences at 10 fps with an exact annotated
  accident-onset frame. Windows ending close to the onset are positives with
  an exponential earliness weight; windows ending well before the onset are
  same-domain temporal negatives; a guard band in between is excluded.
- Normal videos (DAD negatives): mp4 clips with no accident. Every window is
  a true negative. Sampled at a frame stride so the effective rate matches
  the CCD timeline (10 fps).

Each sample is a fixed-length window of ``clip_len`` effective frames ending
at index ``e`` (left-clamped: early windows repeat the first frame so the
model can predict from the first moments of a stream, matching deployment).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..transforms import build_video_transform

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass
class VideoRecord:
    """One source video on the unified 10 fps effective timeline."""

    uid: str
    source: str  # "ccd" | "dad"
    kind: str  # "crash" | "normal"
    num_frames: int  # effective frames
    event_index: Optional[int] = None  # effective index of accident onset
    frames_root: Optional[Path] = None  # ccd: directory of jpgs
    video_path: Optional[Path] = None  # dad: mp4 path
    frame_stride: int = 1  # native frames per effective frame
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Frame readers
# ---------------------------------------------------------------------------


def read_ccd_window(record: VideoRecord, indices: Sequence[int]) -> List[np.ndarray]:
    frames = []
    for idx in indices:
        # CCD files are 1-indexed: C_{vid}_{NN}.jpg
        path = record.frames_root / f"C_{record.uid}_{idx + 1:02d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read CCD frame: {path}")
        frames.append(img)
    return frames


def read_video_window(record: VideoRecord, indices: Sequence[int]) -> List[np.ndarray]:
    """Read effective-frame indices from an mp4 with sequential grabs."""

    native = [idx * record.frame_stride for idx in indices]
    needed = sorted(set(native))
    cap = cv2.VideoCapture(str(record.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {record.video_path}")
    cache: Dict[int, np.ndarray] = {}
    cursor = 0
    for target in needed:
        while cursor < target:
            cap.grab()
            cursor += 1
        ok, frame = cap.read()
        cursor += 1
        if not ok:
            # Fall back to the last frame successfully read.
            if cache:
                frame = cache[max(cache)]
            else:
                cap.release()
                raise RuntimeError(f"Failed to decode frame {target} of {record.video_path}")
        cache[target] = frame
    cap.release()
    return [cache[n] for n in native]


def read_window(record: VideoRecord, indices: Sequence[int]) -> List[np.ndarray]:
    if record.frames_root is not None:
        return read_ccd_window(record, indices)
    return read_video_window(record, indices)


# ---------------------------------------------------------------------------
# Window labelling
# ---------------------------------------------------------------------------


def label_window(
    record: VideoRecord,
    end_index: int,
    fps: float,
    pos_horizon_s: float,
    neg_horizon_s: float,
    tau_s: float,
) -> Tuple[float, float, float]:
    """Return (label, weight, time_to_event_seconds) for a window ending at end_index.

    Crash videos:
      tte <= 0             -> label 1, weight 1        (impact visible)
      0 < tte <= pos_h     -> label 1, weight e^{-tte/tau}  (anticipation zone)
      pos_h < tte < neg_h  -> weight 0                 (guard band, excluded)
      tte >= neg_h         -> label 0, weight 1        (temporal negative)
    Normal videos: label 0, weight 1, tte = +inf.
    """

    if record.kind != "crash" or record.event_index is None:
        return 0.0, 1.0, float("inf")

    tte = (record.event_index - end_index) / fps
    if tte <= 0:
        return 1.0, 1.0, tte
    if tte <= pos_horizon_s:
        return 1.0, math.exp(-tte / tau_s), tte
    if tte < neg_horizon_s:
        return 0.0, 0.0, tte
    return 0.0, 1.0, tte


class AnticipationWindowDataset(Dataset[Dict[str, Any]]):
    """Sample fixed-length windows from crash and normal videos."""

    def __init__(
        self,
        records: List[VideoRecord],
        clip_len: int = 16,
        fps: float = 10.0,
        frame_size: int = 224,
        pos_horizon_s: float = 1.5,
        neg_horizon_s: float = 2.5,
        tau_s: float = 1.2,
        min_end: int = 7,
        zone_probs: Optional[Dict[str, float]] = None,
        is_train: bool = True,
        augmentation: Optional[Dict[str, Any]] = None,
        windows_per_video: int = 1,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for AnticipationWindowDataset.")

        self.records = records
        self.clip_len = clip_len
        self.fps = fps
        self.pos_horizon_s = pos_horizon_s
        self.neg_horizon_s = neg_horizon_s
        self.tau_s = tau_s
        self.min_end = min_end
        self.zone_probs = zone_probs or {"pos": 0.5, "neg": 0.3, "post": 0.2}
        self.is_train = is_train
        self.windows_per_video = max(windows_per_video, 1)

        self.transform = build_video_transform(
            frame_size=frame_size,
            is_train=is_train,
            augmentation=augmentation or {},
        )

        if not is_train:
            self.eval_windows = self._build_eval_windows()

    # -- window enumeration -------------------------------------------------

    def _zone_ranges(self, record: VideoRecord) -> Dict[str, Tuple[int, int]]:
        """Valid inclusive end-index ranges per sampling zone."""

        last = record.num_frames - 1
        if record.kind != "crash" or record.event_index is None:
            return {"neg": (self.min_end, last)}

        event = record.event_index
        pos_lo = max(event - int(round(self.pos_horizon_s * self.fps)), self.min_end)
        neg_hi = event - int(math.ceil(self.neg_horizon_s * self.fps))
        zones: Dict[str, Tuple[int, int]] = {}
        if pos_lo <= min(event, last):
            zones["pos"] = (pos_lo, min(event, last))
        if event + 1 <= last:
            zones["post"] = (event + 1, last)
        if neg_hi >= self.min_end:
            zones["neg"] = (self.min_end, neg_hi)
        return zones

    def _build_eval_windows(self) -> List[Tuple[int, int]]:
        """Deterministic (record_idx, end_index) pairs with mixed labels."""

        windows: List[Tuple[int, int]] = []
        for ridx, record in enumerate(self.records):
            zones = self._zone_ranges(record)
            if record.kind == "crash":
                if "pos" in zones:
                    lo, hi = zones["pos"]
                    # Half the positive horizon before onset.
                    windows.append((ridx, max(lo, hi - int(round(0.5 * self.pos_horizon_s * self.fps)))))
                if "neg" in zones:
                    windows.append((ridx, zones["neg"][1]))
            else:
                lo, hi = zones["neg"]
                windows.append((ridx, (lo + hi) // 2))
                windows.append((ridx, hi))
        return windows

    # -- torch Dataset API ---------------------------------------------------

    def __len__(self) -> int:
        if self.is_train:
            return len(self.records) * self.windows_per_video
        return len(self.eval_windows)

    def _sample_end(self, record: VideoRecord) -> int:
        zones = self._zone_ranges(record)
        names = [z for z in ("pos", "neg", "post") if z in zones]
        weights = [self.zone_probs.get(z, 0.0) for z in names]
        if not names:
            return record.num_frames - 1
        if sum(weights) <= 0:
            weights = [1.0] * len(names)
        zone = random.choices(names, weights=weights, k=1)[0]
        lo, hi = zones[zone]
        return random.randint(lo, hi)

    def _window_indices(self, end_index: int) -> List[int]:
        start = end_index - self.clip_len + 1
        return [max(i, 0) for i in range(start, end_index + 1)]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.is_train:
            record = self.records[index // self.windows_per_video]
            end_index = self._sample_end(record)
        else:
            ridx, end_index = self.eval_windows[index]
            record = self.records[ridx]

        indices = self._window_indices(end_index)
        frames_bgr = read_window(record, indices)
        frames = [
            torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            for f in frames_bgr
        ]
        clip = torch.stack(frames, dim=0).to(torch.float32)
        clip = self.transform(clip)

        label, weight, tte = label_window(
            record,
            end_index,
            self.fps,
            self.pos_horizon_s,
            self.neg_horizon_s,
            self.tau_s,
        )

        return {
            "video": clip,
            "label": torch.tensor(label, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "time_to_event": torch.tensor(tte, dtype=torch.float32),
            "metadata": dict(record.metadata, uid=record.uid, source=record.source),
            "path": str(record.video_path or record.frames_root),
        }


def collate_windows(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "video": torch.stack([item["video"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "weight": torch.stack([item["weight"] for item in batch]),
        "time_to_event": torch.stack([item["time_to_event"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
        "path": [item["path"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def load_ccd_records(manifest_path: str | Path, frames_root: str | Path) -> List[VideoRecord]:
    frames_root = Path(frames_root)
    df = pd.read_csv(manifest_path)
    records: List[VideoRecord] = []
    for row in df.to_dict(orient="records"):
        event = row.get("event_frame")
        event_index = int(event) if event is not None and not pd.isna(event) else None
        records.append(
            VideoRecord(
                uid=str(row["vidname"]).zfill(6),
                source="ccd",
                kind="crash" if int(row["label"]) == 1 and event_index is not None else "normal",
                num_frames=int(row["total_frames"]),
                event_index=event_index,
                frames_root=frames_root,
                metadata={
                    k: row[k]
                    for k in ("timing", "weather", "egoinvolve")
                    if k in row and not pd.isna(row[k])
                },
            )
        )
    return records


def load_dad_negative_records(
    videos_dir: str | Path,
    frame_stride: int = 2,
    native_frames: int = 100,
) -> List[VideoRecord]:
    videos_dir = Path(videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(f"DAD negatives directory not found: {videos_dir}")
    records = []
    for path in sorted(videos_dir.glob("*.mp4")):
        records.append(
            VideoRecord(
                uid=path.stem,
                source="dad",
                kind="normal",
                num_frames=native_frames // frame_stride,
                video_path=path,
                frame_stride=frame_stride,
            )
        )
    return records


def split_records(
    records: List[VideoRecord], val_every: int = 10
) -> Tuple[List[VideoRecord], List[VideoRecord]]:
    """Deterministic interleaved split (every ``val_every``-th record to val)."""

    train = [r for i, r in enumerate(records) if i % val_every != val_every - 1]
    val = [r for i, r in enumerate(records) if i % val_every == val_every - 1]
    return train, val


def build_dataloaders(
    data_config: Any,
    batch_size: int,
    is_distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    ccd_train = load_ccd_records(data_config.train_manifest, data_config.frames_root)
    ccd_val = load_ccd_records(data_config.val_manifest, data_config.frames_root)

    dad_train: List[VideoRecord] = []
    dad_val: List[VideoRecord] = []
    if getattr(data_config, "dad_negatives_root", None):
        dad_all = load_dad_negative_records(
            data_config.dad_negatives_root,
            frame_stride=getattr(data_config, "dad_frame_stride", 2),
        )
        dad_train, dad_val = split_records(dad_all)

    common = dict(
        clip_len=data_config.clip_len,
        fps=data_config.fps,
        frame_size=data_config.frame_size,
        pos_horizon_s=getattr(data_config, "pos_horizon_seconds", 1.5),
        neg_horizon_s=getattr(data_config, "neg_horizon_seconds", 2.5),
        tau_s=getattr(data_config, "tau_seconds", 1.2),
        min_end=getattr(data_config, "min_context_frames", 8) - 1,
    )

    train_dataset = AnticipationWindowDataset(
        records=ccd_train + dad_train,
        is_train=True,
        augmentation=data_config.augmentation,
        windows_per_video=getattr(data_config, "windows_per_video", 1),
        **common,
    )
    val_dataset = AnticipationWindowDataset(
        records=ccd_val + dad_val,
        is_train=False,
        augmentation={},
        **common,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=True,
        collate_fn=collate_windows,
        persistent_workers=data_config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=False,
        collate_fn=collate_windows,
        persistent_workers=data_config.num_workers > 0,
    )
    return train_loader, val_loader
