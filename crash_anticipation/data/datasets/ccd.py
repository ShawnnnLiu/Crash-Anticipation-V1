"""Dataset utilities for the Car Crash Dataset (CCD)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
class CCDSample:
    vidname: str
    label: int
    event_frame: Optional[int]
    total_frames: int
    start_frame: int
    metadata: Dict[str, Any]


class CCDDataset(Dataset[Dict[str, Any]]):
    """Load CCD image sequences using frame annotations."""

    def __init__(
        self,
        manifest_path: str | Path,
        frames_root: str | Path,
        clip_len: int = 16,
        fps: float = 8.0,
        frame_size: int = 224,
        lead_time_seconds: float = 2.0,
        max_offset_seconds: float = 1.0,
        is_train: bool = True,
        augmentation: Optional[Dict[str, Any]] = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required to load CCD image sequences. Install opencv-python.")

        self.manifest_path = Path(manifest_path)
        self.frames_root = Path(frames_root)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        if not self.frames_root.exists():
            raise FileNotFoundError(f"Frames root not found: {self.frames_root}")

        self.clip_len = clip_len
        self.fps = fps
        self.frame_size = frame_size
        self.lead_time_seconds = lead_time_seconds
        self.max_offset_seconds = max_offset_seconds
        self.is_train = is_train

        self.transform = build_video_transform(
            frame_size=frame_size,
            is_train=is_train,
            augmentation=augmentation or {},
        )

        df = pd.read_csv(self.manifest_path)
        required_columns = {"vidname", "label", "event_frame", "total_frames"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Manifest {self.manifest_path} missing columns: {missing}")

        self.samples: List[CCDSample] = []
        for record in df.to_dict(orient="records"):
            event_frame_val = record.get("event_frame")
            event_frame = int(event_frame_val) if not pd.isna(event_frame_val) else None
            start_frame = int(str(record.get("startframe", 0)).lstrip("0") or 0)
            metadata = {
                "video_id": record.get("video_id"),
                "timing": record.get("timing"),
                "weather": record.get("weather"),
                "egoinvolve": record.get("egoinvolve"),
            }
            self.samples.append(
                CCDSample(
                    vidname=str(record["vidname"]).zfill(6),
                    label=int(record["label"]),
                    event_frame=event_frame,
                    total_frames=int(record["total_frames"]),
                    start_frame=start_frame,
                    metadata={k: v for k, v in metadata.items() if v is not None},
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        indices = self._sample_indices(sample)
        frames = [self._read_frame(sample.vidname, idx) for idx in indices]
        clip = torch.stack(frames, dim=0).to(torch.float32)
        clip = self.transform(clip)

        crash_distance = None
        if sample.event_frame is not None:
            crash_distance = max(sample.event_frame - indices[-1], 0)

        time_to_event = (
            crash_distance / self.fps if crash_distance is not None and self.fps > 0 else float("inf")
        )

        return {
            "video": clip,
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "time_to_event": torch.tensor(time_to_event, dtype=torch.float32),
            "metadata": sample.metadata,
            "path": str(self.frames_root / f"C_{sample.vidname}_*.jpg"),
        }

    def _sample_indices(self, sample: CCDSample) -> np.ndarray:
        total_frames = sample.total_frames
        fps = self.fps
        lead_frames = max(int(round(fps * self.lead_time_seconds)), 1)
        max_offset = max(int(round(fps * self.max_offset_seconds)), 0)

        if sample.label == 1 and sample.event_frame is not None:
            crash_frame = min(sample.event_frame, total_frames - 1)
            offset = random.randint(0, max_offset) if self.is_train else max_offset // 2
            end_frame = max(crash_frame - lead_frames - offset, self.clip_len)
            start_frame = max(end_frame - self.clip_len, 0)
        else:
            max_start = max(total_frames - self.clip_len, 0)
            start_frame = random.randint(0, max_start) if self.is_train else max_start // 2

        indices = np.linspace(start_frame, start_frame + self.clip_len - 1, self.clip_len)
        indices = np.clip(indices, 0, total_frames - 1).astype(np.int64)
        return indices

    def _read_frame(self, vidname: str, index: int) -> torch.Tensor:
        frame_number = index + 1  # frames are 1-indexed in filenames
        frame_path = self.frames_root / f"C_{vidname}_{frame_number:02d}.jpg"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).permute(2, 0, 1)
        return tensor


def build_dataloaders(
    data_config: Any,
    batch_size: int,
    is_distributed: bool = False,
) -> tuple[DataLoader, DataLoader]:
    frames_root = getattr(data_config, "frames_root", None)
    if frames_root is None:
        raise ValueError("CCD data config must specify 'frames_root'")

    train_dataset = CCDDataset(
        manifest_path=data_config.train_manifest,
        frames_root=frames_root,
        clip_len=data_config.clip_len,
        fps=data_config.fps,
        frame_size=data_config.frame_size,
        lead_time_seconds=data_config.lead_time_seconds,
        max_offset_seconds=data_config.max_offset_seconds,
        is_train=True,
        augmentation=data_config.augmentation,
    )

    val_dataset = CCDDataset(
        manifest_path=data_config.val_manifest,
        frames_root=frames_root,
        clip_len=data_config.clip_len,
        fps=data_config.fps,
        frame_size=data_config.frame_size,
        lead_time_seconds=data_config.lead_time_seconds,
        max_offset_seconds=data_config.max_offset_seconds,
        is_train=False,
        augmentation={},
    )

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        videos = torch.stack([item["video"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        time_to_event = torch.stack([item["time_to_event"] for item in batch])
        metadata = [item["metadata"] for item in batch]
        paths = [item["path"] for item in batch]
        return {
            "video": videos,
            "label": labels,
            "time_to_event": time_to_event,
            "metadata": metadata,
            "path": paths,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not is_distributed,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=False,
        collate_fn=collate,
    )

    return train_loader, val_loader

