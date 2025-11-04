"""Dataset utilities for the Dashcam Accident Dataset (DAD)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..transforms import build_video_transform

try:
    import decord  # type: ignore

    decord.bridge.set_bridge("torch")
except Exception:  # pragma: no cover - optional dependency
    decord = None

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Sample:
    video_path: Path
    label: int
    event_frame: int | None
    fps: float
    total_frames: int | None
    metadata: Dict[str, Any]


class DADDataset(Dataset[Dict[str, Any]]):
    """Dataset wrapper reading clips from the Dashcam Accident Dataset.

    The dataset expects a CSV manifest with (at least) the following columns:

    ``video_path`` (str): path to video file or directory containing frames
    ``label`` (int): 1 for crash, 0 for non-crash
    ``event_frame`` (int, optional): index of crash event frame (0-based)
    ``fps`` (float, optional): frames per second override
    ``total_frames`` (int, optional): total frame count
    ``split`` (str, optional): dataset split name (train/val/test)

    Additional columns are stored in metadata and returned with each sample.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        clip_len: int = 16,
        fps: float = 8.0,
        frame_size: int = 224,
        sample_strategy: str = "uniform",
        lead_time_seconds: float = 2.0,
        max_offset_seconds: float = 1.0,
        is_train: bool = True,
        cache_frames: bool = False,
        augmentation: Dict[str, Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {self.manifest_path}")

        self.clip_len = clip_len
        self.default_fps = fps
        self.frame_size = frame_size
        self.sample_strategy = sample_strategy
        self.lead_time_seconds = lead_time_seconds
        self.max_offset_seconds = max_offset_seconds
        self.is_train = is_train
        self.cache_frames = cache_frames
        self.transform = build_video_transform(
            frame_size=frame_size,
            is_train=is_train,
            augmentation=augmentation or {},
        )

        df = pd.read_csv(self.manifest_path)
        if "video_path" not in df.columns or "label" not in df.columns:
            raise ValueError("Manifest must contain 'video_path' and 'label' columns")

        self.samples: List[Sample] = []
        for row in df.to_dict(orient="records"):
            video_path = Path(row.pop("video_path"))
            label = int(row.pop("label"))
            event_frame = row.pop("event_frame", None)
            fps_value = float(row.pop("fps", fps)) if row.get("fps") is not None else fps
            total_frames = row.pop("total_frames", None)
            self.samples.append(
                Sample(
                    video_path=video_path,
                    label=label,
                    event_frame=int(event_frame) if pd.notna(event_frame) else None,
                    fps=fps_value,
                    total_frames=int(total_frames) if pd.notna(total_frames) else None,
                    metadata=row,
                )
            )

        self._frame_cache: Dict[Path, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        frames, fps = self._load_clip(sample)
        clip = self.transform(frames)

        crash_distance = None
        if sample.event_frame is not None:
            crash_distance = max(sample.event_frame - self._last_index, 0)

        time_to_event = (
            crash_distance / fps if crash_distance is not None and fps > 0 else float("inf")
        )

        return {
            "video": clip,
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "time_to_event": torch.tensor(time_to_event, dtype=torch.float32),
            "metadata": sample.metadata,
            "path": str(sample.video_path),
        }

    def _load_clip(self, sample: Sample) -> tuple[torch.Tensor, float]:
        path = sample.video_path
        if self.cache_frames and path in self._frame_cache:
            frames = self._frame_cache[path]
            fps = sample.fps
            indices = self._sample_indices(sample, frames.shape[0])
            clip = frames[indices]
            self._last_index = indices[-1]
            return clip, fps

        if path.is_dir():
            frames = self._load_from_frames_dir(path)
            total_frames = frames.shape[0]
            indices = self._sample_indices(sample, total_frames)
            clip = frames[indices]
            self._last_index = indices[-1]
            if self.cache_frames:
                self._frame_cache[path] = frames
            return clip, sample.fps

        if decord is not None:
            reader = decord.VideoReader(str(path))
            total_frames = len(reader)
            indices = self._sample_indices(sample, total_frames)
            clip = reader.get_batch(indices).permute(0, 3, 1, 2).to(torch.float32)
            self._last_index = indices[-1]
            fps = float(reader.get_avg_fps()) if sample.fps <= 0 else sample.fps
            return clip, fps

        if cv2 is None:
            raise RuntimeError(
                "Neither decord nor OpenCV is available for video decoding. "
                "Install decord (preferred) or opencv-python."
            )

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or sample.fps
        indices = self._sample_indices(sample, total_frames)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = cap.read()
            if not success:
                raise RuntimeError(f"Failed to read frame {idx} from {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        cap.release()
        clip = torch.stack(frames, dim=0).to(torch.float32)
        self._last_index = indices[-1]
        return clip, float(fps)

    def _sample_indices(self, sample: Sample, total_frames: int) -> np.ndarray:
        if total_frames <= 0:
            raise ValueError(f"Video {sample.video_path} has no frames")

        fps = sample.fps if sample.fps > 0 else self.default_fps
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

    def _load_from_frames_dir(self, directory: Path) -> torch.Tensor:
        frame_files = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png"))
        if not frame_files:
            raise ValueError(f"No frame images found in directory: {directory}")
        frames = [torch.from_numpy(_read_image(path)) for path in frame_files]
        stacked = torch.stack(frames, dim=0).to(torch.float32)
        return stacked


def collate_dad(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def _read_image(path: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for reading frame directories")
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_dataloaders(
    data_config: Any,
    batch_size: int,
    is_distributed: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders from config objects."""

    train_dataset = DADDataset(
        manifest_path=data_config.train_manifest,
        clip_len=data_config.clip_len,
        fps=data_config.fps,
        frame_size=data_config.frame_size,
        sample_strategy=data_config.sample_strategy,
        lead_time_seconds=data_config.lead_time_seconds,
        max_offset_seconds=data_config.max_offset_seconds,
        is_train=True,
        cache_frames=data_config.cache_frames,
        augmentation=data_config.augmentation,
    )

    val_dataset = DADDataset(
        manifest_path=data_config.val_manifest,
        clip_len=data_config.clip_len,
        fps=data_config.fps,
        frame_size=data_config.frame_size,
        sample_strategy=data_config.sample_strategy,
        lead_time_seconds=data_config.lead_time_seconds,
        max_offset_seconds=data_config.max_offset_seconds,
        is_train=False,
        cache_frames=False,
        augmentation={},
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not is_distributed,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=True,
        collate_fn=collate_dad,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=False,
        collate_fn=collate_dad,
    )

    return train_loader, val_loader

