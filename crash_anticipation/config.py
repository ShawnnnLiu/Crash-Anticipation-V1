"""Utility helpers for loading experiment configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    train_manifest: str
    val_manifest: str
    dataset_type: str = "dad"
    frames_root: Optional[str] = None
    clip_len: int = 16
    fps: float = 8.0
    frame_size: int = 224
    sample_strategy: str = "uniform"
    lead_time_seconds: float = 2.0
    max_offset_seconds: float = 1.0
    num_workers: int = 8
    pin_memory: bool = True
    cache_frames: bool = False
    augmentation: Dict[str, Any] = field(default_factory=dict)
    # Windowed anticipation formulation (dataset_type: windows)
    dad_negatives_root: Optional[str] = None
    dad_test_negatives_root: Optional[str] = None
    dad_frame_stride: int = 2
    pos_horizon_seconds: float = 1.5
    neg_horizon_seconds: float = 2.5
    tau_seconds: float = 1.2
    min_context_frames: int = 8
    windows_per_video: int = 1


@dataclass
class ModelConfig:
    type: str = "videomae"
    backbone: str = "videomae_small_patch16_224"
    pretrained: bool = True
    dropout: float = 0.2
    freeze_backbone_layers: int = 0
    provider: str = "timm"
    hf_name: str = "MCG-NJU/videomae-small"
    temporal_hidden: int = 256


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: Optional[str] = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 4
    max_epochs: int = 20
    grad_clip: Optional[float] = 1.0
    accumulation_steps: int = 1
    eval_interval: int = 1
    save_interval: int = 1
    output_dir: str = "outputs/baseline"
    checkpoint_path: Optional[str] = None
    log_interval: int = 20
    pos_weight: Optional[float] = None
    lead_time_thresholds: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 3.0]
    )
    use_amp: bool = True
    # Knowledge distillation (student training)
    distill_checkpoint: Optional[str] = None
    distill_config: Optional[str] = None
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        data_cfg = config_dict.get("data")
        if data_cfg is None:
            raise ValueError("Configuration must include a 'data' section with manifests")
        model_cfg = config_dict.get("model", {})
        optim_cfg = config_dict.get("optim", {})
        sched_cfg = config_dict.get("scheduler", {})
        train_cfg = config_dict.get("train", {})
        return cls(
            data=DataConfig(**data_cfg),
            model=ModelConfig(**model_cfg),
            optim=OptimizerConfig(**optim_cfg),
            scheduler=SchedulerConfig(**sched_cfg),
            train=TrainingConfig(**train_cfg),
        )


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Load a YAML config file and optionally merge a dictionary of overrides."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle) or {}

    if overrides:
        config_dict = _deep_update(config_dict, overrides)

    return ExperimentConfig.from_dict(config_dict)


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested mapping with overrides."""

    if overrides is None:
        return base

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base

