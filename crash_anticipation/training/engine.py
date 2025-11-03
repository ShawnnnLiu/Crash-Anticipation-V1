"""Training and evaluation loops for crash anticipation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, _LRScheduler

from ..config import ExperimentConfig
from ..utils import AverageMeter, move_to_device, save_checkpoint
from .metrics import compute_classification_metrics, compute_lead_time_metrics


def create_optimizer(model: nn.Module, cfg: ExperimentConfig) -> Optimizer:
    optim_cfg = cfg.optim
    params = (
        model.configure_optim_parameters()
        if hasattr(model, "configure_optim_parameters")
        else model.parameters()
    )

    name = optim_cfg.name.lower()
    if name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=optim_cfg.lr,
            betas=optim_cfg.betas,
            eps=optim_cfg.eps,
            weight_decay=optim_cfg.weight_decay,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=optim_cfg.lr,
            betas=optim_cfg.betas,
            eps=optim_cfg.eps,
            weight_decay=optim_cfg.weight_decay,
        )
    elif name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=optim_cfg.lr,
            momentum=0.9,
            weight_decay=optim_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_cfg.name}")

    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    cfg: ExperimentConfig,
    steps_per_epoch: int,
) -> Optional[_LRScheduler]:
    sched_cfg = cfg.scheduler
    name = (sched_cfg.name or "").lower()

    if name == "":
        return None

    if name == "cosine":
        total_steps = cfg.train.max_epochs * steps_per_epoch
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - cfg.train.accumulation_steps, 1),
            eta_min=sched_cfg.min_lr,
        )
        return scheduler

    if name == "multistep":
        milestones = sched_cfg.__dict__.get("milestones", [int(0.6 * cfg.train.max_epochs)])
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=sched_cfg.__dict__.get("gamma", 0.1),
        )
        return scheduler

    raise ValueError(f"Unsupported scheduler: {sched_cfg.name}")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    cfg: ExperimentConfig,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[_LRScheduler] = None,
    writer: Optional["SummaryWriter"] = None,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()

    train_cfg = cfg.train
    accumulation_steps = max(train_cfg.accumulation_steps, 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(train_cfg.pos_weight, device=device)
        if train_cfg.pos_weight
        else None
    )

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        batch = move_to_device(batch, device)
        inputs = batch["video"]
        labels = batch["label"]

        with autocast(enabled=train_cfg.use_amp and device.type == "cuda"):
            outputs = model(inputs)["logits"]
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        if scaler is not None and train_cfg.use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if train_cfg.grad_clip is not None:
                if scaler is not None and train_cfg.use_amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

            if scaler is not None and train_cfg.use_amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None and not isinstance(scheduler, MultiStepLR):
                scheduler.step()

        loss_meter.update(loss.item() * accumulation_steps, inputs.size(0))

        global_step = epoch * len(dataloader) + step
        if writer is not None:
            writer.add_scalar("train/loss", loss_meter.val, global_step)

        if (step + 1) % train_cfg.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1}/{cfg.train.max_epochs}] Step [{step+1}/{len(dataloader)}] "
                f"Loss: {loss_meter.val:.4f} LR: {lr:.2e}"
            )

    if scheduler is not None and isinstance(scheduler, MultiStepLR):
        scheduler.step()

    return {"loss": loss_meter.avg}


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
    writer: Optional["SummaryWriter"] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(cfg.train.pos_weight, device=device)
        if cfg.train.pos_weight
        else None
    )

    all_logits = []
    all_labels = []
    all_tte = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            inputs = batch["video"]
            labels = batch["label"]
            outputs = model(inputs)["logits"]
            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), inputs.size(0))

            all_logits.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_tte.append(batch["time_to_event"].detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    time_to_event = torch.cat(all_tte)

    metrics = {
        "loss": loss_meter.avg,
    }
    class_metrics = compute_classification_metrics(logits, labels)
    lead_metrics = compute_lead_time_metrics(
        logits,
        labels,
        time_to_event,
        cfg.train.lead_time_thresholds,
    )
    metrics.update(class_metrics)
    metrics.update(lead_metrics)

    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

    return metrics


def save_best_checkpoint(
    metrics: Dict[str, float],
    best_metric: float,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    output_dir: str | Path,
    metric_key: str = "ap",
) -> float:
    current_metric = metrics.get(metric_key, float("nan"))
    if not torch.isfinite(torch.tensor(current_metric)):
        return best_metric

    if current_metric > best_metric:
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metric": current_metric,
        }
        save_path = Path(output_dir) / "best.pt"
        save_checkpoint(checkpoint, save_path)
        return current_metric

    return best_metric

