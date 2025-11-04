"""Entry point for training crash anticipation baseline models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from crash_anticipation.config import load_config
from crash_anticipation.data import build_dataloaders
from crash_anticipation.models import VideoMAEAnticipation
from crash_anticipation.training import (
    create_optimizer,
    create_scheduler,
    evaluate,
    save_best_checkpoint,
    train_one_epoch,
)
from crash_anticipation.utils import (
    ensure_dir,
    save_checkpoint,
    set_seed,
    setup_logging,
)


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' is not in key=value format")
        key, value = pair.split("=", 1)

        try:
            parsed_value: Any = json.loads(value)
        except json.JSONDecodeError:
            # treat as string if not valid JSON
            parsed_value = value

        keys = key.split(".")
        current = overrides
        for sub_key in keys[:-1]:
            current = current.setdefault(sub_key, {})
        current[keys[-1]] = parsed_value

    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crash anticipation baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="Override configuration entries, e.g. train.batch_size=8",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.override or [])
    exp_config = load_config(args.config, overrides)

    logger = setup_logging()
    logger.info("Loaded configuration: %s", exp_config)

    set_seed(exp_config.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    output_dir = ensure_dir(exp_config.train.output_dir)

    writer = SummaryWriter(str(output_dir))

    train_loader, val_loader = build_dataloaders(
        exp_config.data,
        batch_size=exp_config.train.batch_size,
    )
    logger.info(
        "Train batches: %d | Val batches: %d",
        len(train_loader),
        len(val_loader),
    )

    model = VideoMAEAnticipation.from_config(exp_config.model)
    model.to(device)
    logger.info(
        "Model loaded with %d trainable parameters",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = create_optimizer(model, exp_config)
    steps_per_epoch = max(len(train_loader) // exp_config.train.accumulation_steps, 1)
    scheduler = create_scheduler(optimizer, exp_config, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=exp_config.train.use_amp and device.type == "cuda")

    start_epoch = 0
    best_metric = float("-inf")

    if exp_config.train.checkpoint_path:
        ckpt_path = Path(exp_config.train.checkpoint_path)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint.get("epoch", 0)
            best_metric = checkpoint.get("metric", float("-inf"))
            logger.info("Resumed from checkpoint %s", ckpt_path)
        else:
            logger.warning("Checkpoint path %s not found; starting fresh", ckpt_path)

    for epoch in range(start_epoch, exp_config.train.max_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            exp_config,
            scaler=scaler,
            scheduler=scheduler,
            writer=writer,
        )
        logger.info("Epoch %d train metrics: %s", epoch + 1, train_metrics)

        if (epoch + 1) % exp_config.train.eval_interval == 0:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                exp_config,
                writer=writer,
                epoch=epoch + 1,
            )
            logger.info("Epoch %d val metrics: %s", epoch + 1, val_metrics)
            best_metric = save_best_checkpoint(
                val_metrics,
                best_metric,
                epoch + 1,
                model,
                optimizer,
                output_dir,
            )

        if (epoch + 1) % exp_config.train.save_interval == 0:
            checkpoint_path = output_dir / f"epoch_{epoch + 1}.pt"
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metric": best_metric,
                },
                checkpoint_path,
            )

    writer.close()
    logger.info("Training complete. Best AP: %.4f", best_metric)


if __name__ == "__main__":
    main()

