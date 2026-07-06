"""Latency / footprint comparison between the VideoMAE teacher and the
MobileNetV2-GRU student.

Measures, per model:
- parameter count,
- 16-frame sliding-window inference latency (deployment path for the teacher),
- for the student additionally the *streaming* step (one MobileNet forward +
  GRU update per new frame), which is its real deployment cost.

Usage:
    python benchmark_latency.py --device cuda
    python benchmark_latency.py --device cpu   # embedded-ish sanity check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from crash_anticipation.config import load_config
from crash_anticipation.models import build_model


def time_fn(fn, device: torch.device, warmup: int = 10, iters: int = 100) -> dict:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "fps": round(float(1000.0 / arr.mean()), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "teacher:configs/anticipation.yaml:outputs/videomae_anticipation/best.pt",
            "student:configs/student_distill.yaml:outputs/student_mobilenet_gru/best.pt",
        ],
        help="Entries of the form name:config:checkpoint",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    results = {}
    for entry in args.models:
        name, config_path, ckpt_path = entry.split(":", 2)
        cfg = load_config(config_path, overrides={})
        model = build_model(cfg.model)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()

        clip = torch.randn(1, cfg.data.clip_len, 3, cfg.data.frame_size, cfg.data.frame_size, device=device)
        with torch.no_grad():
            window_stats = time_fn(lambda: model(clip), device, iters=args.iters)

        entry_result = {
            "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "window_16f": window_stats,
        }

        if hasattr(model, "forward_stream"):
            frame = torch.randn(1, 3, cfg.data.frame_size, cfg.data.frame_size, device=device)
            state = {"h": None}

            def stream_step():
                _, state["h"] = model.forward_stream(frame, state["h"])

            with torch.no_grad():
                entry_result["streaming_step"] = time_fn(stream_step, device, iters=args.iters)

        results[name] = entry_result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path("outputs/eval") / f"latency_{args.device}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown table.
    print(f"\n| model | params (M) | 16f window (ms) | window fps | streaming step (ms) | streaming fps |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        stream = r.get("streaming_step")
        stream_ms = f"{stream['mean_ms']}" if stream else "—"
        stream_fps = f"{stream['fps']}" if stream else "—"
        print(
            f"| {name} | {r['parameters_millions']} | {r['window_16f']['mean_ms']} "
            f"| {r['window_16f']['fps']} | {stream_ms} | {stream_fps} |"
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
