"""Real-time visualization of crash anticipation predictions.

This script loads the best model checkpoint and streams frames from either:
- A CCD sample (specified via manifest/index or vidname + frames_root), or
- A video file path

For each frame, it maintains a sliding window of length T=clip_len and runs
the model to produce a probability that a crash is imminent. The visualization
overlays the current probability and a binary anticipated/not-anticipated
indicator on the video while saving an annotated MP4.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import random
import time
from pathlib import Path
from typing import Deque, List, Optional, Tuple

# Ensure project root is on sys.path when running as a script
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from crash_anticipation.config import load_config
from crash_anticipation.data.transforms import build_video_transform
from crash_anticipation.models import VideoMAEAnticipation
from crash_anticipation.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time crash anticipation visualization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ccd_baseline.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/ccd_baseline/best.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to an input video file. If provided, overrides CCD options.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/ccd_val.csv",
        help="CCD manifest CSV for selecting a sample (used if --video_path not set)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index in the manifest to visualize (used if --video_path not set)",
    )
    parser.add_argument(
        "--ccd_vidname",
        type=str,
        default=None,
        help="Optional explicit CCD vidname (6-digit string). If provided, use this instead of manifest index.",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default=None,
        help="Optional CCD frames root. Defaults to config.data.frames_root.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output annotated MP4 path. Defaults to outputs/visualizations/{stem}.mp4",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live window while processing",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for anticipated flag",
    )
    parser.add_argument(
        "--stop_sign_path",
        type=str,
        default=None,
        help="Path to a PNG stop sign to overlay when anticipated (with alpha channel)",
    )
    # Batch/run-many options (CCD manifest only)
    parser.add_argument(
        "--batch_pct",
        type=float,
        default=None,
        help="If set (0-100], randomly visualize this percentage of CCD manifest entries",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting batch subset",
    )
    parser.add_argument(
        "--timing_out",
        type=str,
        default=None,
        help="CSV file to append per-stream timings (defaults to outputs/visualizations/timings.csv)",
    )
    return parser.parse_args()


def load_model_and_transform(config_path: str, checkpoint_path: str) -> tuple[VideoMAEAnticipation, dict]:
    cfg = load_config(config_path, overrides={})
    model = VideoMAEAnticipation.from_config(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ckpt = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])  # type: ignore[index]

    transform = build_video_transform(
        frame_size=cfg.data.frame_size,
        is_train=False,
        augmentation={},
    )

    meta = {
        "device": device,
        "clip_len": cfg.data.clip_len,
        "fps": cfg.data.fps,
        "frame_size": cfg.data.frame_size,
        "dataset_type": getattr(cfg.data, "dataset_type", "dad"),
        "frames_root": getattr(cfg.data, "frames_root", None),
    }

    return model, {"transform": transform, "meta": meta}


def bgr_to_chw_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32)
    return tensor


def collect_ccd_frames(frames_root: Path, vidname: str) -> List[Path]:
    # CCD files are named like C_000001_01.jpg, C_000001_02.jpg, ...
    pattern = f"C_{vidname}_*.jpg"
    paths = sorted(frames_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CCD frames found for vidname={vidname} in {frames_root}")
    return paths


def read_manifest_vidname(manifest_path: Path, index: int) -> str:
    import csv

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Index {index} out of range for manifest {manifest_path} with {len(rows)} rows")
    vid = str(rows[index].get("vidname", "")).strip()
    if not vid:
        raise ValueError(f"Manifest {manifest_path} missing 'vidname' at row {index}")
    return vid.zfill(6)


def read_manifest_all_vidnames(manifest_path: Path) -> list[str]:
    import csv as _csv

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        vids = [str(row.get("vidname", "")).strip().zfill(6) for row in reader if row.get("vidname")]
    return vids


def open_video_writer(save_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {save_path}")
    return writer


def overlay_annotations(frame_bgr: np.ndarray, prob: float, threshold: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    # Panel background
    panel_h = max(int(0.12 * h), 40)
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), thickness=-1)

    # Probability bar
    bar_margin = 12
    bar_w = w - 2 * bar_margin
    bar_h = max(int(panel_h * 0.35), 10)
    bar_x1, bar_y1 = bar_margin, h - panel_h + bar_margin
    bar_x2, bar_y2 = bar_x1 + bar_w, bar_y1 + bar_h
    cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2), (64, 64, 64), thickness=2)
    fill_w = int(bar_w * float(np.clip(prob, 0.0, 1.0)))
    cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 165, 255), thickness=-1)

    # Text
    label = f"Crash prob: {prob:.2f}"
    status = "ANTICIPATED" if prob >= threshold else "NORMAL"
    status_color = (0, 255, 0) if prob < threshold else (0, 0, 255)
    t1_org = (bar_x1, bar_y2 + 18)
    t2_org = (bar_x1, bar_y2 + 38)
    cv2.putText(overlay, label, t1_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, status, t2_org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    return overlay


def paste_rgba_onto_bgr(
    dest_bgr: np.ndarray,
    rgba: np.ndarray,
    top: int,
    left: int,
) -> None:
    """Alpha-composite an RGBA image onto a BGR frame in-place at (top, left)."""
    h, w = dest_bgr.shape[:2]
    rh, rw = rgba.shape[:2]
    if top >= h or left >= w:
        return
    rh = min(rh, h - top)
    rw = min(rw, w - left)
    if rh <= 0 or rw <= 0:
        return
    roi = dest_bgr[top : top + rh, left : left + rw]
    rgb = rgba[:rh, :rw, :3]
    alpha = rgba[:rh, :rw, 3:4].astype(np.float32) / 255.0
    inv_alpha = 1.0 - alpha
    # Convert RGB->BGR for blending
    rgb_bgr = rgb[..., ::-1].astype(np.float32)
    roi[:] = (alpha * rgb_bgr + inv_alpha * roi.astype(np.float32)).astype(np.uint8)


def run_stream(
    frames: List[np.ndarray],
    model: VideoMAEAnticipation,
    transform,
    device: torch.device,
    clip_len: int,
    fps: float,
    save_path: Path,
    display: bool,
    threshold: float,
    stop_sign_rgba: Optional[np.ndarray] = None,
    timing_accumulator: Optional[dict] = None,
) -> None:
    # Normalize output size for visualization
    height, width = frames[0].shape[:2]
    target_w = width
    target_h = height

    writer = open_video_writer(save_path, target_w, target_h, fps)

    buffer: Deque[torch.Tensor] = deque(maxlen=clip_len)

    # Prepare a lazily-resized stop sign matching the stream size
    resized_sign: Optional[np.ndarray] = None

    for idx, frame in enumerate(frames):
        # Ensure consistent size
        frame_vis = frame

        # Update buffer with raw CHW tensor (values in [0,255])
        buffer.append(bgr_to_chw_tensor(frame_vis))

        prob = 0.0
        if len(buffer) == clip_len:
            clip = torch.stack(list(buffer), dim=0)  # (T, C, H, W)
            clip = transform(clip)  # (T, C, H, W) normalized/resized
            clip_batched = clip.unsqueeze(0).to(device)  # (1, T, C, H, W)
            with torch.no_grad():
                # Measure forward pass time (synchronize for CUDA)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = model(clip_batched)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                # Accumulate timing if requested
                if timing_accumulator is not None:
                    timing_accumulator["count"] = timing_accumulator.get("count", 0) + 1
                    timing_accumulator["total_s"] = timing_accumulator.get("total_s", 0.0) + (t1 - t0)
                logits = outputs["logits"]  # shape (1,)
                prob = torch.sigmoid(logits).item()

        annotated = overlay_annotations(frame_vis, prob, threshold)

        # Overlay stop sign when anticipated
        if stop_sign_rgba is not None and prob >= threshold:
            if resized_sign is None:
                h, w = annotated.shape[:2]
                # Target width as 18% of frame width, keep aspect
                target_w = max(int(0.18 * w), 64)
                scale = target_w / stop_sign_rgba.shape[1]
                target_h = max(int(stop_sign_rgba.shape[0] * scale), 64)
                resized_sign = cv2.resize(stop_sign_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
            # Place near top-right with margin
            margin = 16
            y = margin
            x = annotated.shape[1] - resized_sign.shape[1] - margin
            paste_rgba_onto_bgr(annotated, resized_sign, y, x)
        writer.write(annotated)

        if display:
            cv2.imshow("Crash Anticipation", annotated)
            # Attempt real-time speed; use small delay
            key = cv2.waitKey(max(int(1000 / max(fps, 1e-3)), 1)) & 0xFF
            if key == 27:  # ESC to quit
                break

    writer.release()
    if display:
        cv2.destroyAllWindows()


def visualize_ccd_sample(
    vidname: str,
    frames_root: Path,
    model: VideoMAEAnticipation,
    transform,
    device: torch.device,
    clip_len: int,
    fps: float,
    out_dir: Path,
    display: bool,
    threshold: float,
    stop_sign_rgba: Optional[np.ndarray],
) -> dict:
    paths = collect_ccd_frames(frames_root, vidname)
    frames_list: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read frame {p}")
        frames_list.append(img)

    save_path = out_dir / f"ccd_{vidname}.mp4"

    # Aggregate timings
    timing_acc = {"count": 0, "total_s": 0.0}
    t_wall0 = time.perf_counter()

    run_stream(
        frames=frames_list,
        model=model,
        transform=transform,
        device=device,
        clip_len=clip_len,
        fps=fps,
        save_path=save_path,
        display=display,
        threshold=threshold,
        stop_sign_rgba=stop_sign_rgba,
        timing_accumulator=timing_acc,
    )

    t_wall1 = time.perf_counter()

    num_inferences = int(timing_acc.get("count", 0))
    total_forward_s = float(timing_acc.get("total_s", 0.0))

    return {
        "id": vidname,
        "num_frames": len(frames_list),
        "fps_input": fps,
        "num_inferences": num_inferences,
        "total_forward_ms": round(total_forward_s * 1000.0, 3),
        "avg_forward_ms": round((total_forward_s / max(num_inferences, 1)) * 1000.0, 3),
        "total_wall_s": round(t_wall1 - t_wall0, 3),
        "output_path": str(save_path),
    }


def main() -> None:
    args = parse_args()
    model, hc = load_model_and_transform(args.config, args.checkpoint)
    transform = hc["transform"]
    meta = hc["meta"]

    fps = float(meta["fps"]) if meta["fps"] else 10.0
    clip_len = int(meta["clip_len"]) if meta["clip_len"] else 16

    # Optional stop sign overlay (load once early so it is available in batch mode)
    stop_sign_rgba: Optional[np.ndarray] = None
    if args.stop_sign_path:
        sign_path = Path(args.stop_sign_path)
        if sign_path.exists():
            tmp = cv2.imread(str(sign_path), cv2.IMREAD_UNCHANGED)
            if tmp is not None and tmp.ndim == 3 and tmp.shape[2] == 4:
                stop_sign_rgba = tmp
            else:
                print(f"Warning: stop sign at {sign_path} is not RGBA PNG; overlay disabled")
        else:
            print(f"Warning: stop sign path not found: {sign_path}")

    # Collect frames to process
    frames_list: List[np.ndarray] = []
    stream_stem = "demo"

    if args.video_path:
        video_path = Path(args.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(cap_fps) if cap_fps and cap_fps > 0 else fps
        stream_stem = video_path.stem
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_list.append(frame)
        cap.release()
    else:
        # CCD frames mode
        frames_root = Path(args.frames_root) if args.frames_root else Path(str(meta["frames_root"]))
        if not frames_root.exists():
            raise FileNotFoundError(f"CCD frames_root not found: {frames_root}")
        out_dir = ensure_dir(Path("outputs/visualizations"))

        # Batch mode if percentage is specified
        if args.batch_pct is not None and args.batch_pct > 0:
            vids = read_manifest_all_vidnames(Path(args.manifest))
            random.Random(args.seed).shuffle(vids)
            k = max(1, int(round(len(vids) * (args.batch_pct / 100.0))))
            selected = vids[:k]

            # Prepare timing CSV
            timing_out = Path(args.timing_out) if args.timing_out else out_dir / "timings.csv"
            write_header = not timing_out.exists()
            with open(timing_out, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "id",
                        "num_frames",
                        "fps_input",
                        "num_inferences",
                        "total_forward_ms",
                        "avg_forward_ms",
                        "total_wall_s",
                        "output_path",
                    ],
                )
                if write_header:
                    writer.writeheader()

                for vid in selected:
                    metrics = visualize_ccd_sample(
                        vidname=vid,
                        frames_root=frames_root,
                        model=model,
                        transform=transform,
                        device=meta["device"],
                        clip_len=clip_len,
                        fps=fps,
                        out_dir=out_dir,
                        display=args.display,
                        threshold=args.threshold,
                        stop_sign_rgba=stop_sign_rgba,
                    )
                    writer.writerow(metrics)

            print(f"Saved {len(selected)} visualizations and timings to {timing_out}")
            return

        # Single CCD sample mode
        if args.ccd_vidname:
            vidname = args.ccd_vidname.zfill(6)
        else:
            vidname = read_manifest_vidname(Path(args.manifest), args.index)
        paths = collect_ccd_frames(frames_root, vidname)
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                raise RuntimeError(f"Failed to read frame {p}")
            frames_list.append(img)
        stream_stem = f"ccd_{vidname}"

    # Prepare output
    out_dir = ensure_dir(Path("outputs/visualizations"))
    save_path = Path(args.save_path) if args.save_path else Path(out_dir) / f"{stream_stem}.mp4"

    run_stream(
        frames=frames_list,
        model=model,
        transform=transform,
        device=meta["device"],
        clip_len=clip_len,
        fps=fps,
        save_path=save_path,
        display=args.display,
        threshold=args.threshold,
        stop_sign_rgba=stop_sign_rgba,
    )

    print(f"Saved visualization to: {save_path}")


if __name__ == "__main__":
    main()


