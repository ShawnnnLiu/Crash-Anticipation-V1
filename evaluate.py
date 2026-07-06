"""Online evaluation protocol for crash anticipation.

Streams every evaluation video through the model exactly as it would run in
deployment (sliding window, one risk estimate per frame) and scores the
resulting risk curves:

- Video-level AP / ROC-AUC: positives are CCD crash videos scored by their
  maximum risk *before* accident onset (a detection after impact is not
  anticipation); negatives are DAD test videos scored by their maximum risk
  anywhere.
- mTTA / TTA@0.8recall: how many seconds before impact the alarm fires.
- Framewise AP on CCD only: pre-onset frames (tte <= pos horizon) vs early
  frames (tte >= neg horizon) of the *same* videos — a same-domain check that
  the model reads the scene rather than the dataset's look.
- Per-condition breakdown: weather / day-night / ego-involvement.

Usage:
    python evaluate.py --config configs/anticipation.yaml \
        --checkpoint outputs/videomae_anticipation/best.pt --name videomae
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from crash_anticipation.data.datasets.windows import (
    load_ccd_records,
    load_dad_negative_records,
    read_window,
)
from crash_anticipation.inference import load_anticipator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online crash anticipation evaluation")
    parser.add_argument("--config", type=str, default="configs/anticipation.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/videomae_anticipation/best.pt")
    parser.add_argument("--name", type=str, default="videomae", help="Run name for the output folder")
    parser.add_argument("--out_dir", type=str, default="outputs/eval")
    parser.add_argument("--threshold", type=float, default=0.5, help="Operating alarm threshold")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on videos per split (debug)")
    return parser.parse_args()


def stream_video(anticipator, record) -> Dict[str, np.ndarray]:
    """Run the model over every effective frame of a video, in order."""

    anticipator.reset()
    indices = list(range(record.num_frames))
    frames = read_window(record, indices)
    probs, latencies = [], []
    for frame in frames:
        est = anticipator.step(frame)
        probs.append(est.prob)
        latencies.append(est.latency_ms)
    return {"probs": np.asarray(probs), "latency_ms": np.asarray(latencies)}


def evaluate_curves(
    pos_curves: List[Dict],
    neg_curves: List[Dict],
    fps: float,
    threshold: float,
    pos_horizon_s: float,
    neg_horizon_s: float,
) -> Dict:
    # Video-level scores: positives use only pre-onset frames.
    pos_scores = np.array([c["probs"][: c["event"]].max() if c["event"] > 0 else 0.0 for c in pos_curves])
    neg_scores = np.array([c["probs"].max() for c in neg_curves])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    scores = np.concatenate([pos_scores, neg_scores])

    metrics: Dict = {
        "video_ap": float(average_precision_score(labels, scores)),
        "video_auroc": float(roc_auc_score(labels, scores)),
        "n_pos_videos": int(len(pos_scores)),
        "n_neg_videos": int(len(neg_scores)),
    }

    # Operating point at the fixed threshold.
    tp = int((pos_scores >= threshold).sum())
    fp = int((neg_scores >= threshold).sum())
    fn = int((pos_scores < threshold).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics.update(
        {
            "threshold": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-9)),
            "false_alarms_per_neg_video": float(fp / max(len(neg_scores), 1)),
        }
    )

    # Time-to-accident at the operating threshold.
    ttas = []
    for c in pos_curves:
        fired = np.where(c["probs"][: c["event"]] >= threshold)[0]
        if fired.size:
            ttas.append((c["event"] - fired[0]) / fps)
    metrics["mtta_s"] = float(np.mean(ttas)) if ttas else 0.0
    metrics["tta_median_s"] = float(np.median(ttas)) if ttas else 0.0

    # TTA at the threshold achieving >= 0.8 recall (sweep).
    sweep = []
    for theta in np.linspace(0.05, 0.95, 19):
        det, sweep_ttas = 0, []
        for c in pos_curves:
            fired = np.where(c["probs"][: c["event"]] >= theta)[0]
            if fired.size:
                det += 1
                sweep_ttas.append((c["event"] - fired[0]) / fps)
        fa = float((neg_scores >= theta).mean()) if len(neg_scores) else 0.0
        sweep.append(
            {
                "theta": float(theta),
                "recall": det / max(len(pos_curves), 1),
                "mtta_s": float(np.mean(sweep_ttas)) if sweep_ttas else 0.0,
                "false_alarm_rate": fa,
            }
        )
    metrics["sweep"] = sweep
    at80 = [s for s in sweep if s["recall"] >= 0.8]
    metrics["mtta_at_recall80_s"] = at80[-1]["mtta_s"] if at80 else 0.0

    # Framewise same-domain check (CCD only, temporal negatives).
    frame_labels, frame_scores = [], []
    pos_h = int(round(pos_horizon_s * fps))
    neg_h = int(round(neg_horizon_s * fps))
    for c in pos_curves:
        event = c["event"]
        for t, p in enumerate(c["probs"]):
            if event - pos_h <= t < event:
                frame_labels.append(1)
                frame_scores.append(p)
            elif t <= event - neg_h:
                frame_labels.append(0)
                frame_scores.append(p)
    if frame_labels and 0 < sum(frame_labels) < len(frame_labels):
        metrics["framewise_ap_ccd_temporal"] = float(
            average_precision_score(np.asarray(frame_labels), np.asarray(frame_scores))
        )
        metrics["n_frames_pos"] = int(sum(frame_labels))
        metrics["n_frames_neg"] = int(len(frame_labels) - sum(frame_labels))

    return metrics


def condition_breakdown(pos_curves, neg_scores, fps, threshold) -> Dict:
    breakdown: Dict = {}
    for key in ("weather", "timing", "egoinvolve"):
        groups: Dict[str, List] = {}
        for c in pos_curves:
            value = str(c["metadata"].get(key, "unknown"))
            groups.setdefault(value, []).append(c)
        breakdown[key] = {}
        for value, curves in sorted(groups.items()):
            scores_pos = np.array([c["probs"][: c["event"]].max() if c["event"] > 0 else 0.0 for c in curves])
            labels = np.concatenate([np.ones_like(scores_pos), np.zeros_like(neg_scores)])
            scores = np.concatenate([scores_pos, neg_scores])
            ttas = []
            for c in curves:
                fired = np.where(c["probs"][: c["event"]] >= threshold)[0]
                if fired.size:
                    ttas.append((c["event"] - fired[0]) / fps)
            breakdown[key][value] = {
                "n": len(curves),
                "ap": float(average_precision_score(labels, scores)),
                "recall": float((scores_pos >= threshold).mean()),
                "mtta_s": float(np.mean(ttas)) if ttas else 0.0,
            }
    return breakdown


def plot_results(metrics, pos_curves, neg_curves, fps, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
        }
    )
    accent, danger, muted = "#0ea5e9", "#ef4444", "#64748b"

    # 1. Score distributions.
    pos_scores = [c["probs"][: c["event"]].max() if c["event"] > 0 else 0.0 for c in pos_curves]
    neg_scores = [c["probs"].max() for c in neg_curves]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(neg_scores, bins=bins, alpha=0.65, label="normal videos", color=accent)
    ax.hist(pos_scores, bins=bins, alpha=0.65, label="pre-crash (before onset)", color=danger)
    ax.set_xlabel("max risk score")
    ax.set_ylabel("videos")
    ax.set_title("Video-level score separation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=150)
    plt.close(fig)

    # 2. Recall / false-alarm / mTTA trade-off.
    sweep = metrics["sweep"]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    thetas = [s["theta"] for s in sweep]
    ax1.plot(thetas, [s["recall"] for s in sweep], color=danger, label="recall")
    ax1.plot(thetas, [s["false_alarm_rate"] for s in sweep], color=muted, ls="--", label="false alarm rate")
    ax1.set_xlabel("alarm threshold")
    ax1.set_ylabel("rate")
    ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(thetas, [s["mtta_s"] for s in sweep], color=accent, label="mTTA (s)")
    ax2.set_ylabel("mean time-to-accident (s)", color=accent)
    ax2.grid(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center left")
    ax1.set_title("Operating characteristics")
    fig.tight_layout()
    fig.savefig(out_dir / "operating_characteristics.png", dpi=150)
    plt.close(fig)

    # 3. Example risk curves.
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharey=True)
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(pos_curves), size=min(6, len(pos_curves)), replace=False)
    for ax, idx in zip(axes.flat, chosen):
        c = pos_curves[idx]
        t = np.arange(len(c["probs"])) / fps
        ax.plot(t, c["probs"], color=danger, lw=1.8)
        ax.axvline(c["event"] / fps, color="k", ls=":", lw=1.2)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"CCD {c['uid']}", fontsize=9)
        ax.set_xlabel("time (s)")
    axes[0, 0].set_ylabel("risk")
    fig.suptitle("Online risk curves (dotted line = accident onset)")
    fig.tight_layout()
    fig.savefig(out_dir / "example_risk_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    anticipator, cfg = load_anticipator(args.config, args.checkpoint)
    fps = cfg.data.fps

    pos_records = load_ccd_records(cfg.data.val_manifest, cfg.data.frames_root)
    neg_records = load_dad_negative_records(
        cfg.data.dad_test_negatives_root, frame_stride=cfg.data.dad_frame_stride
    )
    if args.limit:
        pos_records = pos_records[: args.limit]
        neg_records = neg_records[: args.limit]

    print(f"Streaming {len(pos_records)} crash videos + {len(neg_records)} normal videos...")

    pos_curves, neg_curves, latencies = [], [], []
    for i, record in enumerate(pos_records):
        result = stream_video(anticipator, record)
        latencies.append(result["latency_ms"])
        pos_curves.append(
            {
                "uid": record.uid,
                "probs": result["probs"],
                "event": int(record.event_index),
                "metadata": record.metadata,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  crash videos: {i + 1}/{len(pos_records)}")
    for i, record in enumerate(neg_records):
        result = stream_video(anticipator, record)
        latencies.append(result["latency_ms"])
        neg_curves.append({"uid": record.uid, "probs": result["probs"]})
        if (i + 1) % 50 == 0:
            print(f"  normal videos: {i + 1}/{len(neg_records)}")

    metrics = evaluate_curves(
        pos_curves,
        neg_curves,
        fps,
        args.threshold,
        cfg.data.pos_horizon_seconds,
        cfg.data.neg_horizon_seconds,
    )
    neg_scores = np.array([c["probs"].max() for c in neg_curves])
    metrics["conditions"] = condition_breakdown(pos_curves, neg_scores, fps, args.threshold)

    all_latency = np.concatenate(latencies)
    n_params = sum(p.numel() for p in anticipator.model.parameters())
    metrics["latency_ms_mean"] = float(all_latency.mean())
    metrics["latency_ms_p95"] = float(np.percentile(all_latency, 95))
    metrics["effective_fps"] = float(1000.0 / all_latency.mean())
    metrics["parameters_millions"] = round(n_params / 1e6, 2)

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pad_to = max(len(c["probs"]) for c in pos_curves + neg_curves)
    np.savez_compressed(
        out_dir / "risk_curves.npz",
        pos_uids=[c["uid"] for c in pos_curves],
        pos_probs=np.array([np.pad(c["probs"], (0, pad_to - len(c["probs"]))) for c in pos_curves]),
        pos_events=[c["event"] for c in pos_curves],
        neg_uids=[c["uid"] for c in neg_curves],
        neg_probs=np.array([np.pad(c["probs"], (0, pad_to - len(c["probs"]))) for c in neg_curves]),
    )
    plot_results(metrics, pos_curves, neg_curves, fps, out_dir)

    print(json.dumps({k: v for k, v in metrics.items() if k not in ("sweep", "conditions")}, indent=2))
    print(f"\nFull results saved to {out_dir}")


if __name__ == "__main__":
    main()
