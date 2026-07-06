"""Tile rendered demo clips into a 2x2 showcase reel (ffmpeg xstack).

Usage:
    python demo/make_reel.py outputs/demos/ccd_000015.mp4 outputs/demos/ccd_000123.mp4 \
        outputs/demos/ccd_000475.mp4 outputs/demos/dad_000833.mp4 \
        --out outputs/demos/reel.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2x2 demo reel")
    parser.add_argument("clips", nargs=4, help="Four rendered demo mp4s")
    parser.add_argument("--out", type=str, default="outputs/demos/reel.mp4")
    parser.add_argument("--width", type=int, default=1672, help="Output width")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found on PATH")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    for clip in args.clips:
        cmd += ["-i", str(clip)]
    filter_graph = (
        "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[grid];"
        f"[grid]scale={args.width}:-2:flags=lanczos[out]"
    )
    cmd += [
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "slow", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    print(f"reel written to {out}")


if __name__ == "__main__":
    main()
