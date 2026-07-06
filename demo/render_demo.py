"""Portfolio-grade demo renderer for crash anticipation.

Streams a dashcam clip through the full stack — neural risk model, YOLO +
ByteTrack perception, symbolic dynamics and the advisory rule engine — and
renders an automotive-HUD style visualization:

- left: the footage with tracked-agent markers, threat highlighting and an
  alert banner;
- right: a live dashboard with the risk gauge, rolling risk timeline,
  the active advisory (command + rationale) and per-agent TTC readouts.

All overlay graphics are drawn with PIL at 2x supersampling and downsampled
onto the frame, so lines and typography are anti-aliased instead of the
jagged OpenCV primitives.

Examples:
    # Render specific CCD validation clips
    python demo/render_demo.py --ccd 000015 000123 --gif

    # Auto-pick presentable clips + one normal (no-crash) contrast clip
    python demo/render_demo.py --auto 4 --include-negative
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crash_anticipation.data.datasets.windows import (
    VideoRecord,
    load_ccd_records,
    load_dad_negative_records,
    read_window,
)
from crash_anticipation.inference import load_anticipator
from crash_anticipation.symbolic import AdvisoryEngine, TrackDynamics
from crash_anticipation.symbolic.rules import CAUTION, NORMAL, WARNING

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

SS = 2  # supersampling factor for overlay graphics

COL_BG = (11, 18, 32)  # sidebar background
COL_PANEL = (17, 26, 46)
COL_EDGE = (51, 65, 85)
COL_TEXT = (226, 232, 240)
COL_MUTED = (148, 163, 184)
COL_FAINT = (100, 116, 139)
COL_ACCENT = (34, 211, 238)  # cyan
COL_OK = (52, 211, 153)  # green
COL_WARN = (245, 158, 11)  # amber
COL_DANGER = (239, 68, 68)  # red

LEVEL_COLOR = {NORMAL: COL_OK, CAUTION: COL_WARN, WARNING: COL_DANGER}

FONT_DIR = Path("C:/Windows/Fonts")


def _font(name_options: Sequence[str], size: int) -> ImageFont.FreeTypeFont:
    for name in name_options:
        path = FONT_DIR / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


class Fonts:
    """Font set at supersampled resolution."""

    def __init__(self) -> None:
        self.title = _font(["bahnschrift.ttf", "arialbd.ttf"], 21 * SS)
        self.subtitle = _font(["arial.ttf"], 13 * SS)
        self.risk_big = _font(["bahnschrift.ttf", "arialbd.ttf"], 52 * SS)
        self.level = _font(["bahnschrift.ttf", "arialbd.ttf"], 17 * SS)
        self.command = _font(["bahnschrift.ttf", "arialbd.ttf"], 24 * SS)
        self.body = _font(["arial.ttf"], 14 * SS)
        self.small = _font(["arial.ttf"], 12 * SS)
        self.mono = _font(["consola.ttf"], 13 * SS)
        self.mono_small = _font(["consola.ttf"], 11 * SS)
        self.banner = _font(["bahnschrift.ttf", "arialbd.ttf"], 22 * SS)
        self.chip = _font(["bahnschrift.ttf", "arialbd.ttf"], 13 * SS)


@dataclass
class FrameState:
    """Everything the renderer needs for one frame."""

    t_index: int
    time_s: float
    prob_raw: float
    prob_smooth: float
    level: str
    command: str
    rationale: str
    facts: List = field(default_factory=list)
    threat_id: Optional[int] = None
    latency_ms: float = 0.0
    history: List[float] = field(default_factory=list)
    event_index: Optional[int] = None
    alert_time_s: Optional[float] = None
    clip_label: str = ""
    model_label: str = ""


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class HUDRenderer:
    def __init__(self, video_w: int, video_h: int, fps: float, sidebar_w: int = 392) -> None:
        self.vw = video_w
        self.vh = video_h
        self.sw = sidebar_w
        self.W = video_w + sidebar_w
        self.H = video_h
        self.fps = fps
        self.fonts = Fonts()

    # -- public API ----------------------------------------------------------

    def render(self, frame_bgr: np.ndarray, s: FrameState) -> np.ndarray:
        canvas = np.empty((self.H, self.W, 3), dtype=np.uint8)
        canvas[:, : self.vw] = frame_bgr
        canvas[:, self.vw :] = COL_BG[::-1]  # numpy is BGR

        overlay = Image.new("RGBA", (self.W * SS, self.H * SS), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        self._draw_video_overlays(draw, s)
        self._draw_sidebar(draw, s)

        overlay = overlay.resize((self.W, self.H), Image.LANCZOS)
        base = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
        composed = Image.alpha_composite(base, overlay).convert("RGB")
        return cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)

    # -- video-side drawing ----------------------------------------------------

    def _draw_video_overlays(self, d: ImageDraw.ImageDraw, s: FrameState) -> None:
        # Agent markers.
        for fact in s.facts:
            is_threat = s.threat_id is not None and fact.track_id == s.threat_id
            urgent = fact.ttc_s < 3.0 or is_threat
            if is_threat and s.level == WARNING:
                color = COL_DANGER
            elif urgent:
                color = COL_WARN
            else:
                color = (*COL_MUTED, 200)[:3]
            self._corner_box(d, fact.xyxy, color, thick=(4 if is_threat else 2))
            label = f"{fact.cls_name} #{fact.track_id}"
            if fact.ttc_s != float("inf") and fact.ttc_s < 9.5:
                label += f"  TTC {fact.ttc_s:.1f}s"
            self._label_chip(d, fact.xyxy, label, color)

        # Alert banner.
        if s.level in (CAUTION, WARNING):
            color = LEVEL_COLOR[s.level]
            text = f"{s.command}" if s.level == WARNING else "CAUTION"
            title = "COLLISION ANTICIPATED" if s.level == WARNING else "ELEVATED RISK"
            self._banner(d, title, text, color)

        # Ground-truth impact flash (2 frames) + tag.
        if s.event_index is not None and s.t_index >= s.event_index:
            if s.t_index - s.event_index <= 1:
                d.rectangle(
                    [0, 0, self.vw * SS, self.vh * SS],
                    fill=(255, 80, 80, 60),
                )
            self._impact_tag(d)

        # Footer strip: clip label.
        pad = 10 * SS
        d.text(
            (pad, (self.vh - 24) * SS),
            s.clip_label,
            font=self.fonts.small,
            fill=(*COL_TEXT, 190),
            stroke_width=SS,
            stroke_fill=(0, 0, 0, 160),
        )

        # Bottom risk strip.
        strip_h = 5 * SS
        y0 = self.vh * SS - strip_h
        d.rectangle([0, y0, self.vw * SS, self.vh * SS], fill=(0, 0, 0, 110))
        fill_w = int(self.vw * SS * min(max(s.prob_smooth, 0.0), 1.0))
        d.rectangle([0, y0, fill_w, self.vh * SS], fill=(*self._risk_color(s.prob_smooth), 235))

    def _corner_box(self, d: ImageDraw.ImageDraw, xyxy, color, thick: int = 2) -> None:
        x1, y1, x2, y2 = [int(v * SS) for v in xyxy]
        w, h = x2 - x1, y2 - y1
        tick = max(min(w, h) // 4, 8 * SS)
        t = thick * SS
        c = (*color, 255)
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            d.line([(cx, cy), (cx + dx * tick, cy)], fill=c, width=t)
            d.line([(cx, cy), (cx, cy + dy * tick)], fill=c, width=t)

    def _label_chip(self, d: ImageDraw.ImageDraw, xyxy, text: str, color) -> None:
        x1, y1 = int(xyxy[0] * SS), int(xyxy[1] * SS)
        pad = 5 * SS
        tw = d.textlength(text, font=self.fonts.chip)
        th = 18 * SS
        y_top = max(y1 - th - 6 * SS, 0)
        d.rounded_rectangle(
            [x1, y_top, x1 + tw + 2 * pad, y_top + th],
            radius=4 * SS,
            fill=(10, 15, 28, 210),
            outline=(*color, 255),
            width=SS,
        )
        d.text((x1 + pad, y_top + 2 * SS), text, font=self.fonts.chip, fill=(*COL_TEXT, 255))

    def _banner(self, d: ImageDraw.ImageDraw, title: str, command: str, color) -> None:
        cx = self.vw * SS // 2
        bw, bh = 460 * SS, 64 * SS
        y0 = 18 * SS
        d.rounded_rectangle(
            [cx - bw // 2, y0, cx + bw // 2, y0 + bh],
            radius=10 * SS,
            fill=(10, 15, 28, 215),
            outline=(*color, 255),
            width=2 * SS,
        )
        # Warning triangle.
        tx = cx - bw // 2 + 26 * SS
        ty = y0 + bh // 2
        r = 15 * SS
        d.polygon(
            [(tx, ty - r), (tx - int(r * 0.9), ty + r * 0.75), (tx + int(r * 0.9), ty + r * 0.75)],
            outline=(*color, 255),
            width=2 * SS,
        )
        d.text((tx - 2 * SS, ty - 6 * SS), "!", font=self.fonts.chip, fill=(*color, 255))
        text_x = tx + 30 * SS
        d.text((text_x, y0 + 9 * SS), title, font=self.fonts.chip, fill=(*color, 255))
        d.text((text_x, y0 + 27 * SS), command, font=self.fonts.banner, fill=(*COL_TEXT, 255))

    def _impact_tag(self, d: ImageDraw.ImageDraw) -> None:
        text = "GROUND-TRUTH IMPACT"
        tw = d.textlength(text, font=self.fonts.chip)
        pad = 8 * SS
        x1 = self.vw * SS - tw - 3 * pad
        y1 = 16 * SS
        d.rounded_rectangle(
            [x1, y1, x1 + tw + 2 * pad, y1 + 24 * SS],
            radius=5 * SS,
            fill=(120, 20, 20, 200),
            outline=(*COL_DANGER, 255),
            width=SS,
        )
        d.text((x1 + pad, y1 + 5 * SS), text, font=self.fonts.chip, fill=(*COL_TEXT, 255))

    # -- sidebar drawing -------------------------------------------------------

    def _risk_color(self, p: float) -> Tuple[int, int, int]:
        if p < 0.4:
            return COL_OK
        if p < 0.65:
            return COL_WARN
        return COL_DANGER

    def _draw_sidebar(self, d: ImageDraw.ImageDraw, s: FrameState) -> None:
        x0 = self.vw * SS
        pad = 18 * SS
        x = x0 + pad
        w = self.sw * SS - 2 * pad

        # Header.
        y = 20 * SS
        d.text((x, y), "CRASH ANTICIPATION", font=self.fonts.title, fill=(*COL_TEXT, 255))
        y += 30 * SS
        d.text((x, y), s.model_label, font=self.fonts.subtitle, fill=(*COL_MUTED, 255))
        y += 26 * SS
        d.line([(x, y), (x0 + self.sw * SS - pad, y)], fill=(*COL_EDGE, 255), width=SS)

        # Risk readout + level chip.
        y += 14 * SS
        risk_color = self._risk_color(s.prob_smooth)
        d.text((x, y), f"{s.prob_smooth * 100:3.0f}", font=self.fonts.risk_big, fill=(*risk_color, 255))
        num_w = d.textlength(f"{s.prob_smooth * 100:3.0f}", font=self.fonts.risk_big)
        d.text((x + num_w + 8 * SS, y + 32 * SS), "% RISK", font=self.fonts.level, fill=(*COL_MUTED, 255))

        chip_w, chip_h = 108 * SS, 30 * SS
        cx1 = x0 + self.sw * SS - pad - chip_w
        cy1 = y + 14 * SS
        level_color = LEVEL_COLOR[s.level]
        d.rounded_rectangle(
            [cx1, cy1, cx1 + chip_w, cy1 + chip_h],
            radius=6 * SS,
            fill=(*level_color, 40),
            outline=(*level_color, 255),
            width=SS,
        )
        lw = d.textlength(s.level, font=self.fonts.level)
        d.text((cx1 + (chip_w - lw) / 2, cy1 + 5 * SS), s.level, font=self.fonts.level, fill=(*level_color, 255))

        # Risk timeline.
        y += 74 * SS
        d.text((x, y), "RISK TIMELINE", font=self.fonts.small, fill=(*COL_FAINT, 255))
        y += 20 * SS
        ch_h = 120 * SS
        self._chart(d, x, y, w, ch_h, s)
        y += ch_h + 16 * SS

        # Advisory panel.
        d.text((x, y), "ADVISORY", font=self.fonts.small, fill=(*COL_FAINT, 255))
        y += 20 * SS
        panel_h = 96 * SS
        d.rounded_rectangle(
            [x, y, x + w, y + panel_h],
            radius=8 * SS,
            fill=(*COL_PANEL, 235),
            outline=(*COL_EDGE, 255),
            width=SS,
        )
        cmd_color = LEVEL_COLOR[s.level] if s.level != NORMAL else COL_ACCENT
        d.text((x + 14 * SS, y + 12 * SS), s.command, font=self.fonts.command, fill=(*cmd_color, 255))
        for i, line in enumerate(self._wrap(d, s.rationale, self.fonts.body, w - 28 * SS)[:2]):
            d.text((x + 14 * SS, y + 48 * SS + i * 20 * SS), line, font=self.fonts.body, fill=(*COL_MUTED, 255))
        y += panel_h + 16 * SS

        # Tracked agents.
        d.text((x, y), "TRACKED AGENTS", font=self.fonts.small, fill=(*COL_FAINT, 255))
        y += 20 * SS
        facts = sorted(s.facts, key=lambda f: (f.ttc_s, -f.size_frac))[:4]
        if not facts:
            d.text((x, y + 4 * SS), "none in view", font=self.fonts.body, fill=(*COL_FAINT, 255))
        row_h = 26 * SS
        for fact in facts:
            is_threat = s.threat_id is not None and fact.track_id == s.threat_id
            dot_color = COL_DANGER if (is_threat and s.level == WARNING) else (
                COL_WARN if fact.ttc_s < 3.5 else COL_MUTED
            )
            d.ellipse([x, y + 8 * SS, x + 8 * SS, y + 16 * SS], fill=(*dot_color, 255))
            name = f"{fact.cls_name} #{fact.track_id}"
            d.text((x + 16 * SS, y + 3 * SS), name, font=self.fonts.body, fill=(*COL_TEXT, 255))
            zone_txt = {"left": "L", "center": "C", "right": "R"}[fact.zone]
            ttc_txt = f"TTC {fact.ttc_s:4.1f}s" if fact.ttc_s < 20 else "TTC  --  "
            info = f"{zone_txt}  {ttc_txt}"
            iw = d.textlength(info, font=self.fonts.mono)
            d.text((x + w - iw, y + 4 * SS), info, font=self.fonts.mono, fill=(*COL_MUTED, 255))
            y += row_h

        # Footer: telemetry.
        fy = (self.vh - 30) * SS
        d.line([(x, fy - 8 * SS), (x0 + self.sw * SS - pad, fy - 8 * SS)], fill=(*COL_EDGE, 255), width=SS)
        telemetry = f"t {s.time_s:4.1f}s   inference {s.latency_ms:4.1f} ms   {1000.0 / max(s.latency_ms, 1e-3):4.0f} fps"
        d.text((x, fy), telemetry, font=self.fonts.mono_small, fill=(*COL_FAINT, 255))
        if s.alert_time_s is not None and s.event_index is not None and s.t_index >= s.event_index:
            lead = s.event_index / self.fps - s.alert_time_s
            msg = f"alerted {lead:.1f}s before impact"
            mw = d.textlength(msg, font=self.fonts.mono_small)
            d.text((x + w - mw, fy - 24 * SS), msg, font=self.fonts.mono_small, fill=(*COL_OK, 255))

    def _chart(self, d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, s: FrameState) -> None:
        d.rounded_rectangle(
            [x, y, x + w, y + h],
            radius=8 * SS,
            fill=(*COL_PANEL, 235),
            outline=(*COL_EDGE, 255),
            width=SS,
        )
        inner_pad = 10 * SS
        cx, cy = x + inner_pad, y + inner_pad
        cw, ch = w - 2 * inner_pad, h - 2 * inner_pad

        total = max(len(s.history), int(5 * self.fps))

        def pt(i: int, p: float) -> Tuple[float, float]:
            return (cx + cw * i / max(total - 1, 1), cy + ch * (1.0 - min(max(p, 0.0), 1.0)))

        # Threshold lines.
        for level_p, col in ((0.65, COL_DANGER), (0.40, COL_WARN)):
            ly = cy + ch * (1 - level_p)
            for seg in range(int(cx), int(cx + cw), 12 * SS):
                d.line([(seg, ly), (seg + 6 * SS, ly)], fill=(*col, 110), width=SS)

        # Ground-truth impact marker (drawn once reached).
        if s.event_index is not None and s.t_index >= s.event_index:
            ex = cx + cw * s.event_index / max(total - 1, 1)
            for seg in range(int(cy), int(cy + ch), 10 * SS):
                d.line([(ex, seg), (ex, seg + 5 * SS)], fill=(*COL_TEXT, 140), width=SS)

        if len(s.history) >= 2:
            pts = [pt(i, p) for i, p in enumerate(s.history)]
            # Area fill.
            poly = pts + [(pts[-1][0], cy + ch), (pts[0][0], cy + ch)]
            d.polygon(poly, fill=(*self._risk_color(s.prob_smooth), 45))
            d.line(pts, fill=(*self._risk_color(s.prob_smooth), 255), width=2 * SS, joint="curve")
            # Current point.
            px, py = pts[-1]
            d.ellipse([px - 4 * SS, py - 4 * SS, px + 4 * SS, py + 4 * SS], fill=(*COL_TEXT, 255))

        # Axis labels.
        d.text((cx, cy + ch + 2 * SS), "0s", font=self.fonts.mono_small, fill=(*COL_FAINT, 255))
        end_lbl = f"{total / self.fps:.0f}s"
        elw = d.textlength(end_lbl, font=self.fonts.mono_small)
        d.text((cx + cw - elw, cy + ch + 2 * SS), end_lbl, font=self.fonts.mono_small, fill=(*COL_FAINT, 255))

    @staticmethod
    def _wrap(d: ImageDraw.ImageDraw, text: str, font, max_w: int) -> List[str]:
        words = text.split()
        lines, current = [], ""
        for word in words:
            trial = (current + " " + word).strip()
            if d.textlength(trial, font=font) <= max_w:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def encode_h264(raw_path: Path, out_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "slow", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    return subprocess.run(cmd, check=False).returncode == 0


def export_gif(mp4_path: Path, gif_path: Path, width: int = 720, fps: int = 10) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    filters = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
    )
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4_path), "-vf", filters, str(gif_path)]
    return subprocess.run(cmd, check=False).returncode == 0


def render_clip(
    record: VideoRecord,
    anticipator,
    renderer_cls,
    engine: AdvisoryEngine,
    fps: float,
    out_dir: Path,
    model_label: str,
    perception=None,
    make_gif: bool = False,
) -> Path:
    frames = read_window(record, list(range(record.num_frames)))
    vh, vw = frames[0].shape[:2]
    renderer = renderer_cls(vw, vh, fps)

    anticipator.reset()
    dynamics = TrackDynamics(fps=fps)
    if perception is not None:
        perception.reset()

    raw_path = out_dir / f"_{record.uid}_raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (renderer.W, renderer.H))

    history: List[float] = []
    alert_time_s: Optional[float] = None
    level_hold = 0

    src = "CCD validation" if record.source == "ccd" else "DAD test (no crash)"
    clip_label = f"{src} · clip {record.uid} · {fps:.0f} fps"

    for t, frame in enumerate(frames):
        est = anticipator.step(frame)
        history.append(est.prob_smooth)

        facts = []
        if perception is not None:
            detections = perception.track(frame)
            facts = dynamics.update(detections, frame.shape)

        advisory = engine.decide(est.prob_smooth, facts)

        # Hold WARNING for a few frames to avoid flicker.
        if advisory.level == WARNING:
            level_hold = 6
        elif level_hold > 0 and advisory.level == NORMAL:
            level_hold -= 1
            advisory.level = CAUTION

        if advisory.level == WARNING and alert_time_s is None:
            alert_time_s = t / fps

        state = FrameState(
            t_index=t,
            time_s=t / fps,
            prob_raw=est.prob,
            prob_smooth=est.prob_smooth,
            level=advisory.level,
            command=advisory.command,
            rationale=advisory.rationale,
            facts=facts,
            threat_id=advisory.threat.track_id if advisory.threat else None,
            latency_ms=est.latency_ms,
            history=history,
            event_index=record.event_index,
            alert_time_s=alert_time_s,
            clip_label=clip_label,
            model_label=model_label,
        )
        writer.write(renderer.render(frame, state))

    writer.release()

    final_path = out_dir / f"{record.source}_{record.uid}.mp4"
    if encode_h264(raw_path, final_path):
        raw_path.unlink()
    else:
        raw_path.rename(final_path)

    if make_gif:
        export_gif(final_path, final_path.with_suffix(".gif"))

    return final_path


def pick_presentable(records: List[VideoRecord], anticipator, fps: float, k: int) -> List[VideoRecord]:
    """Choose clips where the model tells a clean story: quiet start, early alarm."""

    scored = []
    for record in records:
        frames = read_window(record, list(range(record.num_frames)))
        anticipator.reset()
        probs = [anticipator.step(f).prob for f in frames]
        probs = np.asarray(probs)
        event = record.event_index or len(probs)
        pre = probs[:event]
        if len(pre) < 10:
            continue
        quiet_start = pre[:8].mean()
        fired = np.where(pre >= 0.5)[0]
        tta = (event - fired[0]) / fps if fired.size else 0.0
        # Reward early confident alarms from a quiet baseline.
        score = (pre.max() - quiet_start) + 0.35 * tta - 0.5 * quiet_start
        scored.append((score, record))
    scored.sort(key=lambda x: -x[0])
    picked = [r for _, r in scored[:k]]
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="Render crash anticipation demo videos")
    parser.add_argument("--config", type=str, default="configs/anticipation.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/videomae_anticipation/best.pt")
    parser.add_argument("--model_label", type=str, default="VideoMAE-S encoder · YOLO11n symbolic layer")
    parser.add_argument("--ccd", nargs="*", default=None, help="Explicit CCD vidnames to render")
    parser.add_argument("--auto", type=int, default=0, help="Auto-select N presentable CCD val clips")
    parser.add_argument("--scan", type=int, default=80, help="How many val clips to scan for --auto")
    parser.add_argument("--include-negative", action="store_true", help="Also render a DAD normal clip")
    parser.add_argument("--no-symbolic", action="store_true", help="Disable YOLO/tracking overlays")
    parser.add_argument("--gif", action="store_true", help="Also export a GIF per clip")
    parser.add_argument("--out_dir", type=str, default="outputs/demos")
    args = parser.parse_args()

    anticipator, cfg = load_anticipator(args.config, args.checkpoint)
    fps = cfg.data.fps
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    perception = None
    if not args.no_symbolic:
        from crash_anticipation.symbolic.perception import ObjectPerception

        perception = ObjectPerception()

    engine = AdvisoryEngine()

    ccd_records = load_ccd_records(cfg.data.val_manifest, cfg.data.frames_root)
    by_uid = {r.uid: r for r in ccd_records}

    targets: List[VideoRecord] = []
    if args.ccd:
        targets += [by_uid[v.zfill(6)] for v in args.ccd]
    if args.auto:
        print(f"Scanning {args.scan} validation clips for presentable stories...")
        targets += pick_presentable(ccd_records[: args.scan], anticipator, fps, args.auto)
    if args.include_negative:
        negatives = load_dad_negative_records(
            cfg.data.dad_test_negatives_root, frame_stride=cfg.data.dad_frame_stride
        )
        targets.append(negatives[3])

    if not targets:
        parser.error("Nothing to render: pass --ccd, --auto N and/or --include-negative")

    for record in targets:
        path = render_clip(
            record,
            anticipator,
            HUDRenderer,
            engine,
            fps,
            out_dir,
            args.model_label,
            perception=perception,
            make_gif=args.gif,
        )
        print(f"rendered {path}")


if __name__ == "__main__":
    main()
