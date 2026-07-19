"""Fast, data-free tests for the core anticipation logic."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crash_anticipation.data.datasets.windows import (
    AnticipationWindowDataset,
    VideoRecord,
    label_window,
)
from crash_anticipation.symbolic.dynamics import TrackDynamics
from crash_anticipation.symbolic.perception import Detection
from crash_anticipation.symbolic.rules import CAUTION, NORMAL, WARNING, AdvisoryEngine
from crash_anticipation.training.losses import AnticipationLoss

FPS = 10.0


def crash_record(event: int = 37, total: int = 50) -> VideoRecord:
    return VideoRecord(
        uid="000001", source="ccd", kind="crash", num_frames=total, event_index=event
    )


def normal_record(total: int = 50) -> VideoRecord:
    return VideoRecord(uid="n1", source="dad", kind="normal", num_frames=total)


class TestLabelWindow:
    def test_post_event_is_full_positive(self):
        label, weight, tte = label_window(crash_record(30), 35, FPS, 1.5, 2.5, 1.2)
        assert label == 1.0 and weight == 1.0 and tte < 0

    def test_anticipation_zone_weight_decays_with_tte(self):
        _, w_close, _ = label_window(crash_record(37), 34, FPS, 1.5, 2.5, 1.2)  # tte 0.3s
        _, w_far, _ = label_window(crash_record(37), 25, FPS, 1.5, 2.5, 1.2)  # tte 1.2s
        assert w_close > w_far > 0
        assert math.isclose(w_far, math.exp(-1.0), rel_tol=1e-6)

    def test_guard_band_has_zero_weight(self):
        label, weight, tte = label_window(crash_record(40), 20, FPS, 1.5, 2.5, 1.2)  # tte 2.0s
        assert weight == 0.0 and 1.5 < tte < 2.5

    def test_temporal_negative(self):
        label, weight, tte = label_window(crash_record(45), 15, FPS, 1.5, 2.5, 1.2)  # tte 3.0s
        assert label == 0.0 and weight == 1.0

    def test_normal_video_is_negative(self):
        label, weight, tte = label_window(normal_record(), 30, FPS, 1.5, 2.5, 1.2)
        assert label == 0.0 and weight == 1.0 and tte == float("inf")


class TestEvalWindows:
    def test_mixed_labels_and_no_guard_band(self):
        records = [crash_record(45), crash_record(30), normal_record()]
        ds = AnticipationWindowDataset(records, is_train=False, fps=FPS)
        labels = []
        for ridx, end in ds.eval_windows:
            label, weight, _ = label_window(ds.records[ridx], end, FPS, 1.5, 2.5, 1.2)
            assert weight > 0, "eval windows must not fall in the guard band"
            labels.append(label)
        assert 0.0 in labels and 1.0 in labels

    def test_early_event_video_has_no_temporal_negative(self):
        ds = AnticipationWindowDataset([crash_record(30)], is_train=False, fps=FPS)
        zones = ds._zone_ranges(ds.records[0])
        assert "neg" not in zones  # event at 3.0s leaves no room before the 2.5s horizon


class TestTrackDynamics:
    def _run(self, boxes):
        dynamics = TrackDynamics(fps=FPS)
        facts = []
        for box in boxes:
            det = Detection(track_id=1, cls_name="car", xyxy=np.asarray(box, dtype=np.float32), conf=0.9)
            facts = dynamics.update([det], (720, 1280))
        return facts[0]

    @staticmethod
    def _project(x_m, z_m, width_m=1.8, height_m=1.5, focal_px=1000.0, cx0=640.0, ground_y=500.0):
        """Pinhole projection of a car at lateral offset x_m, depth z_m."""
        w = focal_px * width_m / z_m
        h = focal_px * height_m / z_m
        cx = cx0 + focal_px * x_m / z_m
        return [cx - w / 2, ground_y - h, cx + w / 2, ground_y]

    def test_head_on_approach_gets_finite_falling_ttc(self):
        # Box centered ahead, growing 12% per frame -> strong looming.
        boxes = []
        size = 60.0
        for _ in range(6):
            half = size / 2
            boxes.append([640 - half, 400 - half, 640 + half, 400 + half])
            size *= 1.12
        fact = self._run(boxes)
        assert fact.ttc_s < 2.0
        assert fact.heading == "approaching"
        assert fact.in_path

    def test_passing_object_is_not_collision_course(self):
        # Object looming but sweeping rapidly to the right edge -> being passed.
        boxes = []
        size, cx = 80.0, 900.0
        for _ in range(6):
            half = size / 2
            boxes.append([cx - half, 500 - half, cx + half, 500 + half])
            size *= 1.15
            cx += 60
        fact = self._run(boxes)
        assert fact.ttc_s == float("inf")
        assert fact.heading == "passing"

    def test_overtaken_adjacent_lane_car_is_passing(self):
        # Ego closes at 10 m/s on a car one lane over (3.5 m). It looms hard
        # and barely drifts in the image, but its projected miss is ~2 car
        # widths -> not a collision course.
        boxes = [self._project(3.5, 30.0 - i) for i in range(10)]
        fact = self._run(boxes)
        assert fact.ttc_s == float("inf")
        assert fact.heading == "passing"
        assert fact.miss_widths > 1.5
        assert not fact.in_path
        assert not fact.converging

    def test_cut_in_on_collision_course_gets_finite_ttc(self):
        # Car merging from one lane over onto a corner-impact course
        # (closest approach 0.9 m): lateral drift is visible, yet the
        # projected miss stays ~0.5 widths -> alarm.
        boxes = [self._project(3.5 - 0.13 * i, 16.0 - 0.8 * i) for i in range(10)]
        fact = self._run(boxes)
        assert fact.ttc_s < 2.5
        assert fact.heading == "approaching"
        assert fact.miss_widths < 1.5

    def test_corridor_narrows_with_distance(self):
        # A distant adjacent-lane car sits near the image center yet is NOT
        # in-path; a car straight ahead at the same depth is.
        adjacent = self._run([self._project(3.5, 40.0)] * 6)
        ahead = self._run([self._project(0.0, 40.0)] * 6)
        assert 0.30 < (adjacent.xyxy[0] + adjacent.xyxy[2]) / 2 / 1280 < 0.70
        assert not adjacent.in_path
        assert ahead.in_path


class TestAdvisoryEngine:
    def _fact(self, **kw):
        from crash_anticipation.symbolic.dynamics import ObjectFacts

        base = dict(
            track_id=1,
            cls_name="car",
            xyxy=np.array([500, 300, 800, 500], dtype=np.float32),
            zone="center",
            in_path=True,
            ttc_s=1.2,
            looming=0.8,
            lateral_v=0.0,
            size_frac=0.1,
            miss_widths=0.2,
            converging=True,
            heading="approaching",
        )
        base.update(kw)
        return ObjectFacts(**base)

    def test_confirmed_threat_gives_evasive_warning(self):
        adv = AdvisoryEngine().decide(0.9, [self._fact()])
        assert adv.level == WARNING
        assert "BRAKE" in adv.command
        assert "TTC" in adv.rationale

    def test_high_risk_without_track_still_brakes(self):
        adv = AdvisoryEngine().decide(0.9, [])
        assert adv.level == WARNING and adv.command == "BRAKE"

    def test_side_threat_steers_away(self):
        adv = AdvisoryEngine().decide(0.9, [self._fact(zone="left", in_path=False)])
        assert "STEER RIGHT" in adv.command

    def test_quiet_scene_is_normal(self):
        adv = AdvisoryEngine().decide(0.05, [self._fact(ttc_s=float("inf"), heading="stable", in_path=False)])
        assert adv.level == NORMAL

    def test_moderate_risk_is_caution(self):
        adv = AdvisoryEngine().decide(0.5, [])
        assert adv.level == CAUTION

    def test_passing_object_is_not_a_threat(self):
        passing = self._fact(
            heading="passing", ttc_s=float("inf"), miss_widths=2.1, converging=False, in_path=False
        )
        adv = AdvisoryEngine().decide(0.1, [passing])
        assert adv.level == NORMAL
        assert adv.threat is None

    def test_crosser_is_threat_only_while_converging(self):
        engine = AdvisoryEngine()
        away = self._fact(heading="crossing", ttc_s=float("inf"), converging=False, in_path=False)
        toward = self._fact(heading="crossing", ttc_s=float("inf"), converging=True, in_path=False)
        assert engine._primary_threat([away]) is None
        assert engine._primary_threat([toward]) is not None


class TestAnticipationLoss:
    def test_zero_weight_samples_are_ignored(self):
        loss_fn = AnticipationLoss()
        logits = torch.tensor([5.0, -5.0])
        labels = torch.tensor([0.0, 0.0])
        # First sample is badly wrong but weight 0 -> only second contributes.
        weights = torch.tensor([0.0, 1.0])
        loss = loss_fn(logits, labels, weights)
        reference = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[1:], labels[1:]
        )
        assert torch.isclose(loss, reference, atol=1e-6)

    def test_uniform_weights_match_plain_bce(self):
        loss_fn = AnticipationLoss()
        logits = torch.randn(8)
        labels = (torch.rand(8) > 0.5).float()
        loss = loss_fn(logits, labels, torch.ones(8))
        reference = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        assert torch.isclose(loss, reference, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
