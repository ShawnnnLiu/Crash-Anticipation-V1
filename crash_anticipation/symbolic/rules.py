"""Transparent rule engine: neural risk x symbolic facts -> advisory.

The neural network decides *whether* the situation is dangerous; the rules
decide *what to do about it* and can always explain themselves. Every
advisory carries the rule that fired and the facts it fired on, so the
system's recommendations are auditable — the core promise of the
neurosymbolic design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .dynamics import ObjectFacts

# Alert levels, in increasing severity.
NORMAL = "NORMAL"
CAUTION = "CAUTION"
WARNING = "WARNING"


@dataclass
class Advisory:
    level: str  # NORMAL | CAUTION | WARNING
    command: str  # e.g. "BRAKE", "BRAKE + STEER LEFT", "COVER BRAKE", "MONITOR"
    rationale: str  # human-readable explanation citing the fired rule
    neural_risk: float
    threat: Optional[ObjectFacts] = None
    secondary: List[ObjectFacts] = field(default_factory=list)

    @property
    def min_ttc(self) -> float:
        return self.threat.ttc_s if self.threat else float("inf")


class AdvisoryEngine:
    def __init__(
        self,
        risk_warning: float = 0.65,
        risk_caution: float = 0.40,
        ttc_warning: float = 2.0,
        ttc_caution: float = 3.5,
    ) -> None:
        self.risk_warning = risk_warning
        self.risk_caution = risk_caution
        self.ttc_warning = ttc_warning
        self.ttc_caution = ttc_caution

    def decide(self, neural_risk: float, facts: List[ObjectFacts]) -> Advisory:
        threat = self._primary_threat(facts)
        others = [f for f in facts if threat is None or f.track_id != threat.track_id]

        # Rule 1: model alarmed AND a closing object confirms it -> evasive command.
        if neural_risk >= self.risk_warning and threat is not None and threat.ttc_s <= self.ttc_warning:
            command = self._evasive_command(threat, facts)
            rationale = (
                f"{threat.cls_name} {self._describe_position(threat)} closing, "
                f"TTC {threat.ttc_s:.1f}s; learned risk {neural_risk:.0%}"
            )
            return Advisory(WARNING, command, rationale, neural_risk, threat, others)

        # Rule 2: model alarmed but no track confirms an imminent collision ->
        # brake on learned risk alone (occluded or unusual hazard).
        if neural_risk >= self.risk_warning:
            if threat is not None:
                rationale = (
                    f"learned risk {neural_risk:.0%} high; nearest agent: "
                    f"{threat.cls_name} {self._describe_position(threat)}"
                    + (f", TTC {threat.ttc_s:.1f}s" if threat.ttc_s != float("inf") else "")
                )
            else:
                rationale = f"learned risk {neural_risk:.0%} high; threat not localized by tracker"
            return Advisory(WARNING, "BRAKE", rationale, neural_risk, threat, others)

        # Rule 3: either signal elevated -> caution.
        if neural_risk >= self.risk_caution or (threat is not None and threat.ttc_s <= self.ttc_caution):
            if threat is not None and threat.ttc_s <= self.ttc_caution:
                rationale = (
                    f"{threat.cls_name} {self._describe_position(threat)} "
                    f"{threat.heading}, TTC {threat.ttc_s:.1f}s"
                )
            else:
                rationale = f"learned risk {neural_risk:.0%} elevated"
            return Advisory(CAUTION, "COVER BRAKE", rationale, neural_risk, threat, others)

        # Rule 4: all clear.
        return Advisory(NORMAL, "MONITOR", "no closing threats", neural_risk, threat, others)

    # -- helpers -------------------------------------------------------------

    def _primary_threat(self, facts: List[ObjectFacts]) -> Optional[ObjectFacts]:
        """Most urgent object: finite TTC first, then in-path, then apparent size.

        Objects being passed ("passing" = looming but projected to miss) never
        qualify; laterally moving objects qualify only while their drift
        converges on the ego line. In-path objects are always watched.
        """

        candidates = [
            f
            for f in facts
            if f.heading == "approaching"
            or (f.heading == "crossing" and f.converging)
            or f.in_path
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda f: (f.ttc_s, not f.in_path, -f.size_frac))

    def _evasive_command(self, threat: ObjectFacts, facts: List[ObjectFacts]) -> str:
        """Steer away from the threat if a side is clear, otherwise brake hard."""

        if threat.zone == "center":
            left_clear = self._side_clear(facts, "left", threat.track_id)
            right_clear = self._side_clear(facts, "right", threat.track_id)
            if threat.ttc_s < 1.0 and right_clear:
                return "BRAKE HARD + STEER RIGHT"
            if threat.ttc_s < 1.0 and left_clear:
                return "BRAKE HARD + STEER LEFT"
            return "BRAKE HARD"
        if threat.zone == "left":
            return "BRAKE + STEER RIGHT" if self._side_clear(facts, "right", threat.track_id) else "BRAKE HARD"
        return "BRAKE + STEER LEFT" if self._side_clear(facts, "left", threat.track_id) else "BRAKE HARD"

    @staticmethod
    def _side_clear(facts: List[ObjectFacts], side: str, exclude_id: int) -> bool:
        return not any(
            f.zone == side and f.track_id != exclude_id and (f.ttc_s < 4.0 or f.size_frac > 0.05)
            for f in facts
        )

    @staticmethod
    def _describe_position(fact: ObjectFacts) -> str:
        base = {"left": "on the left", "center": "ahead", "right": "on the right"}[fact.zone]
        return base + (" in path" if fact.in_path and fact.zone != "center" else "")
