"""Neurosymbolic reasoning layer for crash anticipation.

The neural model answers *"how dangerous is this moment?"* — this package
answers *"what exactly is the threat and what should the driver do?"*:

    perception.py  detect + track road agents (YOLO + ByteTrack)
    dynamics.py    tracks -> symbolic facts (TTC, bearing, closing speed)
    rules.py       facts x neural risk -> actionable advisory with rationale
"""

from .dynamics import TrackDynamics, ObjectFacts
from .rules import Advisory, AdvisoryEngine

__all__ = ["TrackDynamics", "ObjectFacts", "Advisory", "AdvisoryEngine"]
