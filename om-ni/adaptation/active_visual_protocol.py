from __future__ import annotations

from dataclasses import dataclass

BASELINE = 310
ACTIVE_IMAGINATION = 311
IMAGE_SELECTION = 312
ITI = 313
SESSION_START = 301
SESSION_END = 302
TRIAL_START = 303
TRIAL_END = 304

EVENT_CODES = {
    "SESSION_START": SESSION_START,
    "SESSION_END": SESSION_END,
    "TRIAL_START": TRIAL_START,
    "TRIAL_END": TRIAL_END,
    "BASELINE": BASELINE,
    "ACTIVE_IMAGINATION": ACTIVE_IMAGINATION,
    "IMAGE_SELECTION": IMAGE_SELECTION,
    "ITI": ITI,
}


@dataclass(frozen=True, slots=True)
class ActiveVisualTiming:
    baseline_sec: float = 1.0
    active_imagination_sec: float = 2.0
    iti_sec: float = 1.5

