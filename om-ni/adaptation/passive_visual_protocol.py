from __future__ import annotations

from dataclasses import dataclass
import random

BASELINE = 210
IMAGE_SHOW = 211
MOSAIC = 212
RECALL = 213
ITI = 214
SESSION_START = 201
SESSION_END = 202
TRIAL_START = 203
TRIAL_END = 204

EVENT_CODES = {
    "SESSION_START": SESSION_START,
    "SESSION_END": SESSION_END,
    "TRIAL_START": TRIAL_START,
    "TRIAL_END": TRIAL_END,
    "BASELINE": BASELINE,
    "IMAGE_SHOW": IMAGE_SHOW,
    "MOSAIC": MOSAIC,
    "RECALL": RECALL,
    "ITI": ITI,
}


@dataclass(frozen=True, slots=True)
class PassiveVisualTiming:
    baseline_sec: float = 1.0
    image_show_sec: float = 1.5
    mosaic_sec: float = 0.5
    recall_sec: float = 2.0
    iti_sec: float = 1.5


def build_passive_image_order(*, seed: int, n_images: int = 10) -> list[int]:
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    rng = random.Random(int(seed))
    order = list(range(int(n_images)))
    rng.shuffle(order)
    return order

