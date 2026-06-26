from __future__ import annotations

from dataclasses import dataclass
import random

BASELINE = 310
IMAGE_SELECTION = 311
MOSAIC = 312
RECALL = 313
ITI = 314
SESSION_START = 301
SESSION_END = 302
TRIAL_START = 303
TRIAL_END = 304
REST_START = 305
REST_END = 306
KEYPRESS = 307

EVENT_CODES = {
    "SESSION_START": SESSION_START,
    "SESSION_END": SESSION_END,
    "TRIAL_START": TRIAL_START,
    "TRIAL_END": TRIAL_END,
    "BASELINE": BASELINE,
    "IMAGE_SELECTION": IMAGE_SELECTION,
    "MOSAIC": MOSAIC,
    "RECALL": RECALL,
    "ITI": ITI,
    "REST_START": REST_START,
    "REST_END": REST_END,
    "KEYPRESS": KEYPRESS,
}


@dataclass(frozen=True, slots=True)
class ActiveVisualTiming:
    image_selection_sec: float = 0.0
    mosaic_sec: float = 0.5
    recall_sec: float = 2.0
    iti_sec: float = 0.5
    rest_between_hours_sec: float = 300.0


def build_active_image_order(*, seed: int, n_images: int = 20) -> list[int]:
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    rng = random.Random(int(seed))
    order = list(range(int(n_images)))
    rng.shuffle(order)
    return order


def build_full_active_trial_order(*, seed: int, n_images: int = 20, trials_per_image: int = 1000) -> list[int]:
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    if trials_per_image <= 0:
        raise ValueError(f"trials_per_image must be positive, got {trials_per_image}")
    
    rng = random.Random(int(seed))
    order = []
    for _ in range(trials_per_image):
        order.extend(range(n_images))
    
    for _ in range(500):
        rng.shuffle(order)
    
    return order
