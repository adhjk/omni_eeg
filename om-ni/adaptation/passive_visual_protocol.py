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
REST_START = 205
REST_END = 206
EXPERIMENT_START = 215
EXPERIMENT_END = 216

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
    "REST_START": REST_START,
    "REST_END": REST_END,
    "EXPERIMENT_START": EXPERIMENT_START,
    "EXPERIMENT_END": EXPERIMENT_END,
}


@dataclass(frozen=True, slots=True)
class PassiveVisualTiming:
    baseline_sec: float = 0.0
    image_show_sec: float = 1.0
    mosaic_sec: float = 0.5
    recall_sec: float = 2.0
    iti_sec: float = 0.5
    rest_between_hours_sec: float = 300.0


def build_passive_image_order(*, seed: int, n_images: int = 20) -> list[int]:
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    rng = random.Random(int(seed))
    order = list(range(int(n_images)))
    rng.shuffle(order)
    return order


def build_full_trial_order(*, seed: int, n_images: int = 20, trials_per_image: int = 50) -> list[int]:
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
