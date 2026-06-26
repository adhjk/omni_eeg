from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_COUNT = 20
REST_CLASS_ID = 20
REST_LABEL = "静息"


@dataclass(frozen=True, slots=True)
class VisualStimulus:
    item_id: int
    label: str
    text: str
    image_path: Path

    @property
    def display_name(self) -> str:
        return self.text or self.label or f"图片{self.item_id + 1}"


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    candidates = (
        PROJECT_ROOT / path,
        Path.cwd() / path,
        Path.cwd() / "oi-mi" / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def default_visual_stimuli_config(default_image: str = "assets/EEG.png") -> list[dict[str, Any]]:
    return [
        {
            "id": idx,
            "label": f"image_{idx + 1}",
            "text": f"图片{idx + 1}",
            "image": default_image,
        }
        for idx in range(DEFAULT_IMAGE_COUNT)
    ]


def _normalise_stimulus_item(
    item: dict[str, Any],
    *,
    fallback_image_path: Path,
) -> VisualStimulus:
    try:
        item_id = int(item["id"])
    except KeyError as exc:
        raise RuntimeError(f"visual_stimuli item is missing required key 'id': {item!r}") from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"visual_stimuli item has invalid id: {item!r}") from exc

    label = str(item.get("label") or f"image_{item_id + 1}")
    text = str(item.get("text") or item.get("name") or label or f"图片{item_id + 1}")
    image_value = item.get("image") or item.get("image_path")
    image_path = fallback_image_path if image_value is None else resolve_project_path(str(image_value))
    if not image_path.exists():
        raise RuntimeError(
            f"Stimulus image not found for id={item_id}, text={text!r}: {image_path}. "
            "Please update config.visual_stimuli[*].image."
        )
    return VisualStimulus(item_id=item_id, label=label, text=text, image_path=image_path)


def load_visual_stimuli(
    config: dict[str, Any],
    *,
    fallback_image_path: Path,
    expected_count: int = DEFAULT_IMAGE_COUNT,
) -> dict[int, VisualStimulus]:
    raw_items = config.get("visual_stimuli") or default_visual_stimuli_config(str(fallback_image_path))
    if not isinstance(raw_items, list):
        raise RuntimeError("config.visual_stimuli must be a list of stimulus objects.")

    stimuli: dict[int, VisualStimulus] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise RuntimeError(f"config.visual_stimuli entries must be mappings, got {raw_item!r}")
        stimulus = _normalise_stimulus_item(raw_item, fallback_image_path=fallback_image_path)
        if stimulus.item_id in stimuli:
            raise RuntimeError(f"Duplicate visual stimulus id: {stimulus.item_id}")
        stimuli[stimulus.item_id] = stimulus

    expected_ids = set(range(int(expected_count)))
    actual_ids = set(stimuli)
    if actual_ids != expected_ids:
        raise RuntimeError(
            "config.visual_stimuli must define exactly ids "
            f"{sorted(expected_ids)}, got {sorted(actual_ids)}."
        )
    return dict(sorted(stimuli.items()))


def visual_label_names(config: dict[str, Any], *, fallback_image_path: Path | None = None) -> dict[int, str]:
    fallback = fallback_image_path or resolve_project_path("assets/EEG.png")
    try:
        stimuli = load_visual_stimuli(config, fallback_image_path=fallback)
        labels = {idx: stimulus.display_name for idx, stimulus in stimuli.items()}
    except RuntimeError:
        labels = {idx: f"图片{idx + 1}" for idx in range(DEFAULT_IMAGE_COUNT)}
    labels[REST_CLASS_ID] = REST_LABEL
    return labels


def visual_test_mode_prompts(config: dict[str, Any], *, fallback_image_path: Path | None = None) -> dict[int, str]:
    fallback = fallback_image_path or resolve_project_path("assets/EEG.png")
    try:
        stimuli = load_visual_stimuli(config, fallback_image_path=fallback)
        n_images = len(stimuli)
    except RuntimeError:
        n_images = DEFAULT_IMAGE_COUNT
    labels = visual_label_names(config, fallback_image_path=fallback)
    prompts = {idx: f"想象图片 {idx + 1}: {labels[idx]}" for idx in range(n_images)}
    prompts[REST_CLASS_ID] = "保持静息"
    return prompts



