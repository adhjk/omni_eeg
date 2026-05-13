from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from adaptation.active_visual_protocol import EVENT_CODES as ACTIVE_EVENT_CODES
from adaptation.active_visual_protocol import ActiveVisualTiming
from adaptation.calibrator import CalibrationResult
from adaptation.session_recorder import SessionRecorder
from tasks.visual_window import SelectionItem, VisualStimulusWindow
from utils.preprocessing import filter_and_transform

_VISUAL_REST_CLASS_ID = 10
_VISUAL_LABEL_NAMES = {idx: f"图片{idx + 1}" for idx in range(10)} | {_VISUAL_REST_CLASS_ID: "静息"}


@dataclass(slots=True)
class VisualSegment:
    label_id: int
    start_sample: int
    end_sample: int
    name: str


def _build_windows(*, eeg: np.ndarray, segments: list[VisualSegment], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sfreq = float(config["sfreq"])
    window_samples = int(round(float(config["window_sec"]) * sfreq))
    stride_samples = int(round(float(config["step_sec"]) * sfreq))
    start_offset_sec = float(config.get("protocol", {}).get("control_start_offset_sec", 0.2))
    stop_offset_sec = float(config.get("protocol", {}).get("control_stop_offset_sec", 1.8))
    start_offset = int(round(start_offset_sec * sfreq))
    stop_offset = int(round(stop_offset_sec * sfreq))

    raw_windows: list[np.ndarray] = []
    processed_windows: list[np.ndarray] = []
    labels: list[int] = []
    for seg in segments:
        seg_len = int(seg.end_sample - seg.start_sample)
        if seg_len <= window_samples:
            continue
        max_stop = min(seg_len, stop_offset)
        max_start = seg.start_sample + max_stop - window_samples
        for offset in range(start_offset, max_start - seg.start_sample + 1, stride_samples):
            start = int(seg.start_sample + offset)
            stop = int(start + window_samples)
            if stop > eeg.shape[1]:
                continue
            w = eeg[:, start:stop].astype(np.float32)
            raw_windows.append(w)
            processed_windows.append(filter_and_transform(w, sfreq=sfreq))
            labels.append(int(seg.label_id))

    raw_X = (
        np.stack(raw_windows, axis=0).astype(np.float32)
        if raw_windows
        else np.empty((0, eeg.shape[0], window_samples), dtype=np.float32)
    )
    X = (
        np.stack(processed_windows, axis=0).astype(np.float32)
        if processed_windows
        else np.empty((0, eeg.shape[0], window_samples), dtype=np.float32)
    )
    y = np.asarray(labels, dtype=np.int64)
    return raw_X, X, y


def run(
    *,
    config: dict[str, Any],
    acquirer: Any,
    model: Any,
    marker_backend: Any,
    console: Any,
    refresh: Any,
    model_path: Path,
    session_dir: Path,
    stimulus_image_path: Path,
) -> CalibrationResult:
    timing = ActiveVisualTiming()
    blocks = int(config.get("protocol", {}).get("visual_blocks", 1))

    window = VisualStimulusWindow(title="OI-MI", default_image_path=stimulus_image_path)
    window.start()

    recorder: SessionRecorder | None = None
    segments: list[VisualSegment] = []
    trials: list[dict[str, Any]] = []
    try:
        acquirer.start_stream()
        recorder = SessionRecorder(acquirer, sfreq=float(config["sfreq"]), n_channels=int(acquirer.metadata.n_channels))

        def emit_event(name: str, code: int, **payload: Any) -> None:
            marker_backend.send(int(code))
            recorder.add_event(name, marker_code=int(code), **payload)

        def emit_label(label_id: int, **payload: Any) -> None:
            marker_backend.send(int(label_id))
            recorder.add_event("label", marker_code=int(label_id), label_id=int(label_id), **payload)

        def flush() -> None:
            recorder.pull()

        def sleep_with_recording(duration_sec: float) -> None:
            deadline = time.monotonic() + max(float(duration_sec), 0.0)
            while time.monotonic() < deadline:
                flush()
                refresh()
                time.sleep(min(0.02, max(deadline - time.monotonic(), 0.0)))
            flush()
            refresh()

        def pause_between_blocks(block_index: int) -> None:
            evt = window.begin_pause(f"Block {block_index} 结束，按空格继续下一组")
            console.print(f"[bold yellow]Block {block_index} 结束，等待空格继续[/bold yellow]")
            while not evt.is_set():
                flush()
                refresh()
                time.sleep(0.05)

        emit_event("session_start", int(ACTIVE_EVENT_CODES["SESSION_START"]), exp="1.2")
        for block_index in range(blocks):
            if block_index > 0:
                pause_between_blocks(block_index)

            remaining = list(range(10))
            trial_index = 0
            while remaining:
                global_trial = block_index * 10 + int(trial_index)
                emit_event("trial_start", int(ACTIVE_EVENT_CODES["TRIAL_START"]), exp="1.2", block=block_index, trial=global_trial, remaining=list(remaining))

                window.show_black("Baseline", "1s")
                flush()
                baseline_start = int(recorder.sample_count)
                emit_event("baseline", int(ACTIVE_EVENT_CODES["BASELINE"]), exp="1.2", block=block_index, trial=global_trial)
                emit_label(_VISUAL_REST_CLASS_ID, exp="1.2", block=block_index, trial=global_trial)
                sleep_with_recording(timing.baseline_sec)
                flush()
                baseline_end = int(recorder.sample_count)
                segments.append(VisualSegment(_VISUAL_REST_CLASS_ID, baseline_start, baseline_end, "exp_1_2_baseline"))

                window.show_black("黑屏主动想象", "2s")
                console.print("[bold cyan]黑屏主动想象[/bold cyan] 2s")
                flush()
                imagine_start = int(recorder.sample_count)
                emit_event("active_imagination", int(ACTIVE_EVENT_CODES["ACTIVE_IMAGINATION"]), exp="1.2", block=block_index, trial=global_trial)
                sleep_with_recording(timing.active_imagination_sec)
                flush()
                imagine_end = int(recorder.sample_count)

                items = [
                    SelectionItem(item_id=int(img_id), title=f"图片{int(img_id) + 1}", image_path=stimulus_image_path)
                    for img_id in remaining
                ]
                window.show_selection("请选择你刚才想象的图片", "点击缩略图按钮", items)
                emit_event("image_selection", int(ACTIVE_EVENT_CODES["IMAGE_SELECTION"]), exp="1.2", block=block_index, trial=global_trial, remaining=list(remaining))

                chosen: int | None = None
                while chosen is None:
                    flush()
                    refresh()
                    chosen = window.poll_selection()
                    time.sleep(0.02)

                emit_label(int(chosen), exp="1.2", block=block_index, trial=global_trial, chosen=int(chosen))
                segments.append(VisualSegment(int(chosen), imagine_start, imagine_end, "exp_1_2_imagine"))
                trials.append(
                    {
                        "exp": "1.2",
                        "block": int(block_index),
                        "trial_index": int(global_trial),
                        "label_id": int(chosen),
                        "imagine_start_sample": int(imagine_start),
                        "imagine_end_sample": int(imagine_end),
                    }
                )
                remaining.remove(int(chosen))

                window.show_black("ITI", "1.5s")
                emit_event("iti", int(ACTIVE_EVENT_CODES["ITI"]), exp="1.2", block=block_index, trial=global_trial)
                emit_label(_VISUAL_REST_CLASS_ID, exp="1.2", block=block_index, trial=global_trial)
                sleep_with_recording(timing.iti_sec)

                emit_event("trial_end", int(ACTIVE_EVENT_CODES["TRIAL_END"]), exp="1.2", block=block_index, trial=global_trial)
                trial_index += 1

        emit_event("session_end", int(ACTIVE_EVENT_CODES["SESSION_END"]), exp="1.2")
        acquirer.stop_stream()

        metadata = {
            "task_mode": "visual",
            "experiment": "exp_1_2_active",
            "sfreq": float(config["sfreq"]),
            "window_sec": float(config["window_sec"]),
            "step_sec": float(config["step_sec"]),
            "label_names": {str(k): v for k, v in _VISUAL_LABEL_NAMES.items()},
            "event_codes": dict(ACTIVE_EVENT_CODES),
            "trials": trials,
            "segments": [
                {
                    "name": seg.name,
                    "label_id": int(seg.label_id),
                    "label_name": _VISUAL_LABEL_NAMES.get(int(seg.label_id), str(seg.label_id)),
                    "start_sample": int(seg.start_sample),
                    "end_sample": int(seg.end_sample),
                }
                for seg in segments
            ],
        }
        recorder.export(session_dir, metadata=metadata)
        eeg = recorder.to_array()
        raw_X, X, y = _build_windows(eeg=eeg, segments=segments, config=config)
        if X.shape[0] == 0:
            raise RuntimeError("校准未生成任何可用训练窗（请检查采集与 timing 配置）。")

        metrics = model.fit(
            X,
            y,
            epochs=int(config["new_subject_epochs"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            patience=int(config["early_stopping_patience"]),
            head_only=False,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

        np.savez_compressed(
            session_dir / "training_windows_main.npz",
            raw_windows=raw_X,
            processed_windows=X,
            labels=y,
            sfreq=np.asarray([float(config["sfreq"])], dtype=np.float32),
            window_sec=np.asarray([float(config["window_sec"])], dtype=np.float32),
            step_sec=np.asarray([float(config["step_sec"])], dtype=np.float32),
        )

        metrics_path = model_path.with_suffix(".metrics.yaml")
        with metrics_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "task_mode": "visual",
                    "experiment": "exp_1_2_active",
                    "model_path": str(model_path),
                    "session_dir": str(session_dir),
                    "windows_collected": int(X.shape[0]),
                    "metrics": dict(metrics),
                },
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        return CalibrationResult(
            model_path=model_path,
            metrics=metrics,
            windows_collected=int(X.shape[0]),
            calibration_data_path=session_dir / "training_windows_main.npz",
            session_dir=session_dir,
        )
    finally:
        try:
            acquirer.stop_stream()
        except Exception:
            pass
        try:
            window.close()
        except Exception:
            pass
