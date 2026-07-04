from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from adaptation.active_visual_protocol import EVENT_CODES as ACTIVE_EVENT_CODES
from adaptation.active_visual_protocol import ActiveVisualTiming, build_full_active_trial_order
from adaptation.calibrator import CalibrationResult
from adaptation.session_recorder import SessionRecorder
from tasks.mouse_auto_click import click_screen_position
from tasks.visual_stimuli import REST_CLASS_ID, load_visual_stimuli
from tasks.visual_window import SelectionItem, VisualStimulusWindow
from utils.preprocessing import filter_and_transform

_VISUAL_REST_CLASS_ID = REST_CLASS_ID


@dataclass(slots=True)
class VisualSegment:
    label_id: int
    start_sample: int
    end_sample: int
    name: str


def _auto_click_selection_if_enabled(config: dict[str, Any], window: VisualStimulusWindow) -> None:
    if not bool(config.get("auto_click", False)):
        return
    x, y = window.wait_for_selection_target()
    click_screen_position(x, y, duration_sec=0.3)


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
    subject_id: str = "001",
    session_id: str = "01",
    task_name: str = "active",
) -> CalibrationResult:
    timing = ActiveVisualTiming()
    seed = int(config.get("protocol", {}).get("random_seed", 17))
    stimuli = load_visual_stimuli(config, fallback_image_path=stimulus_image_path)
    n_images = len(stimuli)
    label_names = {idx: stimulus.display_name for idx, stimulus in stimuli.items()} | {_VISUAL_REST_CLASS_ID: "静息"}

    window = VisualStimulusWindow(title="OI-MI 主动想象实验", default_image_path=stimulus_image_path)
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

        def pause_for_rest(hour_index: int) -> None:
            console.print(f"[bold yellow]第 {hour_index} 小时结束，休息 5 分钟[/bold yellow]")
            emit_event("rest_start", int(ACTIVE_EVENT_CODES["REST_START"]), hour=hour_index)
            window.show_black("休息时间", "5分钟")
            sleep_with_recording(timing.rest_between_hours_sec)
            window.show_black("休息结束", "即将开始下一阶段")
            emit_event("rest_end", int(ACTIVE_EVENT_CODES["REST_END"]), hour=hour_index)
            time.sleep(2.0)

        emit_event("session_start", int(ACTIVE_EVENT_CODES["SESSION_START"]), exp="1.2", task="active")

        emit_event("experiment_start", int(ACTIVE_EVENT_CODES["EXPERIMENT_START"]), exp="1.2", task="active")

        trials_per_image = 50
        total_trials = n_images * trials_per_image
        console.print(f"[bold cyan]生成 {total_trials} 个 trial，共 {trials_per_image} 组，每组 {n_images} 张图片[/bold cyan]")

        rng = random.Random(int(seed))

        current_trial = 0
        global_trial = 0
        for group_index in range(trials_per_image):
            console.print(f"[bold cyan]第 {group_index + 1}/{trials_per_image} 组开始[/bold cyan]")
            
            available_images = list(range(n_images))
            rng.shuffle(available_images)
            
            while available_images:
                selected_items = [
                    SelectionItem(
                        item_id=int(img_id),
                        title=f"图片{int(img_id) + 1}",
                        image_path=stimuli[int(img_id)].image_path,
                    )
                    for img_id in available_images
                ]
                window.show_selection("请选择一张图片进行想象", f"第{group_index + 1}组，剩余{len(available_images)}张", selected_items)
                console.print(f"[bold yellow][图片选择][/bold yellow] 剩余 {len(available_images)} 张图片可选")
                flush()
                emit_event("image_selection", int(ACTIVE_EVENT_CODES["IMAGE_SELECTION"]), exp="1.2", trial=global_trial, group=group_index)
                _auto_click_selection_if_enabled(config, window)

                chosen: int | None = None
                while chosen is None:
                    flush()
                    refresh()
                    chosen = window.poll_selection()
                    time.sleep(0.02)

                available_images.remove(int(chosen))
                stimulus = stimuli[int(chosen)]

                emit_event(
                    "trial_start",
                    int(ACTIVE_EVENT_CODES["TRIAL_START"]),
                    exp="1.2",
                    trial=global_trial,
                    group=group_index,
                    image_id=int(chosen),
                )
                emit_event("keypress", int(ACTIVE_EVENT_CODES["KEYPRESS"]), exp="1.2", trial=global_trial, chosen=int(chosen))
                console.print(f"[bold yellow][已选择][/bold yellow] 图片{int(chosen) + 1}: {stimulus.text}")

                window.show_cross_mask()
                console.print("[bold yellow][视觉残留消除][/bold yellow] 白色十字 0.5s")
                flush()
                emit_event("mosaic", int(ACTIVE_EVENT_CODES["MOSAIC"]), exp="1.2", trial=global_trial, chosen=int(chosen))
                sleep_with_recording(timing.mosaic_sec)

                window.show_black("回忆图片", "2s")
                console.print("[bold cyan][黑屏回忆][/bold cyan] 2s")
                flush()
                recall_start = int(recorder.sample_count)
                emit_event("recall", int(ACTIVE_EVENT_CODES["RECALL"]), exp="1.2", trial=global_trial, chosen=int(chosen))
                emit_label(int(chosen), exp="1.2", trial=global_trial, chosen=int(chosen))
                sleep_with_recording(timing.recall_sec)
                flush()
                recall_end = int(recorder.sample_count)
                segments.append(VisualSegment(int(chosen), recall_start, recall_end, "exp_1_2_recall"))

                window.show_black("休息", "0.5s")
                flush()
                iti_start = int(recorder.sample_count)
                emit_event("iti", int(ACTIVE_EVENT_CODES["ITI"]), exp="1.2", trial=global_trial)
                emit_label(_VISUAL_REST_CLASS_ID, exp="1.2", trial=global_trial)
                sleep_with_recording(timing.iti_sec)
                flush()
                iti_end = int(recorder.sample_count)
                segments.append(VisualSegment(_VISUAL_REST_CLASS_ID, iti_start, iti_end, "exp_1_2_iti"))

                emit_event("trial_end", int(ACTIVE_EVENT_CODES["TRIAL_END"]), exp="1.2", trial=global_trial, chosen=int(chosen))
                trials.append(
                    {
                        "exp": "1.2",
                        "group": int(group_index),
                        "trial_index": int(global_trial),
                        "image_id": int(chosen),
                        "label_id": int(chosen),
                        "stimulus_label": stimulus.label,
                        "stimulus_text": stimulus.text,
                        "stimulus_image": str(stimulus.image_path),
                    }
                )

                current_trial += 1
                global_trial += 1
                window.set_progress(current_trial, total_trials)
                if current_trial % 100 == 0:
                    console.print(f"[bold cyan]进度[/bold cyan] {current_trial}/{total_trials} ({current_trial/total_trials*100:.1f}%)")
            
            console.print(f"[bold cyan]第 {group_index + 1}/{trials_per_image} 组完成[/bold cyan]")

        emit_event("experiment_end", int(ACTIVE_EVENT_CODES["EXPERIMENT_END"]), exp="1.2")
        emit_event("session_end", int(ACTIVE_EVENT_CODES["SESSION_END"]), exp="1.2")
        acquirer.stop_stream()

        metadata = {
            "task_mode": "visual",
            "experiment": "exp_1_2_active",
            "sfreq": float(config["sfreq"]),
            "window_sec": float(config["window_sec"]),
            "step_sec": float(config["step_sec"]),
            "label_names": {str(k): v for k, v in label_names.items()},
            "stimuli": [
                {
                    "id": int(stimulus.item_id),
                    "label": stimulus.label,
                    "text": stimulus.text,
                    "image": str(stimulus.image_path),
                }
                for stimulus in stimuli.values()
            ],
            "event_codes": dict(ACTIVE_EVENT_CODES),
            "trials": trials,
            "segments": [
                {
                    "name": seg.name,
                    "label_id": int(seg.label_id),
                    "label_name": label_names.get(int(seg.label_id), str(seg.label_id)),
                    "start_sample": int(seg.start_sample),
                    "end_sample": int(seg.end_sample),
                }
                for seg in segments
            ],
        }
        bids_base_dir = session_dir.parent.parent.parent if len(session_dir.parts) >= 3 else session_dir
        bids_dir = recorder.export_bids(
            bids_base_dir,
            subject_id=subject_id,
            session_id=session_id,
            task_name=task_name,
            metadata=metadata,
        )
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
