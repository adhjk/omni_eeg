from __future__ import annotations

import re
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import random

import numpy as np

import streamlit as st
import yaml
from tasks.task_factory import load_task_from_config
from acquisition.factory import AcquirerFactory, register_default_acquirers
from adaptation.active_visual_protocol import ActiveVisualTiming
from adaptation.active_visual_protocol import EVENT_CODES as ACTIVE_EVENT_CODES
from adaptation.passive_visual_protocol import EVENT_CODES as PASSIVE_EVENT_CODES
from adaptation.passive_visual_protocol import PassiveVisualTiming, build_passive_image_order
from adaptation.calibrator import CalibrationResult
from adaptation.session_recorder import SessionRecorder
from cli import build_acquirer, build_game_command_outlet, build_marker_backend, build_model_path, write_config
from decoder.real_time_decoder import PredictionResult
from models.factory import ModelFactory
from utils.markers import LSLCommandOutlet, MarkerBackend
from utils.preprocessing import filter_and_transform
from utils.stream_writer import StreamWriter
from tasks.text_exp_1_1 import run as run_text_exp_1_1
from tasks.text_exp_1_2 import run as run_text_exp_1_2
from tasks.visual_exp_1_1 import run as run_visual_exp_1_1
from tasks.visual_exp_1_2 import run as run_visual_exp_1_2
from tasks.visual_stimuli import REST_CLASS_ID, visual_label_names, visual_test_mode_prompts

# ---------- 资产路径（基于本文件所在 tasks/ 目录的父目录） ----------
_MOTOR_ROOT = Path(__file__).resolve().parent.parent
_LOGO_FILENAME = "OMNI_LOGO_ENG_double_line.svg"

def _resolve_asset_path(filename: str) -> Path | None:
    candidates = (
        _MOTOR_ROOT / "assets" / filename,
        Path.cwd() / "assets" / filename,
        Path.cwd() / "oi-mi" / "assets" / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def _resolve_logo_svg_path() -> Path | None:
    return _resolve_asset_path(_LOGO_FILENAME)

# ---------- 全局配置路径（由 render_motor_gui 注入） ----------
_current_config_path: Path | None = None

def _save_config(cfg: dict) -> None:
    """保存配置到当前 config_path，供页面调用。"""
    if _current_config_path is None:
        st.error("无法保存：未获取到配置文件路径。")
        return
    try:
        write_config(_current_config_path, cfg)
    except Exception as exc:
        st.error(f"保存配置文件失败: {exc}")

_VISUAL_REST_CLASS_ID = REST_CLASS_ID
_VISUAL_LABEL_NAMES = {idx: f"图片{idx + 1}" for idx in range(10)} | {_VISUAL_REST_CLASS_ID: "静息"}
_VISUAL_COMMANDS = {idx: f"IMG_{idx + 1}" for idx in range(10)} | {_VISUAL_REST_CLASS_ID: "REST"}
_VISUAL_TEST_MODE_PROMPTS = {idx: f"想象图片 {idx + 1}" for idx in range(10)} | {_VISUAL_REST_CLASS_ID: "保持静息"}

# ---------- 提示符号与事件解析 ----------
PROMPT_TEXTS = tuple(_VISUAL_TEST_MODE_PROMPTS.values())

_DISPLAY_SYMBOLS = {
    "REST": "○",
    "START": "◎",
    "PAUSE": "◌",
    "TRANSITION": "·",
    "DONE": "✓",
    "ERROR": "✕",
}

def _resolve_cue_symbol(message: str, *, event_type: str) -> tuple[str, bool] | None:
    if "静息" in message or "REST" in message.upper() or "IDLE" in message.upper():
        return _DISPLAY_SYMBOLS["REST"], event_type == "prediction"
    if "图片" in message:
        digits = "".join(ch for ch in message if ch.isdigit())
        return (digits if digits else _DISPLAY_SYMBOLS["TRANSITION"]), event_type == "prediction"
    if "校准完成" in message:
        return _DISPLAY_SYMBOLS["DONE"], False
    if "执行失败" in message:
        return _DISPLAY_SYMBOLS["ERROR"], False
    if "开始" in message and "采集" in message:
        return _DISPLAY_SYMBOLS["START"], False
    if "Baseline" in message or "基线" in message or "休息" in message:
        return _DISPLAY_SYMBOLS["PAUSE"], False
    if "Block " in message or "练习阶段" in message or "实验指导语" in message:
        return _DISPLAY_SYMBOLS["TRANSITION"], False
    return None

SIDEBAR_NAV_PAGES = ("首页", "设置", "连通检测", "校准", "测试模式", "实时解码")

# ---------- StreamlitConsole 与 live view ----------
class StreamlitConsole:
    """Minimal Rich Console substitute that writes into Streamlit placeholders."""

    def __init__(self, cue_placeholder, log_placeholder) -> None:
        self.cue_placeholder = cue_placeholder
        self.log_placeholder = log_placeholder
        self.logs: list[str] = []
        self._lock = threading.Lock()
        self._pending_events: list[tuple[str, str]] = []
        self._ui_thread_id = threading.get_ident()

    def print(self, message, *args, **kwargs) -> None:
        raw_message = str(message)
        msg = re.sub(r"\[.*?\]", "", raw_message).strip()
        if not msg:
            return

        event_type = "log"
        if any(prompt in msg for prompt in PROMPT_TEXTS):
            event_type = "cue"
        elif "confidence:" in msg:
            event_type = "prediction"
        elif _resolve_cue_symbol(msg, event_type="log") is not None:
            event_type = "cue"

        with self._lock:
            self._pending_events.append((event_type, msg))

        if threading.get_ident() == self._ui_thread_id:
            self.render_pending()

    def render_pending(self) -> None:
        with self._lock:
            if not self._pending_events:
                return
            pending = list(self._pending_events)
            self._pending_events.clear()

        log_updated = False
        for event_type, msg in pending:
            if event_type in {"cue", "prediction"}:
                self._render_cue(msg, prediction=(event_type == "prediction"))
                self._append_log(msg)
                log_updated = True
            else:
                self._append_log(msg)
                log_updated = True

        if log_updated:
            self.log_placeholder.code("\n".join(self.logs))

    def _append_log(self, msg: str) -> None:
        self.logs.append(msg)
        if len(self.logs) > 18:
            self.logs.pop(0)

    def _render_cue(self, msg: str, *, prediction: bool) -> None:
        resolved = _resolve_cue_symbol(msg, event_type="prediction" if prediction else "cue")
        symbol = resolved[0] if resolved is not None else "·"
        is_prediction = resolved[1] if resolved is not None else prediction
        bg = "#F0FFF4" if is_prediction else "#F8FAFC"
        color = "#0F766E" if is_prediction else "#C2410C"
        self.cue_placeholder.markdown(
            (
                "<div style='padding: 1.25rem; min-height: 8rem; border-radius: 12px; "
                "display: flex; align-items: center; justify-content: center; "
                f"background-color: {bg}; border: 1px solid #E2E8F0;'>"
                f"<div style='font-size: 4.5rem; line-height: 1; font-weight: 700; color: {color};'>{symbol}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

def init_live_view() -> tuple[StreamlitConsole, callable]:
    cue_box = st.empty()
    log_box = st.empty()
    console = StreamlitConsole(cue_box, log_box)

    def refresh() -> None:
        console.render_pending()
        return

    refresh()
    return console, refresh

@dataclass(slots=True)
class _VisualSegment:
    label_id: int
    start_sample: int
    end_sample: int
    name: str


class VisualCalibrator:
    def __init__(
        self,
        acquirer,
        model,
        marker_backend: MarkerBackend,
        console,
        *,
        sfreq: float,
        window_sec: float,
        step_sec: float,
        model_path: Path,
        calibration_records_dir: Path | None = None,
        protocol_config: Any | None = None,
    ) -> None:
        seed = 17
        start_offset_sec = 0.2
        stop_offset_sec = 1.8
        if protocol_config is not None:
            if isinstance(protocol_config, dict):
                seed = int(protocol_config.get("random_seed", seed))
                start_offset_sec = float(protocol_config.get("control_start_offset_sec", start_offset_sec))
                stop_offset_sec = float(protocol_config.get("control_stop_offset_sec", stop_offset_sec))
            else:
                seed = int(getattr(protocol_config, "random_seed", seed))
                start_offset_sec = float(getattr(protocol_config, "control_start_offset_sec", start_offset_sec))
                stop_offset_sec = float(getattr(protocol_config, "control_stop_offset_sec", stop_offset_sec))
        self._acquirer = acquirer
        self._model = model
        self._marker_backend = marker_backend
        self._console = console
        self._sfreq = float(sfreq)
        self._window_sec = float(window_sec)
        self._step_sec = float(step_sec)
        self._seed = seed
        self._start_offset_sec = start_offset_sec
        self._stop_offset_sec = stop_offset_sec
        self._model_path = model_path
        self._calibration_records_dir = calibration_records_dir

    def load_existing_weights(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self._model_path}")
        self._model.load(self._model_path)

    def calibrate(
        self,
        *,
        duration_sec: int | None,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        head_only: bool,
        heartbeat: Callable[[], None] | None = None,
    ) -> CalibrationResult:
        del duration_sec
        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self._calibration_records_dir / session_stamp if self._calibration_records_dir is not None else None

        self._console.print("[bold cyan]开始视觉想象校准采集[/bold cyan]")
        self._console.print("[bold cyan]实验1.1 被动想象图片[/bold cyan]（10 trial）")
        self._console.print("[bold cyan]实验1.2 主动想象图片[/bold cyan]（10 trial，需 GUI 选择）")
        self._console.print(
            f"[bold cyan]训练切窗[/bold cyan] window={self._window_sec:.2f}s step={self._step_sec:.2f}s "
            "（起止偏移使用 config.protocol.control_start/stop_offset_sec）"
        )

        self._acquirer.start_stream()
        recorder = SessionRecorder(self._acquirer, sfreq=self._sfreq, n_channels=self._acquirer.metadata.n_channels)
        segments: list[_VisualSegment] = []
        trials: list[dict[str, Any]] = []
        try:
            recorder.add_event("session_start", task_mode="visual")
            self._run_experiment_passive(recorder=recorder, heartbeat=heartbeat, segments=segments, trials=trials)
            recorder.add_event("session_end", task_mode="visual")
        finally:
            self._flush_recorder(recorder)
            self._acquirer.stop_stream()
            if heartbeat is not None:
                heartbeat()

        metadata = {
            "task_mode": "visual",
            "sfreq": self._sfreq,
            "window_sec": self._window_sec,
            "step_sec": self._step_sec,
            "labels": {str(k): v for k, v in _VISUAL_LABEL_NAMES.items()},
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
        if session_dir is not None:
            recorder.export(session_dir, metadata=metadata)

        eeg = recorder.to_array()
        raw_X, X, y = self._build_windows(eeg=eeg, segments=segments)
        if X.shape[0] == 0:
            raise RuntimeError("Calibration did not yield any valid training windows.")
        metrics = self._model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            head_only=head_only,
        )
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(self._model_path)
        self._save_metrics(metrics=metrics, windows_collected=int(X.shape[0]), head_only=head_only)
        if session_dir is not None:
            self._save_training_windows(session_dir / "training_windows_main.npz", raw_X=raw_X, X=X, y=y)
        return CalibrationResult(
            model_path=self._model_path,
            metrics=metrics,
            windows_collected=int(X.shape[0]),
            calibration_data_path=(session_dir / "training_windows_main.npz") if session_dir is not None else None,
            session_dir=session_dir,
        )

    def _run_experiment_passive(
        self,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        segments: list[_VisualSegment],
        trials: list[dict[str, Any]],
    ) -> None:
        image_ids = build_passive_image_order(seed=self._seed, n_images=10)
        recorder.add_event("exp_1_1_start")
        for idx, image_id in enumerate(image_ids, start=1):
            trial_start = recorder.sample_count

            baseline_start = recorder.sample_count
            recorder.add_event("baseline_start", exp="1.1", trial=idx, label_id=_VISUAL_REST_CLASS_ID)
            self._marker_backend.send(_VISUAL_REST_CLASS_ID)
            self._sleep_with_recording(1.0, recorder=recorder, heartbeat=heartbeat)
            baseline_end = recorder.sample_count
            segments.append(_VisualSegment(_VISUAL_REST_CLASS_ID, baseline_start, baseline_end, "exp_1_1_baseline"))

            recorder.add_event("image_show", exp="1.1", trial=idx, image_id=image_id)
            self._sleep_with_recording(1.5, recorder=recorder, heartbeat=heartbeat)

            recorder.add_event("mosaic_show", exp="1.1", trial=idx, image_id=image_id)
            self._sleep_with_recording(0.5, recorder=recorder, heartbeat=heartbeat)

            imagine_start = recorder.sample_count
            recorder.add_event("imagine_start", exp="1.1", trial=idx, image_id=image_id, label_id=image_id)
            self._marker_backend.send(int(image_id))
            self._sleep_with_recording(2.0, recorder=recorder, heartbeat=heartbeat)
            imagine_end = recorder.sample_count
            segments.append(_VisualSegment(int(image_id), imagine_start, imagine_end, "exp_1_1_imagine"))

            iti_start = recorder.sample_count
            recorder.add_event("iti", exp="1.1", trial=idx, label_id=_VISUAL_REST_CLASS_ID)
            self._marker_backend.send(_VISUAL_REST_CLASS_ID)
            self._sleep_with_recording(1.5, recorder=recorder, heartbeat=heartbeat)
            iti_end = recorder.sample_count
            segments.append(_VisualSegment(_VISUAL_REST_CLASS_ID, iti_start, iti_end, "exp_1_1_iti"))

            trials.append(
                {
                    "exp": "1.1",
                    "trial_index": idx - 1,
                    "image_id": int(image_id),
                    "label_id": int(image_id),
                    "trial_start_sample": int(trial_start),
                    "imagine_start_sample": int(imagine_start),
                    "imagine_end_sample": int(imagine_end),
                }
            )
        recorder.add_event("exp_1_1_end")

    def _run_experiment_active(
        self,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        segments: list[_VisualSegment],
        trials: list[dict[str, Any]],
    ) -> None:
        raise RuntimeError("Active visual calibration requires Streamlit GUI interaction; run via WebGUI.")

    def _sleep_with_recording(
        self,
        duration_sec: float,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
    ) -> None:
        deadline = time.monotonic() + max(float(duration_sec), 0.0)
        while time.monotonic() < deadline:
            self._flush_recorder(recorder)
            if heartbeat is not None:
                heartbeat()
            time.sleep(min(0.02, max(deadline - time.monotonic(), 0.0)))
        self._flush_recorder(recorder)
        if heartbeat is not None:
            heartbeat()

    def _flush_recorder(self, recorder: SessionRecorder) -> None:
        try:
            recorder.pull()
        except RuntimeError as exc:
            message = str(exc).lower()
            if "stream" in message and "not started" in message:
                raise RuntimeError("采集中断：EEG 流意外停止。请检查设备并重试。") from exc
            raise

    def _build_windows(self, *, eeg: np.ndarray, segments: list[_VisualSegment]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        window_samples = int(round(self._window_sec * self._sfreq))
        stride_samples = int(round(self._step_sec * self._sfreq))
        start_offset = int(round(self._start_offset_sec * self._sfreq))
        stop_offset = int(round(self._stop_offset_sec * self._sfreq))

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
                window = eeg[:, start:stop].astype(np.float32)
                raw_windows.append(window)
                processed_windows.append(filter_and_transform(window, sfreq=self._sfreq))
                labels.append(int(seg.label_id))

        raw_X = np.stack(raw_windows, axis=0).astype(np.float32) if raw_windows else np.empty((0, eeg.shape[0], window_samples), dtype=np.float32)
        X = np.stack(processed_windows, axis=0).astype(np.float32) if processed_windows else np.empty((0, eeg.shape[0], window_samples), dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        return raw_X, X, y

    def _save_training_windows(self, output_path: Path, *, raw_X: np.ndarray, X: np.ndarray, y: np.ndarray) -> None:
        np.savez_compressed(
            output_path,
            raw_windows=raw_X,
            processed_windows=X,
            labels=y,
            sfreq=np.asarray([self._sfreq], dtype=np.float32),
            window_sec=np.asarray([self._window_sec], dtype=np.float32),
            step_sec=np.asarray([self._step_sec], dtype=np.float32),
        )

    def _save_metrics(self, *, metrics: dict[str, float], windows_collected: int, head_only: bool) -> None:
        payload = {
            "model_path": str(self._model_path),
            "task_mode": "visual",
            "windows_collected": int(windows_collected),
            "head_only": bool(head_only),
            "sfreq": float(self._sfreq),
            "window_sec": float(self._window_sec),
            "step_sec": float(self._step_sec),
            "metrics": dict(metrics),
        }
        path = self._model_path.with_suffix(".metrics.yaml")
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


class VisualRealTimeDecoder:
    def __init__(
        self,
        acquirer,
        model,
        console,
        command_outlet: LSLCommandOutlet,
        game_command_outlet: Any | None,
        *,
        sfreq: float,
        window_sec: float,
        step_sec: float,
        confidence_threshold: float,
        mc_dropout_passes: int,
        thread_context: Any | None = None,
        label_names: dict[int, str] | None = None,
        test_mode_prompts: dict[int, str] | None = None,
    ) -> None:
        del game_command_outlet
        self._acquirer = acquirer
        self._model = model
        self._console = console
        self._command_outlet = command_outlet
        self._sfreq = float(sfreq)
        self._window_sec = float(window_sec)
        self._step_sec = float(step_sec)
        self._confidence_threshold = float(confidence_threshold)
        self._mc_dropout_passes = int(mc_dropout_passes)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_context = thread_context
        self._fatal_exc: Exception | None = None
        self._label_names = dict(label_names or _VISUAL_LABEL_NAMES)
        self._test_mode_prompts = dict(test_mode_prompts or _VISUAL_TEST_MODE_PROMPTS)

    def start(self) -> None:
        self._acquirer.start_stream()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._decode_loop, daemon=True)
        if self._thread_context is not None:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(self._thread, self._thread_context)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._acquirer.stop_stream()

    def run_forever(
        self,
        *,
        subject_id: str | None = None,
        save_dir: Path | None = None,
        record: bool = False,
        heartbeat: Callable[[], None] | None = None,
    ) -> None:
        self._record = record
        self._subject_id = subject_id
        if record and subject_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_dir = save_dir or Path("records_storage") / subject_id / "realtime" / timestamp
            self._writer = StreamWriter(self._save_dir)
            self._writer.start(
                {
                    "subject_id": subject_id,
                    "mode": "realtime",
                    "task_mode": "visual",
                    "start_time": time.time(),
                    "sfreq": self._sfreq,
                    "window_sec": self._window_sec,
                    "step_sec": self._step_sec,
                    "channels": self._acquirer.metadata.n_channels,
                }
            )

        self.start()
        try:
            while True:
                self._sleep_with_heartbeat(min(0.1, max(self._step_sec, 0.1)), heartbeat)
                if heartbeat is not None:
                    heartbeat()
                if self._fatal_exc is not None:
                    raise RuntimeError(f"实时解码异常退出: {self._fatal_exc}") from self._fatal_exc
        except KeyboardInterrupt:
            self._console.print("\n[bold red]停止实时解码[/bold red]")
        finally:
            self.stop()
            if heartbeat is not None:
                heartbeat()
            if hasattr(self, "_writer"):
                self._writer.stop()
                self._writer.update_manifest({})
                self._console.print(f"[bold green]实时数据已保存[/bold green] {self._save_dir}")

    def run_test_mode(
        self,
        *,
        subject_id: str,
        marker_backend: MarkerBackend,
        duration_sec: int,
        block_sec: float = 10.0,
        save_dir: Path | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> dict[str, float | int | str]:
        del block_sec
        self._console.print("[bold cyan]测试模式启动（视觉 cue）[/bold cyan]")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = save_dir or Path("records_storage") / subject_id / "test_mode" / timestamp
        writer = StreamWriter(root_dir)
        writer.start(
            {
                "subject_id": subject_id,
                "mode": "test_mode",
                "task_mode": "visual",
                "start_time": time.time(),
                "sfreq": self._sfreq,
                "window_sec": self._window_sec,
                "step_sec": self._step_sec,
                "channels": self._acquirer.metadata.n_channels,
            }
        )

        rng = random.Random(17)
        labels = list(range(10)) + [_VISUAL_REST_CLASS_ID]
        rng.shuffle(labels)
        started = time.monotonic()
        cue_index = 0
        true_labels: list[int] = []
        pred_labels: list[int] = []
        confidences: list[float] = []
        consecutive_acquire_failures = 0

        self._acquirer.start_stream()
        try:
            while time.monotonic() - started < duration_sec:
                label = int(labels[cue_index % len(labels)])
                cue_index += 1
                prompt = self._test_mode_prompts.get(label, f"class-{label}")
                self._console.print(f"[bold yellow][cue][/bold yellow] {prompt}")
                if heartbeat is not None:
                    heartbeat()
                marker_backend.send(label)
                self._sleep_with_heartbeat(1.0, heartbeat)

                trial_end = time.monotonic() + 2.0
                while time.monotonic() < trial_end and time.monotonic() - started < duration_sec:
                    loop_started = time.perf_counter()
                    try:
                        eeg, _ = self._acquirer.get_chunk(self._window_sec)
                        consecutive_acquire_failures = 0
                    except RuntimeError as exc:
                        consecutive_acquire_failures += 1
                        self._console.print(f"[red]采集失败：{exc}[/red]")
                        if consecutive_acquire_failures >= 5:
                            raise RuntimeError(
                                f"测试模式连续 {consecutive_acquire_failures} 次采集失败，停止运行。最后错误: {exc}"
                            ) from exc
                        time.sleep(max(float(self._step_sec), 0.05))
                        continue
                    processed = filter_and_transform(eeg, sfreq=self._sfreq)
                    probabilities = self._model.predict_proba(
                        processed[None, ...],
                        mc_dropout_passes=self._mc_dropout_passes,
                    )[0]
                    result = self._post_process(probabilities)
                    self._console.print(
                        f"[green][预测][/green] {result.label} "
                        f"(confidence: {result.confidence:.2f}, uncertainty: {result.uncertainty:.2f})"
                    )
                    self._command_outlet.push(result.label)
                    pred_class = -1 if result.class_id is None else int(result.class_id)
                    writer.put(window=eeg.astype(np.float32), y_true=label, y_pred=pred_class, confidence=float(result.confidence))
                    true_labels.append(label)
                    pred_labels.append(pred_class)
                    confidences.append(float(result.confidence))
                    if heartbeat is not None:
                        heartbeat()
                    elapsed = time.perf_counter() - loop_started
                    self._sleep_with_heartbeat(max(0.0, self._step_sec - elapsed), heartbeat)
        finally:
            self._acquirer.stop_stream()
            if heartbeat is not None:
                heartbeat()

        if not true_labels:
            raise RuntimeError("Test mode did not collect any EEG windows.")

        y_true = np.asarray(true_labels, dtype=np.int64)
        y_pred = np.asarray(pred_labels, dtype=np.int64)
        accuracy = float(np.mean(y_pred == y_true))
        pred_valid = y_pred >= 0
        valid_accuracy = float(np.mean(y_pred[pred_valid] == y_true[pred_valid])) if np.any(pred_valid) else 0.0

        writer.stop()
        writer.update_manifest({"accuracy": accuracy, "valid_accuracy": valid_accuracy})
        self._console.print(f"[bold green]测试数据已保存[/bold green] {root_dir}")

        return {"windows": len(true_labels), "accuracy": accuracy, "valid_accuracy": valid_accuracy}

    def _decode_loop(self) -> None:
        consecutive_failures = 0
        while not self._stop_event.is_set():
            started_at = time.perf_counter()
            try:
                eeg, _ = self._acquirer.get_chunk(self._window_sec)
                consecutive_failures = 0  # 成功获取，重置失败计数
            except Exception as exc:
                consecutive_failures += 1
                self._console.print(f"[red]采集失败：{exc}[/red]")
                if consecutive_failures >= 5:
                    self._fatal_exc = exc
                    self._stop_event.set()
                    self._console.print(f"[red]连续 {consecutive_failures} 次采集失败，停止解码[/red]")
                    return
                # 等待一小段时间再重试，避免忙等
                wait = min(0.5, max(self._step_sec, 0.1))
                time.sleep(wait)
                continue

            # 正常处理采集到的数据
            processed = filter_and_transform(eeg, sfreq=self._sfreq)
            probabilities = self._model.predict_proba(
                processed[None, ...], mc_dropout_passes=self._mc_dropout_passes
            )[0]
            result = self._post_process(probabilities)
            self._console.print(
                f"[green][预测][/green] {result.label} "
                f"(confidence: {result.confidence:.2f}, uncertainty: {result.uncertainty:.2f})"
            )
            self._command_outlet.push(result.label)

            if hasattr(self, "_record") and self._record and hasattr(self, "_writer"):
                pred_class = -1 if result.class_id is None else int(result.class_id)
                self._writer.put(
                    window=eeg.astype(np.float32),
                    y_true=-1,
                    y_pred=pred_class,
                    confidence=float(result.confidence),
                )

            elapsed = time.perf_counter() - started_at
            sleep_dur = max(0.0, self._step_sec - elapsed)
            if sleep_dur > 0:
                time.sleep(sleep_dur)
                
    def _post_process(self, probabilities: np.ndarray) -> PredictionResult:
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        uncertainty = float(1.0 - confidence)
        if confidence < self._confidence_threshold:
            return PredictionResult(
                label=self._label_names[_VISUAL_REST_CLASS_ID],
                confidence=confidence,
                uncertainty=uncertainty,
                class_id=None,
            )
        return PredictionResult(
            label=self._label_names.get(best_index, f"class-{best_index}"),
            confidence=confidence,
            uncertainty=uncertainty,
            class_id=best_index,
        )

    @staticmethod
    def _sleep_with_heartbeat(duration_sec: float, heartbeat: Callable[[], None] | None) -> None:
        remaining = max(float(duration_sec), 0.0)
        while remaining > 0:
            chunk = min(0.1, remaining)
            time.sleep(chunk)
            remaining -= chunk
            if heartbeat is not None:
                heartbeat()

# ---------- 页面渲染函数 ----------
def render_home() -> None:
    st.title("Omni-Intelligence® 脑机接口系统")
    st.markdown(
        """
        欢迎你，受试者！

        在接下来的任务中，你将进行“视觉想象”实验。实验刺激将以**弹出全屏窗口**呈现（按 ESC 退出）。

        **实验 1.1 被动想象图片（10 次）**

        - 静息态基线 1s
        - 随机展示 1 张图片 1.5s（记忆）
        - 马赛克提示 0.5s → 黑屏
        - 回忆并想象该图片 2s
        - 间隔 1.5s

        **实验 1.2 主动想象图片（10 次）**

        - 静息态基线 1s
        - 黑屏 → 无刺激主动想象任意图片 2s
        - 展示 10 张图（两排）→ 点击你刚才想象的图 → 该图消除
        - 间隔 1.5s → 黑屏 → 想象下一张，直到全部选完

        注意：当前所有图片/黑屏暂用 `assets/EEG.png` 代替。
        """
    )

def render_settings(config: dict) -> None:
    st.title("核心参数配置")
    register_default_acquirers()

    protocol_cfg = config.setdefault("protocol", {})
    trial_timing_cfg = protocol_cfg.setdefault("trial_timing", {})
    output_cfg = config.setdefault("output", {})
    ar_game_cfg = output_cfg.setdefault("ar_game", {})

    subject_id = st.text_input("被试 ID (subject_id)", value=str(config.get("subject_id", "S001")))
    models = ModelFactory.list_models()
    model_name = st.selectbox(
        "默认模型 (model_name)",
        models,
        index=models.index(str(config.get("model_name", "riemann-mdm"))),
    )
    devices = AcquirerFactory.list_devices()
    current_device = str(config.get("device_type", devices[0]))
    device_type = st.selectbox(
        "采集设备 (device_type)",
        devices,
        index=devices.index(current_device) if current_device in devices else 0,
    )

    base_col1, base_col2 = st.columns(2)
    window_sec = float(
        base_col1.number_input(
            "特征窗长 (window_sec / 秒)",
            min_value=0.5,
            value=float(config.get("window_sec", 2.0)),
            step=0.25,
        )
    )
    step_sec = float(
        base_col2.number_input(
            "步长/刷新时间 (step_sec / 秒)",
            min_value=0.05,
            value=float(config.get("step_sec", 0.5)),
            step=0.05,
        )
    )

    st.markdown("### MI Game Control Protocol")
    protocol_col1, protocol_col2, protocol_col3 = st.columns(3)
    control_start_offset_sec = float(
        protocol_col1.number_input(
            "control 有效起点 (秒)",
            min_value=0.0,
            value=float(protocol_cfg.get("control_start_offset_sec", 0.5)),
            step=0.1,
        )
    )
    fixation_sec = float(
        protocol_col2.number_input(
            "fixation 时长 (秒)",
            min_value=0.5,
            value=float(trial_timing_cfg.get("fixation_sec", 2.0)),
            step=0.5,
        )
    )
    cue_sec = float(
        protocol_col3.number_input(
            "cue 时长 (秒)",
            min_value=0.5,
            value=float(trial_timing_cfg.get("cue_sec", 1.0)),
            step=0.5,
        )
    )
    protocol_col4, protocol_col5, protocol_col6 = st.columns(3)
    control_sec = float(
        protocol_col4.number_input(
            "control 时长 (秒)",
            min_value=1.0,
            value=float(trial_timing_cfg.get("control_sec", 5.0)),
            step=0.5,
        )
    )
    iti_sec = float(
        protocol_col5.number_input(
            "iti 时长 (秒)",
            min_value=0.5,
            value=float(trial_timing_cfg.get("iti_sec", 2.0)),
            step=0.5,
        )
    )
    rest_between_blocks_sec = float(
        protocol_col6.number_input(
            "block 间休息 (秒)",
            min_value=0.0,
            value=float(protocol_cfg.get("rest_between_blocks_sec", 35.0)),
            step=5.0,
        )
    )
    subject_col1, subject_col2, subject_col3 = st.columns(3)
    new_subject_blocks = int(
        subject_col1.number_input(
            "新被试 block 数",
            min_value=1,
            value=int(protocol_cfg.get("new_subject_blocks", 6)),
            step=1,
        )
    )
    new_subject_trials_per_class_per_block = int(
        subject_col2.number_input(
            "新被试每类每 block trial 数",
            min_value=1,
            value=int(protocol_cfg.get("new_subject_trials_per_class_per_block", 8)),
            step=1,
        )
    )
    old_subject_trials_per_class = int(
        subject_col3.number_input(
            "老被试每类 trial 数",
            min_value=1,
            value=int(protocol_cfg.get("old_subject_trials_per_class", 8)),
            step=1,
        )
    )
    old_subject_baseline_sec = float(
        st.number_input(
            "老被试 baseline idle 时长 (秒)",
            min_value=1.0,
            value=float(protocol_cfg.get("old_subject_baseline_sec", 60.0)),
            step=5.0,
        )
    )

    st.markdown("### AR 游戏控制")
    ar_col1, ar_col2, ar_col3 = st.columns(3)
    ar_game_enabled = ar_col1.checkbox("启用 AR 游戏 TCP 控制", value=bool(ar_game_cfg.get("enabled", False)))
    ar_game_host = ar_col2.text_input("AR 游戏主机", value=str(ar_game_cfg.get("host", "127.0.0.1")))
    ar_game_port = int(
        ar_col3.number_input(
            "AR 游戏端口",
            min_value=1,
            max_value=65535,
            value=int(ar_game_cfg.get("port", 5005)),
            step=1,
        )
    )
    ar_game_timeout_sec = float(
        st.number_input(
            "AR 游戏 TCP 超时 (秒)",
            min_value=0.1,
            value=float(ar_game_cfg.get("timeout_sec", 1.0)),
            step=0.1,
        )
    )

    if st.button("保存配置", type="primary"):
        config.update(
            {
                "subject_id": subject_id,
                "model_name": model_name,
                "device_type": device_type,
                "window_sec": window_sec,
                "step_sec": step_sec,
            }
        )
        protocol_cfg.update(
            {
                "control_start_offset_sec": control_start_offset_sec,
                "trial_timing": {
                    "fixation_sec": fixation_sec,
                    "cue_sec": cue_sec,
                    "control_sec": control_sec,
                    "iti_sec": iti_sec,
                },
                "new_subject_blocks": new_subject_blocks,
                "new_subject_trials_per_class_per_block": new_subject_trials_per_class_per_block,
                "old_subject_baseline_sec": old_subject_baseline_sec,
                "old_subject_trials_per_class": old_subject_trials_per_class,
                "rest_between_blocks_sec": rest_between_blocks_sec,
            }
        )
        output_cfg["ar_game"] = {
            "enabled": ar_game_enabled,
            "host": ar_game_host,
            "port": ar_game_port,
            "timeout_sec": ar_game_timeout_sec,
        }
        _save_config(config)
        st.success("配置已保存。")

def render_probe(config: dict) -> None:
    st.title("连通检测")
    st.markdown("在正式开始前，先确认采集设备网络可达并能返回 EEG 数据。")
    dur = st.number_input("探测时长 (秒)", min_value=0.1, value=3.0, step=0.5)

    if st.button("开始探测", type="primary"):
        selected_device = str(config.get("device_type", "neuracle"))

        with st.spinner(f"正在尝试连接 {selected_device} ..."):
            try:
                acquirer = build_acquirer(device_name=selected_device, config=config)
                st.info(f"设备对象已创建。尝试读取 {dur:.1f} 秒数据...")
                acquirer.start_stream()
                time.sleep(max(dur, 0.1))
                window, _ = acquirer.get_chunk(float(config.get("window_sec", 2.0)))
                acquirer.stop_stream()

                st.success("设备连通正常。")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Shape", str(window.shape))
                col2.metric("Mean (uV)", f"{window.mean():.3f}")
                col3.metric("Std (uV)", f"{window.std():.3f}")
                col4.metric("Max Abs (uV)", f"{abs(window).max():.3f}")
            except Exception as exc:
                st.error(f"连通失败: {exc}")

def render_calibration(config: dict) -> None:
    st.title("被试校准")
    st.markdown("图片/文本刺激与选择在弹出窗口中完成。本页仅显示日志。")
    experiment = st.radio(
        "实验选择",
        [
            "实验1.1 被动想象",
            "实验1.2 主动想象",
            "实验2.1 文本被动想象",
            "实验2.2 文本主动想象",
        ],
    )
    st.caption("弹窗默认全屏显示；按 ESC 退出全屏。多 block 时，block 间会隐藏窗口并按空格继续。")

    if st.button("开始", type="primary"):
        try:
            subject_id = str(config["subject_id"])
            model_name = str(config["model_name"])
            acquirer = build_acquirer(device_name=str(config["device_type"]), config=config)
            effective_n_channels = int(acquirer.metadata.n_channels)
            console, refresh = init_live_view()
            task = load_task_from_config(config)
            task_console = task.wrap_console(console)
            marker_backend = task.wrap_marker_backend(build_marker_backend(config))
            model = ModelFactory.get(
                model_name,
                n_chans=effective_n_channels,
                sfreq=float(config["sfreq"]),
                n_classes=int(config["n_classes"]),
                n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
            )
            model_path = build_model_path(config, subject_id, model_name, device_name=str(config["device_type"]))
            records_dir = Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
            session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = records_dir / subject_id / "calibration" / session_stamp

            stimulus_path = _resolve_asset_path("EEG.png")
            if stimulus_path is None:
                raise RuntimeError("缺少刺激图片: assets/EEG.png")

            with st.spinner("校准进行中..."):
                if experiment == "实验1.1 被动想象":
                    result = run_visual_exp_1_1(
                        config=config,
                        acquirer=acquirer,
                        model=model,
                        marker_backend=marker_backend,
                        console=task_console,
                        refresh=refresh,
                        model_path=model_path,
                        session_dir=session_dir,
                        stimulus_image_path=stimulus_path,
                    )
                elif experiment == "实验1.2 主动想象":
                    result = run_visual_exp_1_2(
                        config=config,
                        acquirer=acquirer,
                        model=model,
                        marker_backend=marker_backend,
                        console=task_console,
                        refresh=refresh,
                        model_path=model_path,
                        session_dir=session_dir,
                        stimulus_image_path=stimulus_path,
                    )
                elif experiment == "实验2.1 文本被动想象":
                    result = run_text_exp_1_1(
                        config=config,
                        acquirer=acquirer,
                        model=model,
                        marker_backend=marker_backend,
                        console=task_console,
                        refresh=refresh,
                        model_path=model_path,
                        session_dir=session_dir,
                        stimulus_image_path=stimulus_path,
                    )
                else:
                    result = run_text_exp_1_2(
                        config=config,
                        acquirer=acquirer,
                        model=model,
                        marker_backend=marker_backend,
                        console=task_console,
                        refresh=refresh,
                        model_path=model_path,
                        session_dir=session_dir,
                        stimulus_image_path=stimulus_path,
                    )

            refresh()
            st.success("校准完成。")
            st.write(f"- 采集窗口数: **{result.windows_collected}**")
            st.write(f"- 模型保存位置: `{result.model_path}`")
            if result.session_dir is not None:
                st.write(f"- session 保存位置: `{result.session_dir}`")
            if result.calibration_data_path is not None:
                st.write(f"- 校准数据保存位置: `{result.calibration_data_path}`")
        except Exception as exc:
            st.exception(exc)

def render_test_mode(config: dict) -> None:
    st.title("Cue 测试模式")
    st.markdown("运行过程中会展示视觉 cue（图片1~10 + 静息）并展示模型输出日志。")
    duration = st.number_input("测试总时长 (秒)", min_value=30, value=120, step=30)

    if st.button("开始测试", type="primary"):
        try:
            subject_id = str(config["subject_id"])
            model_name = str(config["model_name"])
            acquirer = build_acquirer(
                device_name=str(config["device_type"]),
                config=config,
            )
            effective_n_channels = int(acquirer.metadata.n_channels)
            console, refresh = init_live_view()
            task = load_task_from_config(config)
            task_console = task.wrap_console(console)
            model = ModelFactory.get(
                model_name,
                n_chans=effective_n_channels,
                sfreq=float(config["sfreq"]),
                n_classes=int(config["n_classes"]),
                n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
            )
            model_path = build_model_path(
                config,
                subject_id,
                model_name,
                device_name=str(config["device_type"]),
            )
            if not model_path.exists():
                st.error(f"未找到模型权重文件: {model_path}。请先执行校准。")
                return
            model.load(model_path)

            command_outlet = LSLCommandOutlet(
                stream_name=str(config["output"]["command_stream_name"]),
                stream_type=str(config["output"]["command_stream_type"]),
            )
            decoder = VisualRealTimeDecoder(
                acquirer=acquirer,
                model=model,
                console=task_console,
                command_outlet=command_outlet,
                game_command_outlet=build_game_command_outlet(config),
                sfreq=float(config["sfreq"]),
                window_sec=float(config["window_sec"]),
                step_sec=float(config["step_sec"]),
                confidence_threshold=float(config["confidence_threshold"]),
                mc_dropout_passes=int(config["mc_dropout_passes"]),
                label_names=visual_label_names(config),
                test_mode_prompts=visual_test_mode_prompts(config),
            )

            with st.spinner("测试模式采集中..."):
                result = decoder.run_test_mode(
                    subject_id=subject_id,
                    marker_backend=task.wrap_marker_backend(build_marker_backend(config)),
                    duration_sec=int(duration),
                    block_sec=float(config.get("collect_block_sec", 10.0)),
                    save_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
                    / subject_id
                    / "test_mode",
                    heartbeat=refresh,
                )

            refresh()
            st.success("测试结束。")
            st.write(f"- 记录的窗口数: **{result['windows']}**")
            st.write(f"- 准确率: **{result['accuracy']:.3f}**")
            st.write(f"- 有效准确率: **{result['valid_accuracy']:.3f}**")
        except Exception as exc:
            st.exception(exc)

def render_realtime(config: dict) -> None:
    st.title("实时解码")
    st.markdown("开始后会持续显示模型输出。")
    record = st.checkbox("保存实时脑波数据至本地记录")

    if st.button("开始实时解码", type="primary"):
        try:
            subject_id = str(config["subject_id"])
            model_name = str(config["model_name"])
            acquirer = build_acquirer(
                device_name=str(config["device_type"]),
                config=config,
            )
            effective_n_channels = int(acquirer.metadata.n_channels)
            console, refresh = init_live_view()
            task = load_task_from_config(config)
            task_console = task.wrap_console(console)
            model = ModelFactory.get(
                model_name,
                n_chans=effective_n_channels,
                sfreq=float(config["sfreq"]),
                n_classes=int(config["n_classes"]),
                n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
            )
            model_path = build_model_path(
                config,
                subject_id,
                model_name,
                device_name=str(config["device_type"]),
            )
            if not model_path.exists():
                st.error(f"未找到模型权重文件: {model_path}。请先执行校准。")
                return
            model.load(model_path)

            decoder = VisualRealTimeDecoder(
                acquirer=acquirer,
                model=model,
                console=task_console,
                command_outlet=LSLCommandOutlet(
                    stream_name=str(config["output"]["command_stream_name"]),
                    stream_type=str(config["output"]["command_stream_type"]),
                ),
                game_command_outlet=build_game_command_outlet(config),
                sfreq=float(config["sfreq"]),
                window_sec=float(config["window_sec"]),
                step_sec=float(config["step_sec"]),
                confidence_threshold=float(config["confidence_threshold"]),
                mc_dropout_passes=int(config["mc_dropout_passes"]),
                label_names=visual_label_names(config),
                test_mode_prompts=visual_test_mode_prompts(config),
            )

            with st.spinner("实时解码运行中..."):
                decoder.run_forever(
                    subject_id=subject_id,
                    record=record,
                    save_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
                    / subject_id
                    / "realtime",
                    heartbeat=refresh,
                )
        except Exception as exc:
            st.warning(f"解码已停止: {exc}")

# ---------- 导航与样式 ----------
def _set_gui_nav_mode(page: str) -> None:
    st.session_state.gui_nav_mode = page

def _inject_gui_nav_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMainBlockContainer"] {
          background-color: #ffffff;
          color: #0f172a;
        }
        [data-testid="stHeader"] {
          background-color: #ffffff;
        }
        [data-testid="stToolbar"] {
          color: #334155;
        }
        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, #fff7ed 0%, #ffffff 70%);
          border-right: 1px solid rgba(15, 23, 42, 0.08);
        }
        section[data-testid="stSidebar"] * {
          color: #1e293b;
        }
        section[data-testid="stSidebar"] .stButton > button {
          width: 100%;
          border-radius: 10px;
          padding-top: 0.72rem;
          padding-bottom: 0.72rem;
          font-weight: 600;
          font-size: 0.95rem;
          margin-bottom: 0.35rem;
          outline: none;
          transition: background-color 0.12s ease, border-color 0.12s ease, color 0.12s ease;
        }
        section[data-testid="stSidebar"] .stButton > button:focus-visible {
          box-shadow: 0 0 0 2px rgba(255, 90, 1, 0.4);
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
          background-color: rgba(248, 250, 252, 0.95);
          border: 1px solid rgba(15, 23, 42, 0.12);
          color: rgb(30, 41, 59);
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
          border-color: rgba(255, 90, 1, 0.4);
          background-color: rgba(255, 90, 1, 0.07);
          color: rgb(15, 23, 42);
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"],
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] *,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover * {
          color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"],
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover,
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] *,
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover * {
          color: #ffffff !important;
        }
        .stMarkdown, .stText, p, label, h1, h2, h3, h4, h5, h6 {
          color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------- 主渲染入口 ----------
def render_motor_gui(config: dict, config_path: Path) -> None:
    """Visual 任务完整界面（与 motor 保持一致的页面结构）"""
    global _current_config_path
    _current_config_path = config_path

    _inject_gui_nav_styles()
    st.session_state.setdefault("gui_nav_mode", SIDEBAR_NAV_PAGES[0])

    with st.sidebar:
        logo_path = _resolve_logo_svg_path()
        if logo_path is not None:
            st.image(str(logo_path), width=280)
        st.title("oi-mi 工作台")
        for page in SIDEBAR_NAV_PAGES:
            is_active = st.session_state.gui_nav_mode == page
            st.button(
                page,
                key=f"nav_btn_{page}",
                type="primary" if is_active else "secondary",
                width="stretch",
                on_click=_set_gui_nav_mode,
                args=(page,),
            )
        mode = st.session_state.gui_nav_mode

    if mode == "首页":
        render_home()
    elif mode == "设置":
        render_settings(config)
    elif mode == "连通检测":
        render_probe(config)
    elif mode == "校准":
        render_calibration(config)
    elif mode == "测试模式":
        render_test_mode(config)
    elif mode == "实时解码":
        render_realtime(config)

# ---------- 任务工厂接口 ----------
def get_streamlit_renderer():
    """返回 visual 任务的 Streamlit 渲染器，供 task_factory 调用"""
    return render_motor_gui

def get_calibrator_class():
    return VisualCalibrator

def get_realtime_decoder_class():
    return VisualRealTimeDecoder

# ---------- 原有 Task 包装类（保持不变） ----------
class Task:
    def wrap_console(self, console: Any) -> Any:
        return console

    def wrap_marker_backend(self, backend: Any) -> Any:
        return backend
