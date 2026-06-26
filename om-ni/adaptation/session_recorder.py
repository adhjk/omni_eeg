"""Continuous session recording helpers for calibration runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from acquisition.base import AbstractAcquirer


@dataclass(slots=True)
class SessionEvent:
    name: str
    sample_index: int
    relative_time_sec: float
    payload: dict[str, Any]


class SessionRecorder:
    """Collect continuous EEG and aligned events during one protocol session."""

    def __init__(self, acquirer: AbstractAcquirer, *, sfreq: float, n_channels: int) -> None:
        self._acquirer = acquirer
        self._sfreq = float(sfreq)
        self._n_channels = int(n_channels)
        self._chunks: list[np.ndarray] = []
        self._events: list[SessionEvent] = []
        self._sample_count = 0
        self._started_at = time.monotonic()

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def events(self) -> list[SessionEvent]:
        return list(self._events)

    def pull(self) -> np.ndarray:
        samples, _timestamps = self._acquirer.get_new_samples()
        if samples.size == 0:
            return np.empty((self._n_channels, 0), dtype=np.float32)
        if samples.ndim != 2 or samples.shape[0] < self._n_channels:
            raise RuntimeError(f"Unexpected incremental EEG shape: {samples.shape}")
        eeg = np.asarray(samples[: self._n_channels], dtype=np.float32)
        self._chunks.append(eeg)
        self._sample_count += int(eeg.shape[1])
        return eeg

    def add_event(self, name: str, **payload: Any) -> None:
        self._events.append(
            SessionEvent(
                name=name,
                sample_index=self._sample_count,
                relative_time_sec=time.monotonic() - self._started_at,
                payload=dict(payload),
            )
        )

    def _calculate_event_duration(self, event: SessionEvent) -> float:
        idx = self._events.index(event)
        if idx + 1 < len(self._events):
            next_event = self._events[idx + 1]
            return next_event.relative_time_sec - event.relative_time_sec
        return 0.0

    def export_events_tsv(self, output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8") as f:
            f.write("onset\tduration\ttrial_type\tsample\tvalue\n")
            for event in self._events:
                onset = event.relative_time_sec
                duration = self._calculate_event_duration(event)
                trial_type = event.name
                sample = event.sample_index
                value = event.payload.get("marker_code", "")
                f.write(f"{onset:.6f}\t{duration:.6f}\t{trial_type}\t{sample}\t{value}\n")

    def export_bids(
        self,
        base_dir: Path | str,
        *,
        subject_id: str,
        session_id: str,
        task_name: str,
        metadata: dict[str, Any],
    ) -> Path:
        base_dir = Path(base_dir)
        bids_dir = base_dir / f"sub-{subject_id}" / f"ses-{session_id}" / f"task-{task_name}"
        bids_dir.mkdir(parents=True, exist_ok=True)

        eeg = self.to_array()
        eeg_filename = f"sub-{subject_id}_ses-{session_id}_task-{task_name}_eeg.npy"
        np.save(bids_dir / eeg_filename, eeg)

        events_filename = f"sub-{subject_id}_ses-{session_id}_task-{task_name}_events.tsv"
        self.export_events_tsv(bids_dir / events_filename)

        metadata_filename = f"sub-{subject_id}_ses-{session_id}_task-{task_name}_metadata.json"
        with (bids_dir / metadata_filename).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        return bids_dir

    def export(self, output_dir: Path, *, metadata: dict[str, Any]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        eeg = self.to_array()
        np.save(output_dir / "continuous_eeg.npy", eeg)

        self.export_events_tsv(output_dir / "events.tsv")

        with (output_dir / "events.json").open("w", encoding="utf-8") as handle:
            json.dump([asdict(event) for event in self._events], handle, ensure_ascii=False, indent=2)
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        return output_dir

    def to_array(self) -> np.ndarray:
        try:
            self.pull()
        except RuntimeError as exc:
            if not self._is_stream_not_started_error(exc):
                raise
        if not self._chunks:
            return np.empty((self._n_channels, 0), dtype=np.float32)
        return np.concatenate(self._chunks, axis=1).astype(np.float32)

    @staticmethod
    def _is_stream_not_started_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "not started" in message and "stream" in message
