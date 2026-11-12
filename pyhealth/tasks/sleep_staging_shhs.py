"""Sleep-stage classification for Sleep Heart Health Study recordings."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, ClassVar

import mne
import numpy as np

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)


class SleepStagingSHHS(BaseTask):
    """Create one EEG sample per scored SHHS sleep-stage epoch.

    The SHHS Profusion XML files score the recording in 30-second epochs with
    integer codes: 0 (wake), 1-4 (NREM stages), and 5 (REM). SHHS scoring
    combines historical stages 3 and 4 into one deep-sleep class. Codes outside
    0-5 (for example, active or unscored intervals) are excluded.

    A single EEG lead is used to keep the signal shape consistent when a
    recording is missing one of its two EEG channels. By default the task
    prefers ``EEG (sec)``, then ``EEG``, then the first channel whose name
    contains ``EEG``. Set ``eeg_channel`` to select a specific channel name.
    Signals are resampled to 125 Hz by default so SHHS recordings with different
    native sampling rates still produce the same input shape.

    Args:
        epoch_seconds: Duration of each returned sample in seconds. Must divide
            the 30-second scoring interval evenly. Defaults to 30.
        resample_rate: Target EEG rate in Hz. Defaults to 125, matching the
            typical SHHS1 EEG rate. Must be positive.
        eeg_channel: Optional exact channel name to use. If omitted, a suitable
            EEG channel is selected from the recording's channel names.

    Each returned sample contains ``patient_id``, ``record_id``, ``visit``,
    ``signal`` (shape ``[1, resample_rate * epoch_seconds]``), and ``label``.
    Labels are integers: 0=Wake, 1=N1, 2=N2, 3=Deep (historical stages 3/4),
    4=REM.

    Examples:
        >>> from pyhealth.datasets import SHHSDataset
        >>> from pyhealth.tasks import SleepStagingSHHS
        >>> dataset = SHHSDataset(root="/path/to/shhs/polysomnography")
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     SleepStagingSHHS(), num_workers=1
        ... )
        >>> samples[0]["signal"].shape  # doctest: +SKIP
        torch.Size([1, 3750])
    """

    task_name: ClassVar[str] = "SleepStagingSHHS"
    input_schema: ClassVar[dict[str, Any]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, Any]] = {
        "label": ("multiclass", {"labels": [0, 1, 2, 3, 4]})
    }

    STAGE_MAP: ClassVar[dict[int, int]] = {
        0: 0,  # Wake
        1: 1,  # N1
        2: 2,  # N2
        3: 3,  # Deep sleep (historical N3)
        4: 3,  # Deep sleep (historical N4)
        5: 4,  # REM
    }
    PREFERRED_EEG_CHANNELS: ClassVar[tuple[str, ...]] = ("EEG (sec)", "EEG")

    def __init__(
        self,
        epoch_seconds: int = 30,
        resample_rate: float = 125.0,
        eeg_channel: str | None = None,
    ) -> None:
        super().__init__()
        if (
            isinstance(epoch_seconds, bool)
            or epoch_seconds <= 0
            or 30 % epoch_seconds
        ):
            raise ValueError("epoch_seconds must be a positive divisor of 30.")
        if resample_rate <= 0:
            raise ValueError("resample_rate must be positive.")
        self.epoch_seconds = int(epoch_seconds)
        self.resample_rate = float(resample_rate)
        self.eeg_channel = eeg_channel

    @staticmethod
    def _parse_stage_codes(label_file: str) -> list[int | None]:
        """Read 30-second Profusion stage codes, keeping unknown epochs in place."""
        root = ET.parse(label_file).getroot()
        sleep_stages = root.find(".//SleepStages")
        if sleep_stages is None:
            raise ValueError(f"No <SleepStages> section found in {label_file}")

        codes: list[int | None] = []
        for node in sleep_stages.findall("SleepStage"):
            try:
                codes.append(int(node.text))
            except (TypeError, ValueError):
                # Keep the time position so a later valid label is not shifted.
                codes.append(None)
        return codes

    def _choose_eeg_channel(self, channel_names: list[str]) -> str | None:
        """Pick one EEG channel by its EDF name rather than a fixed row index."""
        if self.eeg_channel is not None:
            by_casefold = {name.casefold(): name for name in channel_names}
            return by_casefold.get(self.eeg_channel.casefold())

        by_normalized = {
            "".join(ch for ch in name.casefold() if ch.isalnum()): name
            for name in channel_names
        }
        for preferred in self.PREFERRED_EEG_CHANNELS:
            normalized = "".join(ch for ch in preferred.casefold() if ch.isalnum())
            if normalized in by_normalized:
                return by_normalized[normalized]

        for name in channel_names:
            lowered = name.casefold()
            if "eeg" in lowered and "eog" not in lowered:
                return name
        return None

    def read_recording(self, signal_file: str) -> tuple[np.ndarray, float] | None:
        """Read one EDF, select one named EEG lead, and resample to target rate.

        Returns the signal as ``(channels, samples)`` in MNE's native voltage
        units, plus the resulting sampling rate. Unreadable files or files with
        no usable EEG lead are skipped with a warning.
        """
        try:
            raw = mne.io.read_raw_edf(signal_file, preload=True, verbose="error")
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Could not read SHHS EDF %s: %s", signal_file, exc)
            return None

        try:
            source_rate = float(raw.info["sfreq"])
            channel_name = self._choose_eeg_channel(list(raw.ch_names))
            if channel_name is None:
                logger.warning(
                    "No EEG channel found in %s (channels: %s); skipping recording.",
                    signal_file,
                    raw.ch_names,
                )
                return None

            if not np.isclose(source_rate, self.resample_rate):
                raw.resample(self.resample_rate, n_jobs=1, verbose="error")

            signal = raw.get_data(picks=[channel_name]).astype(
                np.float32, copy=False
            )
            return signal, float(raw.info["sfreq"])
        finally:
            raw.close()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Turn each patient's SHHS EDF/XML recording pair into epoch samples."""
        samples: list[dict[str, Any]] = []
        subepochs_per_score = 30 // self.epoch_seconds

        for event in patient.get_events(event_type="recordings"):
            recording = self.read_recording(event.signal_file)
            if recording is None:
                continue
            signal, sampling_rate = recording
            samples_per_epoch = int(round(sampling_rate * self.epoch_seconds))
            if samples_per_epoch <= 0:
                logger.warning(
                    "Invalid sampling rate %.6g Hz in %s; skipping recording.",
                    sampling_rate,
                    event.signal_file,
                )
                continue

            raw_codes = self._parse_stage_codes(event.label_file)
            epoch_labels: list[int | None] = []
            for code in raw_codes:
                label = self.STAGE_MAP.get(code) if code is not None else None
                epoch_labels.extend([label] * subepochs_per_score)

            available_signal_epochs = signal.shape[1] // samples_per_epoch
            aligned_epochs = min(available_signal_epochs, len(epoch_labels))
            if available_signal_epochs != len(epoch_labels):
                logger.warning(
                    "EDF/XML length mismatch for %s: %d signal epochs, "
                    "%d label epochs; using %d.",
                    event.record_id,
                    available_signal_epochs,
                    len(epoch_labels),
                    aligned_epochs,
                )

            for epoch_index in range(aligned_epochs):
                label = epoch_labels[epoch_index]
                if label is None:
                    continue
                start = epoch_index * samples_per_epoch
                end = start + samples_per_epoch
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": f"{event.record_id}-{epoch_index}",
                        "visit": event.visit,
                        "signal": signal[:, start:end],
                        "label": label,
                    }
                )

        return samples
