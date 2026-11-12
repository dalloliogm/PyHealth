"""Synthetic tests for the BaseTask SHHS sleep-staging path."""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pyhealth.tasks.sleep_staging_shhs import SleepStagingSHHS


@dataclass
class _Event:
    record_id: str
    visit: str
    signal_file: str
    label_file: str


class _Patient:
    def __init__(self, patient_id: str, events: list[_Event]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None):
        if event_type not in (None, "recordings"):
            return []
        return self._events


class TestSleepStagingSHHS(unittest.TestCase):
    def _write_labels(self, directory: Path, codes: list[str]) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        values = "".join(f"<SleepStage>{code}</SleepStage>" for code in codes)
        path = directory / "shhs1-200001-profusion.xml"
        path.write_text(
            "<?xml version='1.0'?><PSGAnnotation><SleepStages>"
            + values
            + "</SleepStages></PSGAnnotation>"
        )
        return str(path)

    def _patient(self, directory: Path, codes: list[str]) -> _Patient:
        label_file = self._write_labels(directory, codes)
        return _Patient(
            "200001",
            [
                _Event(
                    record_id="shhs1-200001",
                    visit="shhs1",
                    signal_file="/synthetic/shhs1-200001.edf",
                    label_file=label_file,
                )
            ],
        )

    def test_stage_codes_map_to_five_classes_and_bad_codes_keep_time_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            patient = self._patient(Path(tmp), ["0", "9", "1", "2", "3", "4", "5"])
            signal = np.arange(7 * 30 * 125, dtype=np.float32).reshape(1, -1)
            task = SleepStagingSHHS()
            with patch.object(task, "read_recording", return_value=(signal, 125.0)):
                samples = task(patient)

        self.assertEqual(
            [sample["label"] for sample in samples], [0, 1, 2, 3, 3, 4]
        )
        self.assertEqual(
            [sample["record_id"] for sample in samples],
            [
                "shhs1-200001-0",
                "shhs1-200001-2",
                "shhs1-200001-3",
                "shhs1-200001-4",
                "shhs1-200001-5",
                "shhs1-200001-6",
            ],
        )
        self.assertTrue(
            all(sample["signal"].shape == (1, 30 * 125) for sample in samples)
        )
        self.assertTrue(all(sample["patient_id"] == "200001" for sample in samples))
        self.assertTrue(all(sample["visit"] == "shhs1" for sample in samples))

    def test_edf_shorter_than_xml_is_truncated_without_shifting_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            patient = self._patient(Path(tmp), ["0", "1", "2", "3"])
            signal = np.zeros((1, 2 * 30 * 125), dtype=np.float32)
            task = SleepStagingSHHS()
            with patch.object(task, "read_recording", return_value=(signal, 125.0)):
                samples = task(patient)

        self.assertEqual([sample["label"] for sample in samples], [0, 1])
        self.assertEqual(len(samples), 2)

    def test_subepochs_repeat_the_scored_stage_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            patient = self._patient(Path(tmp), ["3", "5"])
            signal = np.zeros((1, 4 * 15 * 125), dtype=np.float32)
            task = SleepStagingSHHS(epoch_seconds=15)
            with patch.object(task, "read_recording", return_value=(signal, 125.0)):
                samples = task(patient)

        self.assertEqual([sample["label"] for sample in samples], [3, 3, 4, 4])
        self.assertTrue(
            all(sample["signal"].shape == (1, 15 * 125) for sample in samples)
        )

    def test_channel_selection_uses_mne_names_and_supports_fallback(self):
        task = SleepStagingSHHS()
        self.assertEqual(
            task._choose_eeg_channel(["ECG", "EEG", "EEG (sec)"]), "EEG (sec)"
        )
        self.assertEqual(task._choose_eeg_channel(["ECG", "EEG1"]), "EEG1")
        self.assertIsNone(task._choose_eeg_channel(["ECG", "EOG(L)"]))
        self.assertEqual(
            SleepStagingSHHS(eeg_channel="C3-A2")._choose_eeg_channel(
                ["EEG", "C3-A2"]
            ),
            "C3-A2",
        )

    def test_epoch_seconds_must_divide_original_30_second_labels(self):
        with self.assertRaisesRegex(ValueError, "divisor of 30"):
            SleepStagingSHHS(epoch_seconds=7)


if __name__ == "__main__":
    unittest.main()
