"""SHHS metadata indexing tests using synthetic file trees."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pyhealth.datasets.shhs import SHHSDataset, _metadata_rows


class TestSHHSDatasetMetadata(unittest.TestCase):
    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def _write_fixture(self, root: Path) -> None:
        self._touch(root / "edfs/shhs1/shhs1-200001.edf")
        self._touch(root / "edfs/shhs2/shhs2-200001.edf")
        self._touch(root / "edfs/shhs1/shhs1-200002.edf")
        (root / "annotations-events-profusion/shhs1").mkdir(parents=True)
        (root / "annotations-events-profusion/shhs1/shhs1-200001-profusion.xml").write_text(
            "<PSGAnnotation />"
        )
        # Older SHHS2 archives use the "label" directory and omit the suffix.
        (root / "annotations-events-profusion/label").mkdir(parents=True)
        (root / "annotations-events-profusion/label/shhs2-200001.xml").write_text(
            "<PSGAnnotation />"
        )

    def test_indexes_both_visits_under_the_same_participant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)

            rows = _metadata_rows(root)

            self.assertEqual([row["record_id"] for row in rows], [
                "shhs1-200001",
                "shhs2-200001",
            ])
            self.assertEqual({row["patient_id"] for row in rows}, {"200001"})
            self.assertEqual([row["visit"] for row in rows], ["shhs1", "shhs2"])
            for row in rows:
                self.assertTrue(Path(row["signal_file"]).is_absolute())
                self.assertTrue(Path(row["label_file"]).is_absolute())
                self.assertTrue(Path(row["signal_file"]).is_file())
                self.assertTrue(Path(row["label_file"]).is_file())

    def test_skips_edf_without_annotation_and_bad_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._touch(root / "edfs/shhs1/shhs1-200001.edf")
            self._touch(root / "edfs/shhs1/not-an-shhs-recording.edf")
            (root / "annotations-events-profusion/shhs1").mkdir(parents=True)
            (root / "annotations-events-profusion/shhs1/shhs1-200001.xml").write_text(
                "<PSGAnnotation />"
            )

            with self.assertLogs("pyhealth.datasets.shhs", level="WARNING") as logs:
                rows = _metadata_rows(root)

            self.assertEqual([row["record_id"] for row in rows], ["shhs1-200001"])
            self.assertTrue(any("unexpected filename" in msg for msg in logs.output))

    def test_prepare_metadata_writes_configured_csv_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            dataset = SHHSDataset.__new__(SHHSDataset)
            dataset.root = str(root)
            dataset.prepare_metadata()

            csv_path = root / "shhs-pyhealth.csv"
            self.assertTrue(csv_path.is_file())
            frame = pd.read_csv(csv_path, dtype=str)
            self.assertEqual(
                list(frame.columns),
                ["patient_id", "record_id", "visit", "signal_file", "label_file"],
            )
            self.assertEqual(frame["patient_id"].tolist(), ["200001", "200001"])
            first_contents = csv_path.read_bytes()

            # An existing metadata file is preserved rather than silently
            # changing the indexed cohort on a later construction.
            self._touch(root / "edfs/shhs1/shhs1-200003.edf")
            dataset.prepare_metadata()
            self.assertEqual(csv_path.read_bytes(), first_contents)

    def test_empty_root_does_not_write_an_empty_metadata_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SHHSDataset.__new__(SHHSDataset)
            dataset.root = tmp
            with self.assertLogs("pyhealth.datasets.shhs", level="WARNING"):
                dataset.prepare_metadata()
            self.assertFalse((Path(tmp) / "shhs-pyhealth.csv").exists())


if __name__ == "__main__":
    unittest.main()
