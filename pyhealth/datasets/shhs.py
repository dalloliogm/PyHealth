"""Sleep Heart Health Study (SHHS) polysomnography dataset.

The dataset indexes SHHS Visit 1 and Visit 2 EDF recordings and their
Profusion sleep-stage annotations. It writes one metadata row per paired
recording; epoching and label interpretation are handled by the sleep-staging
task.
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

if TYPE_CHECKING:
    from pyhealth.tasks.sleep_staging_shhs import SleepStagingSHHS

logger = logging.getLogger(__name__)

_CSV_NAME = "shhs-pyhealth.csv"
_RECORDING_RE = re.compile(r"^(shhs[12])-(\d+)$", re.IGNORECASE)


def _annotation_candidates(root: Path, visit: str, stem: str) -> list[Path]:
    """Return annotation paths in the layouts used by NSRR SHHS releases.

    Most downloads place Visit 1 annotations in a ``shhs1`` directory and
    Visit 2 annotations in ``shhs2``. Some older copies put Visit 2 XML files
    under a directory named ``label`` instead. The filename may include the
    ``-profusion`` suffix or use the EDF stem directly.
    """
    annotation_root = root / "annotations-events-profusion"
    names = (f"{stem}-profusion.xml", f"{stem}.xml")
    folders = [visit]
    if visit == "shhs2":
        folders.append("label")
    candidates = [
        annotation_root / folder / name for folder in folders for name in names
    ]
    candidates.extend(annotation_root / name for name in names)
    return candidates


def _metadata_rows(root: Path) -> list[dict[str, str]]:
    """Build stable metadata rows for available SHHS EDF/XML pairs."""
    edf_root = root / "edfs"
    rows: list[dict[str, str]] = []

    for visit in ("shhs1", "shhs2"):
        visit_dir = edf_root / visit
        if not visit_dir.is_dir():
            logger.info("SHHS visit directory not found: %s", visit_dir)
            continue

        for edf_path in sorted(visit_dir.iterdir()):
            if not edf_path.is_file() or edf_path.suffix.lower() != ".edf":
                continue

            match = _RECORDING_RE.fullmatch(edf_path.stem)
            if match is None or match.group(1).lower() != visit:
                logger.warning(
                    "Skipping SHHS EDF with an unexpected filename for %s: %s",
                    visit,
                    edf_path.name,
                )
                continue

            candidates = _annotation_candidates(root, visit, edf_path.stem)
            annotation_path = next((path for path in candidates if path.is_file()), None)
            if annotation_path is None:
                logger.warning(
                    "Skipping SHHS EDF without a matching Profusion XML annotation: %s",
                    edf_path,
                )
                continue

            rows.append(
                {
                    # Visits from the same person must share a patient ID so
                    # patient-level splits cannot leak between visits.
                    "patient_id": match.group(2),
                    "record_id": edf_path.stem.lower(),
                    "visit": visit,
                    "signal_file": str(edf_path.resolve()),
                    "label_file": str(annotation_path.resolve()),
                }
            )

    rows.sort(key=lambda row: (row["patient_id"], row["visit"], row["record_id"]))
    return rows


class SHHSDataset(BaseDataset):
    """SHHS overnight polysomnography recordings and stage annotations.

    The root is the SHHS ``polysomnography`` directory containing
    ``edfs/shhs1``, ``edfs/shhs2`` and ``annotations-events-profusion``.
    Metadata generation pairs each EDF with its Profusion XML annotation and
    keeps Visit 1 and Visit 2 as separate records under the same participant.

    Metadata is written to ``shhs-pyhealth.csv`` in ``root`` or to the
    root-keyed PyHealth cache if the source directory is not writable.

    Args:
        root: SHHS ``polysomnography`` directory.
        dataset_name: Dataset name stored by :class:`BaseDataset`.
            Defaults to ``"shhs"``.
        config_path: Optional YAML config path. Defaults to
            ``configs/shhs.yaml``.
        **kwargs: Arguments forwarded to :class:`BaseDataset`, including
            ``dev``, ``refresh_cache`` and ``num_workers``.

    Examples:
        >>> from pyhealth.datasets import SHHSDataset
        >>> dataset = SHHSDataset(root="/data/shhs/polysomnography")  # doctest: +SKIP
        >>> sample_dataset = dataset.set_task()  # doctest: +SKIP
        >>> sample_dataset[0]["signal"].shape  # doctest: +SKIP
        (2, 3750)
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        self.root = str(Path(root).expanduser().resolve())
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "shhs.yaml"
        self.prepare_metadata()

        super().__init__(
            root=resolve_metadata_root(root, "shhs", _CSV_NAME),
            tables=["recordings"],
            dataset_name=dataset_name or "shhs",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Index paired EDF and Profusion XML files if metadata is absent."""
        root = Path(self.root).expanduser()
        if metadata_csv_exists(root, "shhs", _CSV_NAME):
            return

        rows = _metadata_rows(root)
        if not rows:
            logger.warning("No paired SHHS EDF/XML recordings found under %s", root)
            return

        metadata = pd.DataFrame(rows)
        write_metadata_csv(metadata, root, "shhs", _CSV_NAME)

    @property
    def default_task(self) -> "SleepStagingSHHS":
        """Return the 30-second SHHS sleep-staging task."""
        from pyhealth.tasks.sleep_staging_shhs import SleepStagingSHHS

        return SleepStagingSHHS()
