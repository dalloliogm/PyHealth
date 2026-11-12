"""Train a sleep-stage classifier on the Sleep Heart Health Study (SHHS).

The individual-level SHHS data are hosted by the National Sleep Research
Resource (NSRR). Request and receive approval for the SHHS Data Access and
Use Agreement (DAUA) before downloading the recordings. This example does not
download the data.

Download the SHHS polysomnography signals and Profusion staging annotations,
then point ``SHHS_ROOT`` (or ``--root``) at the directory containing the
``edfs/`` and ``annotations-events-profusion/`` folders. For example::

    SHHS_ROOT=/data/shhs/polysomnography \\
        python examples/eeg/sleep_staging/sleep_staging_shhs.py --dev

Omit ``--dev`` to use the full dataset. ``--dev`` applies PyHealth's
development-mode patient cap; it is useful for checking the pipeline, not for
reproducing a benchmark result.
"""

import argparse
import os

from pyhealth.datasets import PatientSplit, SHHSDataset, get_dataloader
from pyhealth.models import SparcNet
from pyhealth.tasks import SleepStagingSHHS
from pyhealth.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.environ.get("SHHS_ROOT"),
        help="SHHS polysomnography root (or set SHHS_ROOT)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--dev",
        action="store_true",
        help="limit the dataset for a development run",
    )
    args = parser.parse_args()
    if not args.root:
        parser.error(
            "set --root or SHHS_ROOT to your downloaded SHHS polysomnography "
            "directory; data access requires an approved NSRR DAUA"
        )

    dataset = SHHSDataset(root=args.root, dev=args.dev)
    train_dataset, val_dataset, test_dataset = dataset.set_task(
        SleepStagingSHHS(),
        split=PatientSplit(ratios=(0.7, 0.15, 0.15), seed=0),
    )
    print(
        "samples train/validation/test:",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = SparcNet(dataset=train_dataset)
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="accuracy",
    )
    print("Test metrics:", trainer.evaluate(test_loader))


if __name__ == "__main__":
    main()
