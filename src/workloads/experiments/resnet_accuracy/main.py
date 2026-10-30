"""How much top-1 accuracy ResNet-18 keeps at each simulated precision."""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from workloads.experiments.imagenette import validation_loader
from workloads.experiments.quantization import (
    PRECISIONS,
    convert,
    fold_batchnorm,
)
from workloads.models.resnet import preprocess, resnet18

RESULTS = Path(__file__).resolve().parents[4] / "results" / "resnet_accuracy"


def device() -> torch.device:
    """Apple-silicon GPU if available, else CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def top1(model: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    """Fraction of images whose highest-scoring class is the right one."""
    model.to(dev).eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(dev), labels.to(dev)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return correct / total


def main() -> None:
    """Score ResNet-18 at each precision and write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        action="append",
        choices=PRECISIONS,
        help="precision to run (repeatable; default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    precisions = args.precision or list(PRECISIONS)

    dev = device()
    loader = validation_loader(preprocess(), batch_size=args.batch_size)
    folded = fold_batchnorm(resnet18())

    rows = []
    for precision in precisions:
        accuracy = top1(convert(folded, precision), loader, dev)
        rows.append({"precision": precision, "top1": f"{accuracy:.4f}"})
        print(f"{precision:>5}: {accuracy:.2%}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "resnet18.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["precision", "top1"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
