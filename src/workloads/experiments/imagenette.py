"""Imagenette, a 10-class subset of ImageNet, for classification experiments.

Imagenette keeps ImageNet's folder names (WordNet IDs such as ``n01440764``),
so a model trained on all 1000 ImageNet classes can be scored on it directly:
each folder maps to that class's index among ImageNet's 1000 outputs.
"""

from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[3] / "data" / "imagenette2-160"

# WordNet ID of each Imagenette folder -> its index among ImageNet's 1000
# classes (the order torchvision's pretrained models use).
IMAGENET_INDEX = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def validation_loader(
    transform: nn.Module, batch_size: int = 64, root: Path = ROOT
) -> DataLoader:
    """Imagenette's validation images, labeled with ImageNet class indices.

    Args:
        transform: Preprocessing to apply to each image, e.g.
            ``workloads.models.resnet.preprocess()``.
        batch_size: Images per batch.
        root: The unpacked ``imagenette2-160`` directory.

    Returns:
        Batches of ``(images, labels)`` where each label is an ImageNet index
        (0-999), so it can be compared with the model's argmax directly.
    """
    dataset = ImageFolder(root / "val", transform=transform)
    # ImageFolder numbers the folders 0-9; swap in the ImageNet index.
    dataset.target_transform = lambda label: IMAGENET_INDEX[
        dataset.classes[label]
    ]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
