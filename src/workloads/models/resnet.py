"""ResNet-18 with torchvision's pretrained ImageNet weights."""

from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18 as _resnet18

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def resnet18() -> nn.Module:
    """ResNet-18 in eval mode, downloading the weights on first use."""
    return _resnet18(weights=WEIGHTS).eval()


def preprocess() -> nn.Module:
    """The resize, crop and normalization the weights were trained with.

    Images must go through exactly these steps, or accuracy drops for reasons
    that have nothing to do with precision.
    """
    return WEIGHTS.transforms()
