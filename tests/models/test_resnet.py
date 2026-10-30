"""Tests for the ResNet-18 loader."""

import pytest
import torch

from workloads.models.resnet import resnet18


@pytest.mark.slow
def test_output_shape() -> None:
    """One 224x224 image gives 1000 ImageNet logits (downloads weights)."""
    with torch.no_grad():
        logits = resnet18()(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 1000)
