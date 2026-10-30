"""Tests for the shared quantization helpers."""

import torch

from workloads.experiments.quantization import quantize


def test_int4_example() -> None:
    """The worked example in quantize's docstring."""
    x = torch.tensor([[0.7, -0.35, 0.1]])
    expected = torch.tensor([[0.7, -0.4, 0.1]])
    assert torch.allclose(quantize(x, 4, dim=1), expected)


def test_grid_is_symmetric() -> None:
    """Int4 never uses -8: every value lands on -7..7 times its scale."""
    x = torch.randn(16, 64)
    scale = x.abs().amax(dim=1, keepdim=True) / 7
    steps = quantize(x, 4, dim=1) / scale
    assert torch.allclose(steps, steps.round(), atol=1e-4)
    assert steps.abs().max() <= 7 + 1e-4


def test_all_zero_vector_stays_zero() -> None:
    """A zero row has scale ~0 and must not produce NaNs."""
    x = torch.zeros(2, 8)
    assert torch.equal(quantize(x, 8, dim=1), x)


def test_int8_error_is_small() -> None:
    """Int8 rounding moves each value by at most half a step."""
    x = torch.randn(32, 128)
    step = x.abs().amax(dim=1, keepdim=True) / 127
    assert ((quantize(x, 8, dim=1) - x).abs() <= step / 2 + 1e-6).all()
