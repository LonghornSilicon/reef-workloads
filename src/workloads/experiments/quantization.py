"""Simulated int8 and int4: which precisions exist and how they round.

Same conventions as reef's original ``workloads/`` (see story_quality):
values are rounded onto a symmetric integer grid (±127 for int8, ±7 for int4)
and multiplied back by their scale, so the arithmetic still runs in fp32.
That measures what rounding does to accuracy, not integer speed.

- int8 weights: one scale per output channel.
- int4 weights: one scale per group of ``INT4_GROUP`` inputs within each
  output channel. For a conv weight ``(out, in, kh, kw)`` the inputs are the
  ``in * kh * kw`` values each filter sees, the K dimension of the im2col
  matmul.
- Activations (``w*a8``, ``w*a4``): each Conv2d/Linear input is rounded right
  before the layer runs.
"""

import copy
from itertools import pairwise

import torch
from torch import Tensor, nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

PRECISIONS = ("fp32", "int8", "int4", "w8a8", "w4a8")
INT4_GROUP = 32


def quantize(x: Tensor, bits: int, dim: int) -> Tensor:
    """Symmetric rounding with one scale per vector along ``dim``.

    Example (int4): ``[0.7, -0.35, 0.1]`` has max 0.7, so scale = 0.7 / 7 =
    0.1, giving ``round([7, -3.5, 1]) = [7, -4, 1]`` and back ``[0.7, -0.4,
    0.1]``.

    Args:
        x: Values to round.
        bits: 8 or 4.
        dim: Axis the scale is shared along (``keepdim`` so it broadcasts).

    Returns:
        ``x`` rounded onto the grid, same shape and dtype.
    """
    limit = 2 ** (bits - 1) - 1
    amax = torch.clamp(
        x.abs().amax(dim=dim, keepdim=True), torch.finfo(x.dtype).tiny
    )
    scale = amax / limit
    scaled = x / scale
    scaled = torch.clamp(torch.round(scaled), -limit, limit)

    return scaled * scale


def fold_batchnorm(model: nn.Module) -> nn.Module:
    """A copy of ``model`` with every BatchNorm merged into the conv before it.

    At inference BatchNorm is ``y = a * x + b`` per channel, with
    ``a = gamma / sqrt(var + eps)`` and ``b = beta - mean * a``. Multiplying
    each output channel of the conv's weight by ``a`` (and its bias by ``a``,
    plus ``b``) gives the same output with no BatchNorm. Reef stores these
    folded weights, so they are the ones to quantize.

    Args:
        model: An eval-mode model whose Conv2d layers are each followed by a
            BatchNorm2d (e.g. ResNet-18).

    Returns:
        A folded copy; the BatchNorms are replaced with ``nn.Identity``.
    """
    model = copy.deepcopy(model).eval()
    for parent in model.modules():
        children = list(parent.named_children())
        for (conv_name, conv), (bn_name, bn) in pairwise(children):
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(parent, conv_name, fuse_conv_bn_eval(conv, bn))
                setattr(parent, bn_name, nn.Identity())
    if any(isinstance(m, nn.BatchNorm2d) for m in model.modules()):
        raise ValueError("a BatchNorm2d does not directly follow a Conv2d")
    return model


def convert(model: nn.Module, precision: str) -> nn.Module:
    """A copy of the fp32 ``model`` simulating ``precision``.

    Args:
        model: An fp32 model, BatchNorm already folded.
        precision: One of ``PRECISIONS``.

    Returns:
        A copy with every Conv2d and Linear weight rounded (and, for ``w*a8``,
        a forward pre-hook rounding each of their inputs to int8).
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}")
    model = copy.deepcopy(model)
    if precision == "fp32":
        return model
    weight_bits = 4 if precision in ("int4", "w4a8") else 8
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                layer.weight.copy_(_round_weight(layer.weight, weight_bits))
            if precision.startswith("w"):
                layer.register_forward_pre_hook(_round_input)
    return model


def _round_weight(weight: Tensor, bits: int) -> Tensor:
    """Round a Conv2d/Linear weight: per output channel, or int4 groups."""
    rows = weight.reshape(weight.shape[0], -1)  # (out, in * kh * kw)
    if bits == 8:
        return quantize(rows, 8, dim=1).reshape(weight.shape)
    # Zero-pad each row to a whole number of groups (the first conv has
    # 3 * 7 * 7 = 147 inputs); zeros never change a group's max.
    k = rows.shape[1]
    padded = nn.functional.pad(rows, (0, -k % INT4_GROUP))
    groups = padded.reshape(rows.shape[0], -1, INT4_GROUP)
    rounded = quantize(groups, 4, dim=2).reshape(rows.shape[0], -1)
    return rounded[:, :k].reshape(weight.shape)


def _round_input(layer: nn.Module, args: tuple[Tensor, ...]) -> tuple:
    """Round a layer's input to int8 just before it runs.

    One scale per position along dim 1: per pixel across channels for a
    conv input ``(n, c, h, w)``, per row for a Linear input ``(n, features)``,
    like reef's one scale per token.
    """
    return (quantize(args[0], 8, dim=1), *args[1:])
