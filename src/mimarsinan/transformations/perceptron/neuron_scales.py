"""[U-PCH] A perceptron's weight scale per output neuron: the per-row peaks a per-neuron projection sizes its grids
from, and the one broadcast every reader of a per-neuron scale takes. [U-PFOLD r2] The output neuron's axis of a
weight, bias or code tensor is the one its reader declares for that view (``position_fold.output_axes``: a weight's
dim 0 and a bias's only dim per input, dim 1 of both per position), passed in, never a literal index. A
per-perceptron scale (0-d) passes through every helper unchanged, byte for byte."""

from __future__ import annotations

import torch

from mimarsinan.transformations.perceptron.position_fold import along_outputs, output_rows


def is_per_neuron_scale(scale) -> bool:
    """Whether ``scale`` holds one entry per output neuron (a 1-d tensor) rather than one per perceptron."""
    return isinstance(scale, torch.Tensor) and scale.dim() >= 1


def row_peaks(t: torch.Tensor, axis: int) -> torch.Tensor:
    """``max |t|`` per output neuron (on ``axis``, the reader's), the per-neuron peak a per-neuron grid divides
    ``q_max`` by."""
    return output_rows(t.abs(), axis).amax(dim=1) if t.dim() >= 1 else t.abs()


def scale_view(scale, like: torch.Tensor, axis: int):
    """``scale`` shaped to broadcast against ``like`` (a weight, bias or code tensor whose output neurons lie on
    ``axis``, the reader's): a per-neuron scale along that axis, a per-perceptron scale unchanged."""
    if not is_per_neuron_scale(scale) or like.dim() == 0:
        return scale
    if scale.numel() != like.shape[axis]:
        raise ValueError(f"a per-neuron scale of {scale.numel()} neurons read against {like.shape[axis]} output rows")
    return along_outputs(scale, axis, like.dim())
