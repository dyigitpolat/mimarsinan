"""BatchNorm lowering SSOT: which BatchNorm folds into its weight op (the predicates), the Perceptron normalization
slot of a folded BN, else one exact host affine op."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.models.nn.layers import ChannelsLastBatchNorm1d
from mimarsinan.torch_mapping.fx_shape_utils import node_output_shape, strip_batch

# The BatchNorms the converter lowers: folded into a weight op, else an exact host affine op.
BATCHNORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)


def has_running_statistics(bn: nn.Module) -> bool:
    """Eval-mode BatchNorm is a fixed affine op only when it normalizes with running statistics."""
    return getattr(bn, "running_mean", None) is not None and getattr(bn, "running_var", None) is not None


def batchnorm_channel_axis(bn: nn.Module, rank: int) -> int:
    """The non-negative axis a BatchNorm normalizes in a rank-``rank`` tensor."""
    return rank - 1 if isinstance(bn, ChannelsLastBatchNorm1d) else 1


def batchnorm_folds_into(bn: nn.Module, weight: nn.Module, rank: Optional[int]) -> bool:
    """Whether ``bn`` is a per-output-channel affine of ``weight``'s rank-``rank`` output (an exact fold)."""
    if not isinstance(bn, BATCHNORM_TYPES) or not has_running_statistics(bn) or rank is None or rank < 2:
        return False
    out_axis = rank - 1 if isinstance(weight, nn.Linear) else 1
    out_channels = weight.out_features if isinstance(weight, nn.Linear) else getattr(weight, "out_channels", None)
    return batchnorm_channel_axis(bn, rank) == out_axis and bn.num_features == out_channels


class BatchNormAffine(nn.Module):
    """Eval-mode BatchNorm as the per-channel affine ``u * x + c`` (``u = gamma / sqrt(var + eps)``, ``c = beta - u * mean``).

    Mode-independent and frozen (buffers only); ``channel_axis`` is 1, or -1 for a channels-last BatchNorm.
    """

    scale: torch.Tensor
    shift: torch.Tensor

    def __init__(self, scale: torch.Tensor, shift: torch.Tensor, channel_axis: int = 1):
        super().__init__()
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("shift", shift.detach().clone())
        self.channel_axis = int(channel_axis)

    @classmethod
    def from_batchnorm(cls, bn: nn.Module) -> "BatchNormAffine":
        mean, var = getattr(bn, "running_mean", None), getattr(bn, "running_var", None)
        if not isinstance(bn, BATCHNORM_TYPES) or mean is None or var is None:
            raise ValueError(
                f"BatchNormAffine needs a BatchNorm1d/2d with running statistics; got {type(bn).__name__} "
                f"(track_running_stats={getattr(bn, 'track_running_stats', None)})"
            )
        mean, var = mean.detach(), var.detach()
        gamma = bn.weight.detach() if bn.weight is not None else torch.ones_like(var)
        beta = bn.bias.detach() if bn.bias is not None else torch.zeros_like(mean)
        scale = gamma / torch.sqrt(var + bn.eps)
        return cls(scale, beta - scale * mean, -1 if isinstance(bn, ChannelsLastBatchNorm1d) else 1)

    def extra_repr(self) -> str:
        return f"channels={self.scale.numel()}, channel_axis={self.channel_axis}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        view = [1] * x.dim()
        view[self.channel_axis] = -1
        return x * self.scale.view(view) + self.shift.view(view)


def batchnorm_host_op(node: fx.Node, bn: nn.Module, source) -> ComputeOpMapper:
    """A BatchNorm that folds into no weight op: one host ComputeOp computing it exactly."""
    shape = strip_batch(node_output_shape(node))
    return ComputeOpMapper(
        source, BatchNormAffine.from_batchnorm(bn),
        input_shape=shape, output_shape=shape, name=node.name,
    )


def copy_batchnorm_state(dst_bn: nn.Module, src_bn: nn.Module) -> None:
    """Copy affine params and running stats between BatchNorms (possibly 2d src -> 1d dst).

    The destination follows the source's dtype first, so converting an fp64
    (or fp16) model never silently downcasts through a fresh fp32 module.
    """
    if not isinstance(dst_bn, BATCHNORM_TYPES) or not isinstance(src_bn, BATCHNORM_TYPES):
        return
    if src_bn.running_mean is not None:
        dst_bn.to(src_bn.running_mean.dtype)
    if src_bn.weight is not None and dst_bn.weight is not None:
        dst_bn.weight.data.copy_(src_bn.weight.data)
    if src_bn.bias is not None and dst_bn.bias is not None:
        dst_bn.bias.data.copy_(src_bn.bias.data)
    if src_bn.running_mean is not None and dst_bn.running_mean is not None:
        dst_bn.running_mean.copy_(src_bn.running_mean)
    if src_bn.running_var is not None and dst_bn.running_var is not None:
        dst_bn.running_var.copy_(src_bn.running_var)
    if src_bn.num_batches_tracked is not None and dst_bn.num_batches_tracked is not None:
        dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)


def perceptron_normalization(bn: nn.Module) -> nn.BatchNorm1d:
    """The Perceptron normalization slot of a BatchNorm folded into its weight op, carrying eps and momentum.

    The slot is an affine BatchNorm1d of the source's 1-d class (channels-last kept; a BatchNorm2d becomes a
    BatchNorm1d the conv mappers apply to the (B, C, H*W) view), with weight 1 and bias 0 when the source
    has none. An affine BatchNorm1d is deep-copied, which builds that same slot.
    """
    if isinstance(bn, nn.BatchNorm1d) and bn.affine:
        return copy.deepcopy(bn)
    if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
        raise TypeError(f"perceptron_normalization folds a BatchNorm1d/2d; got {type(bn).__name__}")
    slot_class = type(bn) if isinstance(bn, nn.BatchNorm1d) else nn.BatchNorm1d
    slot = slot_class(bn.num_features, eps=bn.eps, momentum=bn.momentum)
    copy_batchnorm_state(slot, bn)
    return slot
