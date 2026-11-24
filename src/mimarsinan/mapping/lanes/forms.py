"""Lane forms: the host-op payloads a lane realizes (ReLU as the clamp, a two-tensor add, an average pool), and the
constant-zero pad, which executed lanes take as on-chip wiring (``plan.py``).

Each form is a subset of a requirement-table kind (``mapping/capability/op_kinds.py``): a payload this module does
not recognize keeps its op on the host, where it is exact, and the census prints it there.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.models.nn.activations.lane_quantization import LaneAddModule, LanePoolModule, LaneReduceModule

_RELU_FUNCTIONS = (F.relu, torch.relu)
_ADD_FUNCTIONS = (operator.add, torch.add)
#: ``F.pad`` and the padding builtin the converter records (``torch._C._nn.pad``, which ``F.pad`` wraps).
_PAD_FUNCTIONS = (F.pad, getattr(getattr(torch._C, "_nn"), "pad"))
_PAD_ARGUMENTS = ("pad", "mode", "value")


def unwrap_lane(module: Optional[nn.Module]) -> Optional[nn.Module]:
    """The host payload under a lane end the Readout Law step installed."""
    if isinstance(module, LaneAddModule):
        return module.add
    if isinstance(module, (LanePoolModule, LaneReduceModule)):
        return module.pool
    return module


def _plain_adapter(module, functions) -> bool:
    return (isinstance(module, ComputeAdapter) and module.fn in functions
            and getattr(module, "_bound_count", 0) == 0 and not module.extra_args and not module.kwargs)


def is_clamp_form(module: Optional[nn.Module]) -> bool:
    """ReLU: the clamp law with ``lo = 0`` on an unsigned grid (class L)."""
    module = unwrap_lane(module)
    return type(module) is nn.ReLU or _plain_adapter(module, _RELU_FUNCTIONS)


def is_add_form(module: Optional[nn.Module], input_shapes: Sequence[Optional[Tuple[int, ...]]]) -> bool:
    """A two-tensor add of equal shapes, no broadcast and no scale (class E)."""
    shapes = [tuple(s) for s in input_shapes if s is not None]
    return (_plain_adapter(unwrap_lane(module), _ADD_FUNCTIONS) and len(input_shapes) == 2 and len(shapes) == 2
            and shapes[0] == shapes[1])


@dataclass(frozen=True)
class PadForm:
    """A constant-zero pad of a per-sample input: per dimension, the elements added before and after (a negative
    amount crops). Wiring: each output element is an input element or the constant 0."""

    widths: Tuple[Tuple[int, int], ...]

    def output_shape(self, input_shape: Sequence[int]) -> Tuple[int, ...]:
        return tuple(int(d) + b + a for d, (b, a) in zip(input_shape, self.widths))


def zero_pad_form(module: Optional[nn.Module], input_shape: Optional[Sequence[int]]) -> Optional[PadForm]:
    """The form of a constant-zero pad over a per-sample input of ``input_shape`` (``F.pad`` or its builtin, mode
    constant, value 0, integer amounts on the per-sample dimensions only), or ``None``."""
    if not isinstance(module, ComputeAdapter) or module.fn not in _PAD_FUNCTIONS or input_shape is None:
        return None
    if getattr(module, "_bound_count", 0) or module.arg_template is not None or len(module.extra_args) > 3:
        return None
    call = dict(zip(_PAD_ARGUMENTS, module.extra_args))
    if set(module.kwargs) - set(_PAD_ARGUMENTS) or set(call) & set(module.kwargs):
        return None
    call.update(module.kwargs)
    pad, mode, value = call.get("pad"), call.get("mode", "constant"), call.get("value")
    if mode not in (None, "constant") or isinstance(value, bool) or not (value is None or value == 0):
        return None
    if (not isinstance(pad, (tuple, list)) or not pad or len(pad) % 2
            or not all(isinstance(p, int) and not isinstance(p, bool) for p in pad)):
        return None
    shape = tuple(int(d) for d in input_shape)
    if len(pad) // 2 > len(shape):  # the pad would reach the batch dimension
        return None
    widths = [(0, 0)] * len(shape)
    for k in range(len(pad) // 2):  # F.pad's amounts start at the last dimension
        widths[len(shape) - 1 - k] = (int(pad[2 * k]), int(pad[2 * k + 1]))
    form = PadForm(tuple(widths))
    return form if all(d > 0 for d in form.output_shape(shape)) else None


@dataclass(frozen=True)
class PoolForm:
    """A 2-D sum window of ``kernel`` at ``stride`` over zero padding ``padding``, divided by the window size."""

    kernel: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]

    @property
    def divisor(self) -> int:
        return int(self.kernel[0]) * int(self.kernel[1])


def _pair(value) -> Tuple[int, int]:
    return (int(value[0]), int(value[1])) if isinstance(value, (tuple, list)) else (int(value), int(value))


def _adaptive_size(module) -> Optional[Tuple[int, int]]:
    size = module.output_size
    if size is None or (isinstance(size, (tuple, list)) and None in size):
        return None
    return _pair(size)


def is_pool_form(module: Optional[nn.Module]) -> bool:
    """An average pool whose divisor is the window size everywhere (class P): an adaptive pool to a fixed size,
    or a fixed window without ceil mode, without excluded padding and without a divisor override."""
    module = unwrap_lane(module)
    if type(module) is nn.AdaptiveAvgPool2d:
        return _adaptive_size(module) is not None
    if type(module) is nn.AvgPool2d:
        padding = _pair(module.padding)
        return not (module.ceil_mode or module.divisor_override is not None
                    or (any(padding) and not module.count_include_pad))
    return False


def pool_form(module: Optional[nn.Module], input_shape: Optional[Tuple[int, ...]]) -> Optional[PoolForm]:
    """The window of a pool form over a ``(C, H, W)`` input, or ``None`` (not a pool form, or an adaptive size
    that does not divide the input)."""
    module = unwrap_lane(module)
    if not is_pool_form(module) or input_shape is None or len(input_shape) != 3:
        return None
    height, width = int(input_shape[1]), int(input_shape[2])
    if type(module) is nn.AdaptiveAvgPool2d:
        out = _adaptive_size(module)
        if out is None or height % out[0] or width % out[1]:
            return None
        kernel = (height // out[0], width // out[1])
        return PoolForm(kernel, kernel, (0, 0))
    assert isinstance(module, nn.AvgPool2d)
    kernel = _pair(module.kernel_size)
    stride = _pair(module.stride if module.stride is not None else module.kernel_size)
    return PoolForm(kernel, stride, _pair(module.padding))
