"""Followers an operand product absorbs into its output law (N2 design §2.5): a positive constant multiply or
divide (into ``M0``) and a ReLU (``lo = 0``), each the only reader of the product's output."""

from __future__ import annotations

import numbers
import operator
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.compute_modules import ComputeAdapter

SCALE, RELU = "scale", "relu"
_MULTIPLY = (operator.mul, torch.mul)
_DIVIDE = (operator.truediv, torch.div, torch.true_divide)
_RELU = (F.relu, torch.relu, torch.Tensor.relu)


@dataclass(frozen=True)
class Follower:
    """One absorbed follower: ``kind`` SCALE by its positive ``constant`` (a multiply, or a divide), or RELU;
    the float forward applies it as traced, the law takes ``factor``."""

    kind: str
    constant: float = 1.0
    divide: bool = False

    @property
    def factor(self) -> float:
        return 1.0 / self.constant if self.divide else self.constant

    def apply(self, y: torch.Tensor) -> torch.Tensor:
        if self.kind == RELU:
            return F.relu(y)
        return y / self.constant if self.divide else y * self.constant


def _static_positive(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    value = float(value)
    return value if value > 0.0 else None


def absorbable_follower(mapper) -> Optional[Follower]:
    """The follower ``mapper`` is, or None: a one-source host op on the canonical argument layout (the source, then
    one static literal) with no bound tensor and no keyword, or a ReLU module or call."""
    if not isinstance(mapper, ComputeOpMapper) or len(mapper.get_source_mappers()) != 1:
        return None
    if mapper.module_kwargs or mapper.output_index is not None or mapper.output_value_offset is not None:
        return None
    module = mapper.module
    if isinstance(module, nn.ReLU) and not module.inplace:
        return Follower(RELU)
    if not isinstance(module, ComputeAdapter) or module.arg_template is not None or module._bound_count:
        return None
    if module.fn in _RELU and not module.extra_args and module.kwargs in ({}, {"inplace": False}):
        return Follower(RELU)
    if module.kwargs:
        return None
    if len(module.extra_args) != 1:
        return None
    constant = _static_positive(module.extra_args[0])
    if constant is None:
        return None
    if module.fn in _MULTIPLY:
        return Follower(SCALE, constant)
    if module.fn in _DIVIDE:
        return Follower(SCALE, constant, divide=True)
    return None
