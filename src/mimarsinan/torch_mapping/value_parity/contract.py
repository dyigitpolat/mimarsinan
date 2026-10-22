"""The function a converted model is contracted to equal: a named, declared transform of the native model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


def first_output(output: object) -> torch.Tensor:
    """The compared output: the tensor itself, or the first element of a tuple or list.

    The converter represents a tuple or list output by its first element
    (``MapperGraphConverter._handle_output``), so that is what is compared.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(
        f"the probed model returned {type(output).__name__}; the value probe "
        "compares a tensor or the first tensor of a tuple or list"
    )


def _no_declared_change(model: nn.Module) -> nn.Module:
    return model


@dataclass(frozen=True)
class FunctionContract:
    """``reference`` applies the declared change to the native witness; ``select_output`` picks the compared tensor."""

    name: str
    statement: str
    reference: Callable[[nn.Module], nn.Module] = _no_declared_change
    select_output: Callable[[object], torch.Tensor] = first_output


NATIVE_EVAL_IDENTITY = FunctionContract(
    name="native_eval_identity",
    statement=(
        "converted(x) == native.eval()(x) for every input of the declared shape: "
        "BatchNorm at its running statistics, dropout off, every absorbed "
        "activation at the native law and parameters; no conversion step declares "
        "a change of function; a tuple or list output is represented by its first "
        "element"
    ),
)
