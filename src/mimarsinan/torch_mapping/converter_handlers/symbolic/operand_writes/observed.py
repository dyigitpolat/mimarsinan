"""What the traced program's run shows of its writes (review round 15): the tracer's shape propagation at batch 1,
recording for every node the state of each tensor it reads (version counter, shape, strides, offset, storage) before
and after it runs, and the operands its result shares storage with.

A tensor's version counter counts its in-place writes, and a view shares its base's counter, so a value whose state
at a read differs from its state when its node ran was changed in place in between (``stale``), through itself or
through a view of its storage; an operand whose state differs after a node ran was written by that node
(``written``). An inference tensor keeps no counter: its bytes stand in for it (review round 16). These are facts of
one run at batch 1, the backstop of the declared rule (``declared.py``): the probe's observation of hypothesis T3
(``value_parity/certificate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils import _pytree as pytree

from mimarsinan.mapping.support.compute_modules import write_counter
from mimarsinan.torch_mapping.conversion_probe import guarded

__all__ = ["WriteRecord", "WriteRecorder"]


def _tensors(value: Any) -> List[torch.Tensor]:
    return [leaf for leaf in pytree.tree_leaves(value) if isinstance(leaf, torch.Tensor)]


def _state(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """The version counter (an inference tensor's bytes, ``write_counter``: review round 16), shape, strides, offset
    and storage of ``tensor``."""
    counter, _ = guarded(lambda: write_counter(tensor))
    full, _ = guarded(lambda: (tuple(tensor.stride()), tensor.storage_offset(), tensor.untyped_storage()._cdata))
    return (counter, tuple(tensor.shape), *(full or ()))


def _states(value: Any) -> Tuple[Tuple[Any, ...], ...]:
    return tuple(_state(tensor) for tensor in _tensors(value))


_IS_ALIAS_OF = getattr(torch._C, "_is_alias_of")
"""Torch's own test that two tensors share storage."""


def _shares(result: Any, operand: Any) -> bool:
    """A tensor of ``result`` shares storage with a tensor of ``operand``."""
    pairs = [(a, b) for a in _tensors(result) for b in _tensors(operand)]
    return any(a is b or bool(guarded(lambda: _IS_ALIAS_OF(a, b))[0]) for a, b in pairs)


@dataclass
class WriteRecord:
    """One run: per node, the operands it wrote (``written``), the operands its result shares storage with
    (``shares``) and the operands its result is (``identical``); every read of a value changed since its node ran
    (``stale``: reader, operand); and the strides of every node's result that is one tensor (``strides``, review round
    16: the layout at which a call the schema may let alias its operand copied it)."""

    written: Dict[fx.Node, List[fx.Node]] = field(default_factory=dict)
    shares: Dict[fx.Node, List[fx.Node]] = field(default_factory=dict)
    identical: Dict[fx.Node, List[fx.Node]] = field(default_factory=dict)
    stale: List[Tuple[fx.Node, fx.Node]] = field(default_factory=list)
    strides: Dict[fx.Node, Tuple[int, ...]] = field(default_factory=dict)


class WriteRecorder(ShapeProp):
    """The tracer's shape propagation (``ShapeProp``), recording a ``WriteRecord``."""

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self.record = WriteRecord()
        self._defined: Dict[fx.Node, Tuple[Tuple[Any, ...], ...]] = {}

    def run_node(self, n: fx.Node) -> Any:
        operands = [(arg, self.env[arg]) for arg in n.all_input_nodes if arg in self.env]
        before = {arg: _states(value) for arg, value in operands}
        self.record.stale += [(n, arg) for arg, state in before.items() if state != self._defined.get(arg, state)]
        result = super().run_node(n)
        written = [arg for arg, value in operands if _states(value) != before[arg]]
        if written:
            self.record.written[n] = written
        shared = [arg for arg, value in operands if _shares(result, value)]
        if shared:
            self.record.shares[n] = shared
            self.record.identical[n] = [arg for arg, value in operands if value is result]
        self._defined[n] = _states(result)
        if isinstance(result, torch.Tensor):
            found, _ = guarded(lambda: tuple(result.stride()))
            if found is not None:
                self.record.strides[n] = found
        return result
