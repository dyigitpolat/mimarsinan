"""Constant-OFF floor declarations of activation nodes (D19): the one registry the constant-OFF bias clip reads.

A node has a constant-OFF floor when it outputs 0 for every pre-activation at or below 0; a spiking node has it when
a node whose every per-cycle input is at most 0 never fires. A channel whose largest pre-activation over its proven
input range is at most 0 is then constant-OFF, and bias below that bound is unobservable. A node class declares the
floor with ``declare_off_floor`` and a predicate on the instance (state such as a LIF node's window-start membrane can
decide it); a subclass reads its nearest declared base. An undeclared node has none: an affine package's
``nn.Identity``, GELU, LeakyReLU, or any module nobody declared, so an undeclared node is never clipped.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch.nn as nn

from mimarsinan.models.nn.activations.autograd import LeakyGradReLU
from mimarsinan.models.nn.activations.lif import LIFActivation
from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation

OffFloorPredicate = Callable[[nn.Module], bool]

_DECLARED: Dict[type, OffFloorPredicate] = {}


def _always(_node: nn.Module) -> bool:
    return True


def declare_off_floor(cls: type, predicate: OffFloorPredicate = _always) -> None:
    """Declare that instances of ``cls`` have the floor whenever ``predicate(instance)`` holds."""
    _DECLARED[cls] = predicate


def off_floor_declarations() -> Dict[type, OffFloorPredicate]:
    """A copy of the registry: every declared class and its predicate."""
    return dict(_DECLARED)


def has_off_floor(node: Optional[nn.Module]) -> bool:
    """Whether ``node`` declares a constant-OFF floor (the nearest declared class in its MRO decides)."""
    if node is None:
        return False
    for cls in type(node).__mro__:
        predicate = _DECLARED.get(cls)
        if predicate is not None:
            return bool(predicate(node))
    return False


def _lif_membrane_stays_below_threshold(node: nn.Module) -> bool:
    """An IF node fed at most 0 per cycle keeps its membrane at or below its window-start value, so it never fires
    while that value, snapped up to the armed membrane lattice (a half quantum), stays below the threshold. A node
    pickled before the window-start guard existed carries no ``membrane_init`` and starts at 0."""
    if_node = getattr(node, "if_node")
    lattice = getattr(if_node, "lattice_scale", None)
    start = float(node.__dict__.get("membrane_init", 0.0))
    reach = start + (0.5 / float(lattice) if lattice else 0.0)
    return reach < float(getattr(if_node, "v_threshold"))


declare_off_floor(nn.ReLU)
declare_off_floor(LeakyGradReLU)  # ReLU forward; the leak is in the backward only
declare_off_floor(LIFActivation, _lif_membrane_stays_below_threshold)
declare_off_floor(TTFSActivation)  # value mode: a staircase of clamp(relu(x) / theta); cycle mode never reaches 1
declare_off_floor(TTFSCycleActivation)  # a staircase of clamp(relu(x) / theta)
