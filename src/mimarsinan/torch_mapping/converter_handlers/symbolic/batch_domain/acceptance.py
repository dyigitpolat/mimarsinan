"""The kernel acceptance rule: which output extents of a tensor kernel the symbolic pass accepts (review round 14).

A kernel's output extent is accepted only when its declared batch rule states it in the affine batch domain
(``kernels.rule_of``) and torch's own runs agree with the stated extent at every batch they ran at, two distinct batches
at least: the FakeTensor run at the hint batch and the seeded traces at batches 1, 2 and 3 (review round 16), or, where
the FakeTensor run raised, the seeded traces alone, whose extents the rule never reads (review round 15: each is
``Opaque`` to it; they only cross-check). Every other extent is ``Opaque``: one no rule states, one the rule states as
opaque, one torch's runs contradict or checked at one batch only. Two affine forms of the batch that agree at two
distinct batches are equal, so a stated extent of the wrong form (a constant where torch's is ``B``) is never
accepted. [corrected in review round 16: where the FakeTensor run returned, it was the only cross-check, and a stated 7
where torch's extent is ``B`` agreed with it at the hint batch 7] A data-dependent extent of torch's run
(an unbacked size: a Boolean mask's count, ``nonzero``'s rows) is never accepted either; it is ``Opaque`` with the
origin ``DATA_DEPENDENT``, never folded and no batch decision.

Soundness lemma (kernel acceptance). If every operand extent and scalar of a declared kernel has its domain value's
meaning at every batch B >= 1, every accepted extent's meaning is the kernel's native extent at every B >= 1 at which
the kernel returns: an accepted extent is the rule's value, which the kernels' lemma (``kernels.py``) makes native. The
cross-check against torch's runs (two batches at least) adds no fold: it only makes extents opaque, so a rule that
misstated torch's affine shape function as another affine form is caught (two such forms agree at two batches only
when they are equal), and an extent no rule states is never torch's symbolic value. A tensor the seeded traces show as
no native value (``None`` at that leaf at every traced batch, an empty tensor in torch's FakeTensor run) is read by no
node of the traced program, and keeps the hint-only check. [restated in review round 16]
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.forms import Opaque, meaning
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.kernel_rules import DATA_DEPENDENT as STATED_DATA
from mimarsinan.torch_mapping.converter_handlers.symbolic.pass_result import DATA_DEPENDENT, Decision

__all__ = ["accept", "agrees", "data_dependent", "hint_int", "unbacked"]


def unbacked(extent: Any) -> bool:
    """A data-dependent size of torch's run (an unbacked symbol)."""
    return isinstance(extent, torch.SymInt) and bool(free_unbacked_symbols(extent))


def hint_int(extent: Any) -> int:
    """An extent's value at the hint batch (a data-dependent one, which has none, as 1: torch-side only)."""
    if isinstance(extent, torch.SymInt):
        return 1 if unbacked(extent) else int(extent.node.require_hint())
    return int(extent)


def agrees(stated: Any, extents: Mapping[int, Any], least: int = 2) -> bool:
    """The stated extent is torch's at every batch it ran at (``extents``: batch -> torch's extent), ``least`` batches
    at least (two, review round 16); an unbacked size agrees with none."""
    if isinstance(stated, Opaque):
        return True
    if len(extents) < least or any(unbacked(extent) for extent in extents.values()):
        return False
    return all(hint_int(extent) == meaning(stated, batch) for batch, extent in extents.items())


def data_dependent(extent: Any) -> Opaque:
    """A data-dependent size of torch's run as a domain value: never folded, no batch decision."""
    return Opaque(str(extent.node.expr), DATA_DEPENDENT)


def accept(operator_name: str, stated: Any, extents: Tuple[Any, ...], observed: Mapping[int, Tuple[Any, ...]],
           inherited: Optional[Decision], no_native: bool = False) -> Tuple[Any, ...]:
    """The domain extents of one output tensor of a kernel: the stated ones (``stated``: the rule's tuple, or anything
    else when the rule states none) torch's runs agree with (``extents``: torch's at the hint; ``observed``: batch ->
    torch's extents, two batches at least; one for a tensor the seeded traces show as no tensor, ``no_native``: no
    node of the traced program reads it, natively ``None``), else ``Opaque``. A size no rule states carries the
    decision of the kernel's first opaque operand (``inherited``); every other new ``Opaque`` has no origin yet: the
    pass gives it the kernel's own decision."""
    if not isinstance(stated, tuple) or len(stated) != len(extents):
        reason = f"a size of {operator_name} that no declared batch rule states"
        return tuple(data_dependent(got) if unbacked(got) else Opaque(reason, inherited) for got in extents)
    dims = []
    for axis, (want, got) in enumerate(zip(stated, extents)):
        if want is STATED_DATA:
            dims.append(data_dependent(got) if unbacked(got) else Opaque(
                f"a size of {operator_name} torch does not compute as data-dependent"))
        elif isinstance(want, Opaque):
            dims.append(want)
        elif agrees(want, {batch: shape[axis] for batch, shape in observed.items()}, 1 if no_native else 2):
            dims.append(want)
        elif unbacked(got):
            dims.append(data_dependent(got))
        else:
            dims.append(Opaque(f"a size of {operator_name} on which its declared batch rule and torch's runs differ, "
                               "or that torch's runs checked at one batch only"))
    return tuple(dims)
