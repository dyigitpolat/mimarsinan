"""Batch layouts: static extents proved by the symbolic-batch pass in the affine batch domain; layout violations
observed or proved.

A static extent is a proof, never a sample, and only a PURE program has one (``symbolic/purity.py``: declared
operators, declared value types; ``impurity`` names the first impure node of any other, and the symbolic pass never
runs for it). The pass computes every value of a pure program in the affine batch domain (``symbolic/symbolic_batch``,
review round 14): an axis is static where its extent is a batch-free integer the domain computed, the value is not
taken from the seeded traces nor computed from an opaque batch value (``SymbolicPass.unproved``), no data-dependent
size lies upstream, and a seeded batch-1 trace agrees. A violation (a value not carrying the batch on axis 0 alone) is
a fact at one batch: it is observed in seeded concrete traces at batches 1, 2 and 3 (``symbolic/trace_values.py``), or
proved for every batch. A value an opaque batch value (a decision) reaches is named by ``batch_decision``, a
per-sample shape it reaches by ``shape_decision``, a reshape axis 0 it reaches by ``batch_axis_decision``. No random
state is read.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch.fx as fx

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import BATCH, Affine, Opaque, describe
from mimarsinan.torch_mapping.converter_handlers.symbolic.pass_result import DATA_DEPENDENT
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import Impurity, ProgramPurity, program_purity
from mimarsinan.torch_mapping.converter_handlers.symbolic.symbolic_batch import (
    SYMBOLIC_HINT,
    Decision,
    Extent,
    SymbolicPass,
    SymbolicShape,
    symbolic_shapes,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.trace_values import (
    TRACE_BATCHES,
    Trace,
    trace,
    trace_shapes,
)

__all__ = ["BATCH_SIZE", "BatchLayouts", "Decision", "SYMBOLIC_HINT", "TRACE_BATCHES", "trace_shapes"]


class _BatchSize:
    """The marker of an unproven extent (the batch size, or any extent no proof makes static); never evaluated.

    A layout marks every unproven axis with this one object, so it says only "not static": ``is_static_tree``
    (``shape_value_mixin.py``) keeps every tree that holds it away from Python code at conversion time, and its value
    is a host op's at run time. Observing it directly (arithmetic, comparison, truth, coercion, hashing) raises
    ``TypeError`` as a tripwire, not as the mechanism: containers compare their entries by identity first, so
    ``(BATCH_SIZE, 6) == (BATCH_SIZE, 6)`` holds without asking it. Identity survives copying and pickling.
    """

    def _opaque(self, *_args):
        raise TypeError("the batch size is unknown at conversion time")

    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _opaque
    __bool__ = __index__ = __int__ = __float__ = __complex__ = __hash__ = _opaque

    def __repr__(self) -> str:
        return "B"

    def __copy__(self) -> "_BatchSize":
        return self

    def __deepcopy__(self, _memo) -> "_BatchSize":
        return self

    def __reduce__(self) -> str:
        return "BATCH_SIZE"


BATCH_SIZE = _BatchSize()

def _is_batch(extent: Extent) -> bool:
    return extent == BATCH


def _describe(dims: Tuple[Extent, ...]) -> str:
    """``(B, 6)`` style: a form as ``2*B + 1``, a data-dependent size by name, an opaque extent as ``?``."""
    shown = [d.reason if isinstance(d, Opaque) and d.origin is DATA_DEPENDENT else "?" if isinstance(d, Opaque)
             else describe(d) for d in dims]
    data = sorted({d.reason for d in dims if isinstance(d, Opaque) and d.origin is DATA_DEPENDENT})
    sizes = f", {', '.join(data)} a data-dependent size{'s' if len(data) > 1 else ''}" if data else ""
    return f"({', '.join(shown)}{',' if len(shown) == 1 else ''}) for every batch B >= 1{sizes}"


class BatchLayouts:
    """The batch layout of every traced value of ``gm``; the symbolic pass and the traces run once, on first use.
    ``purity``: the traced program's (``program_purity`` of ``gm`` when not given)."""

    def __init__(self, gm: fx.GraphModule, purity: Optional[ProgramPurity] = None):
        self._gm = gm
        self._purity = purity
        self._proof: Optional[SymbolicPass] = None
        self._traces: Optional[Tuple[Trace, ...]] = None

    @property
    def purity(self) -> ProgramPurity:
        if self._purity is None:
            self._purity = program_purity(self._gm)
        return self._purity

    @property
    def impurity(self) -> Optional[Impurity]:
        """The first node that makes the program impure; ``None`` for a pure program."""
        return self.purity.impurity

    def _traced(self) -> Tuple[Trace, ...]:
        if self._traces is None:
            self._traces = tuple(trace(self._gm, b) for b in TRACE_BATCHES)
        return self._traces

    def _pass(self) -> SymbolicPass:
        """The symbolic pass of a pure program; an impure one's is empty (nothing proved, nothing decided)."""
        if self._proof is None:
            self._proof = symbolic_shapes(self._gm, self._traced()) if self.purity.pure else SymbolicPass(None)
        return self._proof

    def _symbolic(self, node: fx.Node) -> Optional[SymbolicShape]:
        return self._pass().shapes.get(node)

    def batch_decision(self, node: fx.Node) -> Optional[Decision]:
        """The first batch decision ``node`` or an input is computed from (``SymbolicPass.batch_decision``)."""
        return self._pass().batch_decision(node)

    def shape_decision(self, node: fx.Node) -> Optional[Decision]:
        """The decision ``node``'s per-sample shape depends on (``SymbolicPass.shape_decision``)."""
        return self._pass().shape_decision(node)

    def batch_axis_decision(self, node: fx.Node) -> Optional[Decision]:
        """The decision a pure reshape's axis 0 depends on (``SymbolicPass.batch_axis_decision``)."""
        return self._pass().batch_axis_decision(node)

    def proof_statement(self) -> Optional[str]:
        """What the program's shapes are proved by: an impure program's statement, else the symbolic pass's
        (``SymbolicPass.proof_statement``); ``None`` when no shape was asked for, so the pass never ran."""
        if self.impurity is not None:
            return self.impurity.statement()
        return None if self._proof is None else self._proof.proof_statement()

    def opaque(self) -> Optional[Decision]:
        """The first opaque batch value of a pure program (traced order; the pass runs if it has not), ``None`` when
        there is none or the program is impure."""
        return self._pass().opaque if self.purity.pure else None

    def _observed(self, node: fx.Node) -> Tuple[Optional[Tuple[int, ...]], ...]:
        return tuple(record.shapes.get(node) for record in self._traced())

    def layout(self, node: fx.Node) -> Optional[Tuple[Any, ...]]:
        """Per axis: the proved static extent, or the unproven-extent marker ``BATCH_SIZE``; ``None`` when no proof
        covers the value. The rank is proved with the layout. No node reads a written operand after the write (the
        tracer's ``operand_writes``: it reads the write's result), so a layout is never a pre-write shape (review round
        15: ``z.shape`` after ``torch.resize_as_(z, y)`` was ``z``'s node's layout). A static extent agrees with every
        seeded trace that shows the value (review round 16: with the batch-1 trace alone, an extent agreeing at batch 1
        and at the hint was static)."""
        symbolic = self._symbolic(node)
        observed = self._observed(node)
        one = observed[0]
        if symbolic is None or not self._pass().proved(node) or one is None or len(one) != len(symbolic.dims):
            return None
        traced = [shape for shape in observed if shape is not None and len(shape) == len(one)]
        return tuple(d if type(d) is int and all(shape[axis] == d for shape in traced) else BATCH_SIZE
                     for axis, d in enumerate(symbolic.dims))

    def extents(self, node: fx.Node) -> Optional[Tuple[Any, ...]]:
        """Per axis the domain's proved extent of ``node``, an ``int`` or an affine form ``a*B + c`` (coverage W7: the
        batch-merge push-forward reads a merged axis's ``a``); ``None`` when no proof covers the value, an extent is
        opaque, or a seeded trace that shows the value disagrees with it."""
        symbolic = self._symbolic(node)
        if symbolic is None or not self._pass().proved(node) or any(isinstance(d, Opaque) for d in symbolic.dims):
            return None
        dims = tuple(symbolic.dims)
        for batch, shape in zip(TRACE_BATCHES, self._observed(node)):
            if shape is not None and (len(shape) != len(dims) or any(
                    (d.at(batch) if isinstance(d, Affine) else d) != n for d, n in zip(dims, shape))):
                return None
        return dims

    def violation(self, node: fx.Node) -> Optional[str]:
        """Why ``node``'s value does not carry the batch on axis 0 alone; ``None`` when nothing shows it."""
        return self._observed_violation(node) or self._proved_violation(node)

    def _observed_violation(self, node: fx.Node) -> Optional[str]:
        found = self._observed(node)
        shapes = [shape for shape in found if shape is not None]
        if len(shapes) < len(found):
            return None
        observed = dict(zip(TRACE_BATCHES, shapes))
        described = ", ".join(f"{s} at batch {b}" for b, s in observed.items())
        if len({len(s) for s in observed.values()}) > 1:
            return f"the rank changes with the batch size ({described})"
        if any(not s or s[0] != b for b, s in observed.items()):
            return f"axis 0 is not the batch axis ({described})"
        if any(len(set(dims)) > 1 for dims in zip(*(s[1:] for s in observed.values()))):
            return f"an axis other than axis 0 carries the batch ({described})"
        return None

    def _proved_violation(self, node: fx.Node) -> Optional[str]:
        """A shape the domain computed that does not carry the batch on axis 0 alone (a data-dependent size counts: it
        is a violation for every input whose size differs from the batch). A shape an opaque batch value reaches is
        that decision's to refuse (``shape_decision``), not a violation."""
        symbolic = self._symbolic(node)
        if symbolic is None or self.shape_decision(node) is not None or not self._pass().static_proof:
            return None
        dims = symbolic.dims
        if not dims or not _is_batch(dims[0]):
            return f"axis 0 is not the batch axis ({_describe(dims)})"
        if any(type(d) is not int for d in dims[1:]):
            return f"an axis other than axis 0 is not static ({_describe(dims)})"
        return None
