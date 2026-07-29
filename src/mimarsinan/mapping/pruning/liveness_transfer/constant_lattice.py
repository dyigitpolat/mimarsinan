"""The constant lattice [W4b-2]: ``TOP > CONST(c)`` over structural lines.

A LINE is one structural signal of the mapping graph — a NeuralCore neuron
column output, a ComputeOp output port, or a wiring source kind. Its lattice
value is either

- ``TOP``  — live, value unknown (represented by ABSENCE from the map), or
- ``CONST(c)`` — the line carries exactly ``c`` on every timestep.

The order is ``TOP > CONST(c)`` with no bottom and no CONST-CONST joins: every
line has exactly ONE producer, so a line is never asked to merge two
different constants. Descent is one-way (a line moves TOP -> CONST at most
once and never changes value), which is why the graph fixpoint terminates:
the state space is bounded by (#lines) descents plus the (already monotone)
elimination sets. A re-derivation that disagrees with a recorded constant is a
BUG in a transfer rule, so it raises instead of silently widening.

The lattice bottoms out on the wiring itself, which is why the rules stay
uniform: an OFF axon reads ``0.0``, an ALWAYS-ON axon reads ``1.0``, and an
ELIMINATED producer column reads ``0.0`` (the executor invariant the whole
elimination framework already relies on). Model-input lines are ``TOP``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import AbstractSet, Dict, Mapping, Tuple

from mimarsinan.mapping.ir import IRSource

__all__ = [
    "ConstantLattice",
    "ConstantLatticeError",
    "OFF_SOURCE_CONSTANT",
    "ALWAYS_ON_SOURCE_CONSTANT",
    "Port",
    "source_constant",
]

Port = Tuple[int, int]

# An off-wired axon delivers nothing; an eliminated producer column is
# rewired to off, which is why elimination IS the CONST(0) case of this
# lattice rather than a separate mechanism.
OFF_SOURCE_CONSTANT = 0.0

# An always-on axon reads the executor's ``on_value`` every timestep; it is
# the crossbar encoding of a per-neuron bias (``chip_export`` folds such a row
# into ``hardware_bias`` verbatim), so CONST(1.0) is its exact lattice value.
ALWAYS_ON_SOURCE_CONSTANT = 1.0


class ConstantLatticeError(RuntimeError):
    """A transfer rule re-derived a line's constant with a different value."""


@dataclass
class ConstantLattice:
    """Monotone map of CONST lines; absence means TOP (live, unknown).

    ``admits_nonzero`` is the domain gate (see ``constant_policy``): when
    False the lattice only accepts CONST(0), which reproduces the W4b-1
    zero-only semantics exactly while still letting joins and opaque ops
    relay deadness.
    """

    admits_nonzero: bool = True
    values: Dict[Port, float] = field(default_factory=dict)

    def get(self, port: Port) -> float | None:
        """The line's constant, or None for TOP."""
        return self.values.get(port)

    def is_const(self, port: Port) -> bool:
        return port in self.values

    def descend(self, port: Port, value: float) -> bool:
        """Move ``port`` from TOP to CONST(value); True iff that was new.

        Refuses (stays TOP, returns False) for non-finite values and for
        non-zero constants outside the admitting domain. Raises when a
        recorded constant would change — descent must be monotone.
        """
        v = float(value)
        if not math.isfinite(v):
            return False
        if v != 0.0 and not self.admits_nonzero:
            return False
        recorded = self.values.get(port)
        if recorded is None:
            self.values[port] = v
            return True
        if recorded != v:
            raise ConstantLatticeError(
                f"constant lattice is not monotone at line {port}: recorded "
                f"CONST({recorded!r}) then re-derived CONST({v!r}). A transfer "
                "rule computed a line's value from state that had already "
                "been consumed; the fixpoint cannot be trusted."
            )
        return False

    def snapshot(self) -> Mapping[Port, float]:
        """An immutable-by-convention view for reporting and certification."""
        return dict(self.values)


def source_constant(
    src: object,
    *,
    lattice: ConstantLattice,
    pruned_cols: Mapping[int, AbstractSet[int]],
) -> float | None:
    """THE line-value query: what does this axon/op input read, or None (TOP)?

    One function serves every transfer rule (ComputeOp forward, NeuralCore row
    fold, NeuralCore column derivation), which is what keeps the rules uniform
    across op classes: they all ask the same question about their inputs.
    """
    if not isinstance(src, IRSource):
        return None
    if src.is_off():
        return OFF_SOURCE_CONSTANT
    if src.is_always_on():
        return ALWAYS_ON_SOURCE_CONSTANT
    if src.node_id < 0:
        return None  # model input (-2) and any future wiring kind: TOP
    port = (src.node_id, src.index)
    recorded = lattice.get(port)
    if recorded is not None:
        # A RECORDED constant outranks "eliminated => 0". A CONST(c != 0)
        # producer column is only ever eliminated once every reader has folded
        # c away (orphaning needs all readers dead, and a carrier-backed
        # column cannot starve), so the recorded value is exactly what the
        # consumers that mattered were compiled against. Letting the zero win
        # here would flip the line's value under the fixpoint and trip the
        # monotonicity guard on a perfectly sound fold.
        return recorded
    if src.index in pruned_cols.get(src.node_id, frozenset()):
        return OFF_SOURCE_CONSTANT
    return None
