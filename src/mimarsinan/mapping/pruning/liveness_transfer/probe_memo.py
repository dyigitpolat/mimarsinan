"""Run-scoped memoisation of ``derive_constant_outputs`` probe batteries.

The battery is pure in (arithmetic, op, bitwise input key): hosted modules are
never mutated by folds (folds land on core matrices and biases), eval-mode
toggles are restored, and RNG is forked. One memo shared across every fresh
context of a single analysis run (arms, final, replay) therefore removes every
repeated battery while remaining bit-identical by construction. Validity is
scoped to one run (fixed graph and transfer policy); [U-DAR] the key carries
the domain of the deployed arithmetic the battery ran in, so a battery of one
domain is never replayed under another's.
"""

from __future__ import annotations

import struct
from typing import Dict, MutableMapping, Sequence, Tuple

from mimarsinan.mapping.ir.deployment_dtype import DeployedArithmetic
from mimarsinan.mapping.pruning.liveness_transfer import constant_transfer

__all__ = ["ProbeMemo", "derive_constant_outputs_memoized", "probe_key_bytes"]

ProbeMemo = MutableMapping[Tuple[str, int, bytes], Dict[int, float]]

_TOP_MARK = b"\x00"
_CONST_MARK = b"\x01"


def probe_key_bytes(in_values: Sequence[float | None]) -> bytes:
    """Bitwise-exact key: TOP and ``-0.0``/``0.0`` never collide (float ``==``
    would merge the zeros and could replay the wrong battery)."""
    return b"".join(
        _TOP_MARK if v is None else _CONST_MARK + struct.pack("<d", v)
        for v in in_values
    )


def derive_constant_outputs_memoized(
    op, transfer, in_values: Sequence[float | None], memo: ProbeMemo | None, *,
    arithmetic: DeployedArithmetic, twins: "dict | None" = None,
) -> Dict[int, float]:
    """The battery in ``arithmetic``, replayed from ``memo`` when this exact key already ran.

    Refusals ({}) are cached too — a refused probe still paid the full
    battery, and the same inputs refuse again deterministically.
    """
    def run() -> Dict[int, float]:
        return constant_transfer.derive_constant_outputs(op, transfer, in_values, arithmetic=arithmetic, twins=twins)

    if memo is None:
        return run()
    key = (arithmetic.domain, op.id, probe_key_bytes(in_values))
    hit = memo.get(key)
    if hit is None:
        hit = run()
        memo[key] = hit
    return hit
