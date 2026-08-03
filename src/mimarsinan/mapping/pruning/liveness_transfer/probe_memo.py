"""Run-scoped memoisation of ``derive_constant_outputs`` probe batteries.

The battery is pure in (op, bitwise input key): hosted modules are never
mutated by folds (folds land on core matrices and biases), eval-mode toggles
are restored, and RNG is forked. One memo shared across every fresh context
of a single analysis run (arms, final, replay) therefore removes every
repeated battery while remaining bit-identical by construction. Validity is
scoped to one run: fixed graph, transfer policy and deployment dtypes.
"""

from __future__ import annotations

import struct
from typing import Dict, MutableMapping, Sequence, Tuple

from mimarsinan.mapping.pruning.liveness_transfer import constant_transfer

__all__ = ["ProbeMemo", "derive_constant_outputs_memoized", "probe_key_bytes"]

ProbeMemo = MutableMapping[Tuple[int, bytes], Dict[int, float]]

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
    op, transfer, in_values: Sequence[float | None], memo: ProbeMemo | None
) -> Dict[int, float]:
    """The battery, replayed from ``memo`` when this exact key already ran.

    Refusals ({}) are cached too — a refused probe still paid the full
    battery, and the same inputs refuse again deterministically.
    """
    if memo is None:
        return constant_transfer.derive_constant_outputs(op, transfer, in_values)
    key = (op.id, probe_key_bytes(in_values))
    hit = memo.get(key)
    if hit is None:
        hit = constant_transfer.derive_constant_outputs(op, transfer, in_values)
        memo[key] = hit
    return hit
