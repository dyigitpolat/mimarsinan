"""Differential gate: the vectorized probe surround vs the frozen scalar oracle.

The battery (five executions, fork_rng, eval toggling) is untouched by the
vectorization — production's ``_probe`` is REUSED by the oracle below, so this
differential isolates exactly what changed: input assembly, carriability,
region satisfiability and the resolve loop. The oracle is the verbatim scalar
resolve path frozen at 4dcff2df. Any mismatch — value, refusal, key set —
kills the vectorized path (the O3 rule). Compare as float hex: bit-exact.
"""

import numpy as np
import pytest
import torch
from torch import nn

import mimarsinan.mapping.pruning.liveness_transfer.constant_transfer as ct
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.pruning.liveness_transfer.transfer_types import (
    OPAQUE_TRANSFER,
    LivenessTransfer,
)

from unit.mapping.constant_vehicles import srcs


def _op(module, n_in, n_out=None):
    return ComputeOp(
        id=17, name="diff-probe",
        input_sources=srcs([(0, j) for j in range(n_in)]),
        op_type=type(module).__name__,
        params={"module": module, "input_shape": (n_in,)},
        input_shape=(n_in,), output_shape=(n_out if n_out is not None else n_in,),
    )


def _oracle(op, transfer, in_values):
    """The scalar resolve path, frozen verbatim (reuses production's battery)."""
    n_in = int(len(op.input_sources.flatten()))
    if n_in == 0 or len(in_values) != n_in:
        return {}
    module = ct._module_of(op)
    if module is None:
        return ct._relay_constants(op, in_values)
    known = {i: v for i, v in enumerate(in_values) if v is not None}
    if not known:
        return {}
    dtype = ct._deployment_dtype(op)
    if dtype is None or not all(
        float(torch.tensor(v, dtype=dtype)) == float(v) for v in known.values()
    ):
        return {}
    if transfer.is_opaque or not transfer.out_to_ins:
        if len(known) < n_in:
            return {}
    elif not any(r.issubset(known) for r in transfer.out_to_ins.values()):
        return {}
    probes = [
        [known.get(i, filler) for i in range(n_in)] for filler in ct._PROBE_FILLERS
    ]
    probed = ct._probe(op, module, probes, dtype)
    if probed is None:
        return {}
    y_a, y_b = probed
    flat_a = y_a.flatten().tolist()
    flat_b = y_b.flatten().tolist()
    if not transfer.is_opaque and transfer.out_to_ins:
        regions = {int(o): frozenset(r) for o, r in transfer.out_to_ins.items()}
    else:
        every = frozenset(range(n_in))
        regions = {o: every for o in range(len(flat_a))}
    resolved = {}
    for o, value in enumerate(flat_a):
        region = regions.get(o)
        if region is None or not region.issubset(known):
            continue
        if flat_b[o] != value:
            continue
        resolved[o] = float(value)
    return resolved


def _hex(d):
    return {k: float(v).hex() for k, v in d.items()}


class _NaNAtZero(nn.Module):
    """0/0 at exactly 0.0 input: NaN outputs must refuse identically."""

    def forward(self, x):
        return x / x


def _mapped_transfer(n_in, regions):
    return LivenessTransfer(
        kind="elementwise_1to1",
        out_to_ins={o: frozenset(r) for o, r in regions.items()},
        in_to_outs={},
    )


CASES = []
_rng = np.random.default_rng(23)
for trial in range(6):
    n = int(_rng.integers(3, 24))
    vals = [
        None if _rng.random() < 0.4
        else float(np.float32(_rng.normal()))          # f32-carriable
        for _ in range(n)
    ]
    CASES.append(("opaque_sigmoid", nn.Sigmoid().eval(), OPAQUE_TRANSFER, vals, n))
    regions = {o: [o] for o in range(n)}               # elementwise 1:1
    CASES.append(
        ("mapped_identity", nn.Identity(), _mapped_transfer(n, regions), vals, n)
    )
    pool = {o: [2 * o, 2 * o + 1] for o in range(n // 2)}   # pooling-style
    CASES.append(
        ("mapped_pool", nn.Identity(), _mapped_transfer(n, pool), vals, n)
    )

CASES.append(  # carriability refusal: 1 + 2**-40 is not f32-representable
    ("uncarriable", nn.Sigmoid().eval(), OPAQUE_TRANSFER,
     [1.0 + 2.0 ** -40, 0.5, 0.25], 3)
)
CASES.append(  # -0.0 through an elementwise map
    ("negative_zero", nn.Identity(),
     _mapped_transfer(3, {0: [0], 1: [1], 2: [2]}), [-0.0, None, 2.5], 3)
)
CASES.append(  # NaN outputs refuse on both sides
    ("nan_outputs", _NaNAtZero(), OPAQUE_TRANSFER, [0.0, 0.0, 1.0], 3)
)
CASES.append(  # region indices beyond the output tensor are never visited
    ("region_out_of_range", nn.Identity(),
     _mapped_transfer(4, {0: [0], 9: [1]}), [1.5, 0.75, None, None], 4)
)
CASES.append(  # all-TOP guarded upstream, but the seam must agree anyway
    ("all_top", nn.Sigmoid().eval(), OPAQUE_TRANSFER, [None, None, None], 3)
)


@pytest.mark.parametrize("label,module,transfer,vals,n", CASES,
                         ids=[c[0] + f"_{i}" for i, c in enumerate(CASES)])
def test_vectorized_equals_frozen_scalar_oracle(label, module, transfer, vals, n):
    op = _op(module, n)
    got = ct.derive_constant_outputs(op, transfer, list(vals))
    want = _oracle(_op(module, n), transfer, list(vals))
    assert _hex(got) == _hex(want), (
        f"{label}: vectorized path diverged from the frozen scalar oracle"
    )


class TestTheDifferentialCanFail:
    """Corrupt the vectorized result and prove the hex compare rejects it."""

    def test_a_perturbed_value_is_caught(self):
        op = _op(nn.Sigmoid().eval(), 3)
        got = ct.derive_constant_outputs(op, OPAQUE_TRANSFER, [0.0, 0.0, 0.0])
        want = _oracle(_op(nn.Sigmoid().eval(), 3), OPAQUE_TRANSFER, [0.0, 0.0, 0.0])
        assert got and want
        got = dict(got)
        got[0] = float(np.nextafter(got[0], 1.0))
        assert _hex(got) != _hex(want)
