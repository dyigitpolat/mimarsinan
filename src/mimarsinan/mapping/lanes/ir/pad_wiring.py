"""A planned constant-zero pad as on-chip wiring (branch tcad/residual-sum-rounding): the pad leaves the program and
its readers read its input's sources, with an off source (the constant 0) at each padded element. An off element is
what a pruned column already is: an operand row reading it adds 0, a pool window and a crossbar row read nothing."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.lanes.forms import PadForm

_OFF = IRSource(node_id=-1, index=0)


def padded_sources(sources, input_shape: Sequence[int], form: PadForm) -> np.ndarray:
    """The pad's output elements (flat, C order) as sources: ``sources`` (flat over ``input_shape``) placed by
    ``form``, an off source at each padded element, a negative amount cropping."""
    grid = np.asarray(sources, dtype=object).reshape(tuple(int(d) for d in input_shape))
    grid = grid[tuple(slice(max(0, -b), d - max(0, -a)) for d, (b, a) in zip(grid.shape, form.widths))]
    out = np.full(form.output_shape(input_shape), _OFF, dtype=object)
    out[tuple(slice(max(0, b), max(0, b) + d) for d, (b, _a) in zip(grid.shape, form.widths))] = grid
    return out.reshape(-1)
