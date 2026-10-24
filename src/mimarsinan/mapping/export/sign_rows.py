"""The per-axon sign gate: every used row of a per-axon core carries one sign and a magnitude the chip stores."""

from __future__ import annotations

from typing import Any, Iterator, Optional

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.packing.softcore.sign_split import mixed_rows
from mimarsinan.mapping.platform.platform_constraints import PER_AXON


class SignRowsError(ToolchainRefusal, ValueError):
    """A per-axon core row the chip cannot program: both signs on one row, or the q_min code (a sign bit and a
    (bits - 1)-bit magnitude store at most 2^(bits-1) - 1, so -2^(bits-1) is refused, never saturated)."""


def _per_axon_cores(hybrid_mapping: Any) -> Iterator[tuple]:
    for stage_index, stage in enumerate(hybrid_mapping.stages):
        if stage.kind != "neural" or stage.hard_core_mapping is None:
            continue
        for level in (stage, *(getattr(stage, "retimed_level_stages", None) or ())):
            for core_index, core in enumerate(level.hard_core_mapping.cores):
                if getattr(core, "weight_sign_granularity", None) == PER_AXON:
                    yield stage_index, core_index, core


def assert_sign_rows(hybrid_mapping: Any, *, weight_bits: Optional[int] = None) -> None:
    """Refuse by name a per-axon core row with both signs, or (with ``weight_bits``) a magnitude past
    ``2^(weight_bits-1) - 1``."""
    bits = None if weight_bits is None else int(weight_bits)
    limit = None if bits is None else (1 << (bits - 1)) - 1
    for stage_index, core_index, core in _per_axon_cores(hybrid_mapping):
        grid = np.asarray(core.get_core_matrix())
        both = np.flatnonzero(mixed_rows(grid))
        if both.size:
            raise SignRowsError(
                f"stage {stage_index} core {core_index}: row {int(both[0])} holds both signs on a per-axon core "
                f"(one static sign per axon row)")
        if limit is not None and grid.size:
            peak = float(np.abs(grid).max())
            if peak > limit:
                row = int(np.flatnonzero(np.abs(grid).max(axis=1) == peak)[0])
                raise SignRowsError(
                    f"stage {stage_index} core {core_index}: row {row} needs magnitude {peak:g} on a per-axon "
                    f"core of {bits}-bit weights (a sign and a magnitude of at most {limit})")
