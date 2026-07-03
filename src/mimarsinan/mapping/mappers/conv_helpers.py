"""Shared helpers for the shared-weight convolution mappers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from mimarsinan.mapping.ir import IRSource


def _chunk_sizes(total: int, chunk: int):
    assert chunk > 0
    sizes = []
    remaining = int(total)
    while remaining > 0:
        sizes.append(min(chunk, remaining))
        remaining -= sizes[-1]
    return sizes


def pad_source_grid(
    input_sources: np.ndarray,
    pad_width: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Pad an IRSource object grid with OFF sources (np.pad's stubs reject object fills)."""
    off_source = IRSource(node_id=-1, index=0)
    return np.pad(
        input_sources,
        pad_width,
        mode="constant",
        constant_values=cast(Any, off_source),
    )
