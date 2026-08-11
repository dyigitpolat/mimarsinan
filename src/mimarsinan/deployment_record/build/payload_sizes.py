"""Payload sizing SSOT: exact programmed-weight bytes + compressed span counts.

The documented ``weight_bits`` edge-case choice: ``None`` (an undeclared
platform weight width) FAILS LOUD. ``SegmentCoreRecord.params_bytes`` is an
exact integer per the schema (docs/deployment_record_schema.md §2.1), so an
undeclared width cannot size a payload honestly — and every real pipeline
resolves ``weight_bits`` into ``platform_constraints_resolved`` before the
mapping steps run.
"""

from __future__ import annotations

from typing import Any


def require_weight_bits(weight_bits: Any) -> int:
    """The resolved platform's weight width, validated (None/non-positive raise)."""
    if weight_bits is None:
        raise ValueError(
            "payload sizing requires a declared platform weight width; "
            "'weight_bits' is missing from the resolved platform constraints"
        )
    bits = int(weight_bits)
    if bits <= 0:
        raise ValueError(f"weight_bits must be positive, got {bits}")
    return bits


def params_bytes(cells_used: int, weight_bits: Any) -> int:
    """``ceil(cells_used * weight_bits / 8)`` — exact programmed-payload bytes."""
    bits = require_weight_bits(weight_bits)
    cells = int(cells_used)
    if cells < 0:
        raise ValueError(f"cells_used must be >= 0, got {cells}")
    return -((cells * bits) // -8)


def core_connectivity_entries(hard_core: Any) -> int:
    """Compressed axon-source span count of one packed hard core.

    Reuses the span SSOT — ``HardCore.get_axon_source_spans()`` caches
    ``compress_spike_sources``, the same run-length encoding behind the
    codegen ``compress_sources_to_spans`` export — never a reimplementation.
    """
    return len(hard_core.get_axon_source_spans())
