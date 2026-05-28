"""Regression: hybrid rate forward must bind recording types used at runtime."""

from __future__ import annotations

import ast
from pathlib import Path

from mimarsinan.chip_simulation.recording.records import SegmentSpikeRecord


def test_rate_forward_binds_segment_spike_record():
    import mimarsinan.models.spiking.hybrid.rate_forward as mod

    assert getattr(mod, "SegmentSpikeRecord", None) is SegmentSpikeRecord


def test_rate_forward_no_undefined_segment_spike_record_in_ast():
    """Static guard mirroring ``scripts/check_undefined_names.py`` for this module."""
    root = Path(__file__).resolve().parents[3]
    path = root / "src" / "mimarsinan" / "models" / "spiking" / "hybrid" / "rate_forward.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    loads = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports.add(alias.asname or alias.name)
    assert "SegmentSpikeRecord" in loads
    assert "SegmentSpikeRecord" in imports
