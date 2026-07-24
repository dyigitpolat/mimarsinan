"""[cert-plan W2] twin-cert failure diagnostic: per-node schedule tables."""

from __future__ import annotations

import sys

import torch

from mimarsinan.certification.twin_schedule import twin_schedule_diagnostic
from mimarsinan.pipelining.core.simulation_factory import (
    build_identity_mapping_for_pipeline,
)


def test_diagnostic_reports_per_node_latency_tables():
    sys.path.insert(0, "tests/unit/certification")
    from test_count_alignment import _tiny_with_provenance

    torch.manual_seed(0)
    _repr, ir, hybrid = _tiny_with_provenance()
    identity = build_identity_mapping_for_pipeline(ir, pipeline_config=None)
    report = twin_schedule_diagnostic(identity, hybrid)
    assert "node" in report
    assert ("identical for all" in report) or (
        "identity=" in report and "packed=" in report)


def test_diagnostic_survives_missing_placements():
    from types import SimpleNamespace

    bare = SimpleNamespace(stages=[])
    report = twin_schedule_diagnostic(bare, bare)
    assert "no neural stages" in report or "no placement" in report
