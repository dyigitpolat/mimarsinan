"""[calculus §17/PR44] perceptron-aligned count capture: IR provenance maps
backend stage outputs onto NF-walk perceptron counts, generically over packing."""

from __future__ import annotations

import sys

import pytest
import torch

from mimarsinan.certification.count_alignment import (
    PerceptronCountAssembler,
    intersect_aligned,
    nf_perceptron_counts,
)
from mimarsinan.certification.spike_certificate import certify_spike_counts


def _tiny_with_provenance():
    sys.path.insert(0, "tests/unit/models")
    from test_hybrid_sync_counts import _tiny

    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        build_hybrid_hard_core_mapping,
    )

    repr_, _ = _tiny()
    repr_.assign_perceptron_indices()
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    return repr_, ir, hybrid


def _captured_backend_counts(ir, hybrid, x):
    sys.path.insert(0, "tests/unit/models")
    from test_hybrid_sync_counts import _flow

    import mimarsinan.models.spiking.hybrid.lif_step as ls

    assembler = PerceptronCountAssembler(ir)
    orig = ls.HybridLifStepMixin._run_neural_segment_rate

    def spy(self, stage, **kw):
        counts = orig(self, stage, **kw)
        assembler.capture_stage(stage.output_map, counts)
        return counts

    ls.HybridLifStepMixin._run_neural_segment_rate = spy
    try:
        with torch.no_grad():
            _flow(hybrid, synchronized=True)(x)
    finally:
        ls.HybridLifStepMixin._run_neural_segment_rate = orig
    return assembler


def test_assembler_requires_provenance():
    class _Bare:
        nodes = ()

    with pytest.raises(ValueError, match="provenance"):
        PerceptronCountAssembler(_Bare())


def test_nf_vs_hcm_counts_certify_via_perceptron_alignment():
    sys.path.insert(0, "tests/unit/models")
    from test_hybrid_sync_counts import T as _T

    torch.manual_seed(0)
    repr_, ir, hybrid = _tiny_with_provenance()
    x = torch.rand(3, 8) * 0.9

    ref = nf_perceptron_counts(repr_, _T, x)
    assert set(ref), "the NF walk recorded no perceptron counts"

    assembler = _captured_backend_counts(ir, hybrid, x)
    got = assembler.assemble()
    assert set(got), "the backend capture assembled no perceptron counts"

    aligned_ref, aligned_got, report = intersect_aligned(ref, got)
    assert aligned_ref, report
    cert = certify_spike_counts(
        lambda _b: aligned_ref, lambda _b: aligned_got, [x], backend="hcm",
    )
    assert cert.passed, cert.summary() + " | " + report
    assert cert.exact_match_fraction == 1.0
    assert cert.max_abs_delta == 0.0


class _Node:
    def __init__(self, nid, pi, col=None, sl=None):
        self.id = nid
        self.perceptron_index = pi
        self.perceptron_output_column = col
        self.perceptron_output_slice = sl


class _Slice:
    def __init__(self, node_id, offset, size):
        self.node_id = node_id
        self.offset = offset
        self.size = size


class _IR:
    def __init__(self, nodes):
        self.nodes = nodes


def test_assembler_places_columns_and_tiles_canonically():
    # perceptron 0: 2 columns x 3 channels, column 1 split into two tiles.
    ir = _IR([
        _Node(10, 0, col=0, sl=(0, 3)),
        _Node(11, 0, col=1, sl=(0, 2)),
        _Node(12, 0, col=1, sl=(2, 3)),
    ])
    asm = PerceptronCountAssembler(ir)
    stage = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    asm.capture_stage(
        [_Slice(10, 0, 3), _Slice(11, 3, 2), _Slice(12, 5, 1)], stage,
    )
    got = asm.assemble()
    assert asm.last_report == "all covered"
    torch.testing.assert_close(
        got[0], torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )


def test_assembler_drops_uncaptured_and_overlapping_perceptrons():
    ir = _IR([
        _Node(10, 0, col=0, sl=(0, 2)),
        _Node(11, 1, col=0, sl=(0, 2)),
        _Node(12, 1, col=0, sl=(1, 3)),
    ])
    asm = PerceptronCountAssembler(ir)
    asm.capture_stage([_Slice(11, 0, 2), _Slice(12, 2, 2)],
                      torch.ones(1, 4))
    got = asm.assemble()
    assert got == {}
    assert "p0" in asm.last_report and "uncaptured" in asm.last_report
    assert "p1" in asm.last_report and "overlap" in asm.last_report


def test_certify_flow_counts_one_call_on_the_tiny_fixture():
    """The sanctioned pipeline path: stage_count_recorder seam, sync discipline
    forced and restored, certificate + detail report in one call."""
    from mimarsinan.certification.count_alignment import certify_flow_counts

    sys.path.insert(0, "tests/unit/models")
    from test_hybrid_sync_counts import _flow

    torch.manual_seed(0)
    repr_, ir, hybrid = _tiny_with_provenance()
    x = torch.rand(3, 8) * 0.9

    flow = _flow(hybrid, synchronized=False)
    cert, detail = certify_flow_counts(repr_, ir, flow, x, backend="hcm")
    assert cert.passed and cert.exact_match_fraction == 1.0
    assert "all covered" in detail
    assert flow.lif_execution_synchronized is False
    assert flow.stage_count_recorder is None


def test_streaming_discipline_counts_certify_against_the_same_oracle():
    """[§16 staircase theorem] streaming per-window counts equal the NF sync
    oracle exactly — the metric-of-record cell, verified not assumed."""
    from mimarsinan.certification.count_alignment import certify_flow_counts

    sys.path.insert(0, "tests/unit/models")
    from test_hybrid_sync_counts import _flow

    torch.manual_seed(0)
    repr_, ir, hybrid = _tiny_with_provenance()
    x = torch.rand(3, 8) * 0.9

    flow = _flow(hybrid, synchronized=True)
    cert, _detail = certify_flow_counts(
        repr_, ir, flow, x, backend="hcm", discipline="streaming",
    )
    assert cert.passed, cert.summary()
    assert cert.exact_match_fraction == 1.0
    assert flow.lif_execution_synchronized is True


def test_intersect_aligned_rejects_width_mismatch_on_common_key():
    ref = {1: torch.zeros(2, 4)}
    got = {1: torch.zeros(2, 5)}
    with pytest.raises(ValueError, match="width"):
        intersect_aligned(ref, got)


def test_intersect_aligned_reports_dropped_keys():
    ref = {0: torch.zeros(1, 2), 1: torch.ones(1, 2)}
    got = {1: torch.ones(1, 2), 2: torch.zeros(1, 3)}
    aligned_ref, aligned_got, report = intersect_aligned(ref, got)
    assert set(aligned_ref) == {1} and set(aligned_got) == {1}
    assert "reference-only" in report and "backend-only" in report
