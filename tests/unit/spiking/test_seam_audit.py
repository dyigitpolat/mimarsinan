"""Defect-injection validation of the seam-certificate auditor (spiking_deployment_calculus.md sec.9).

The auditor is UNTRUSTED until it (a) reports no Type-B on a clean armed
fixture, and (b) localizes every planted defect to the right site with the
right class: currency mismatch (B), offset on an unarmed op (B), wire/value
twin divergence (B), capacity saturation (C). The fixture is the signed-seam
topology at unit scale (LIF -> signed LayerNorm host op -> LIF, offload).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.spiking.seam_audit import audit_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 32


def _lif_perceptron(out_ch, in_features, theta):
    p = Perceptron(out_ch, in_features, normalization=nn.Identity())
    p.set_activation_scale(theta)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    p.base_activation = lif
    p.activation = lif
    return p


def _seam_repr(theta1: float = 1.7, theta2: float = 0.9, arm: bool = True):
    """input(8) -> P1(6, theta1) -> LayerNorm(6) -> P2(3, theta2), offload."""
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = _lif_perceptron(6, 8, theta1)
    m1 = PerceptronMapper(inp, p1)
    ln = nn.LayerNorm(6)
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.fill_(0.8)
    host = ComputeOpMapper(m1, ln, input_shape=(6,), output_shape=(6,))
    p2 = _lif_perceptron(3, 6, theta2)
    p2.per_input_scales = torch.full((6,), float(theta1))
    p2.set_input_activation_scale(torch.tensor(float(theta1)))
    m2 = PerceptronMapper(host, p2)
    repr_ = ModelRepresentation(m2)
    mark_encoding_layers(repr_, placement="offload")
    if arm:
        compute_per_source_scales(repr_)
    return repr_, p1, p2, host


def _audit(repr_):
    torch.manual_seed(7)
    x = torch.rand(64, 8)
    return audit_model(repr_, T, x)


class TestCleanFixture:
    def test_no_type_b_on_clean_armed_fixture(self):
        repr_, _, _, _ = _seam_repr()
        ledger = _audit(repr_)
        assert ledger.type_b == (), [
            (c.site, c.kind, c.note) for c in ledger.type_b
        ]

    def test_ledger_covers_boundaries_and_kernels(self):
        repr_, _, _, _ = _seam_repr()
        ledger = _audit(repr_)
        kinds = {c.kind for c in ledger.certificates}
        assert {"boundary", "kernel", "currency", "host_twin"} <= kinds
        assert len(ledger.by_kind("boundary")) >= 2  # input->P1 and host->P2
        assert len(ledger.by_kind("kernel")) == 2

    def test_signed_seam_reads_capacity_not_convention(self):
        """The un-shifted signed LN seam loses band mass: an honest Type-C
        coverage read, never a Type-B (the taxonomy's central distinction)."""
        repr_, _, _, _ = _seam_repr()
        ledger = _audit(repr_)
        seam = [
            c for c in ledger.by_kind("boundary")
            if c.classification == "C" and c.oob_fraction > 0.01
        ]
        assert seam, [
            (c.site, c.classification, c.oob_fraction)
            for c in ledger.by_kind("boundary")
        ]


class TestInjectedDefects:
    def test_currency_mismatch_is_type_b(self):
        """A consumer fold stamped at 2x the producer currency is V-B's
        signature: the auditor must flag the consumer's currency certificate."""
        repr_, _, p2, _ = _seam_repr()
        p2.per_input_scales = torch.full((6,), 2 * 1.7)
        p2.set_input_activation_scale(torch.tensor(2 * 1.7))
        ledger = _audit(repr_)
        hits = [c for c in ledger.type_b if c.kind == "currency"]
        assert hits and all("currency" in c.note for c in hits)

    def test_offset_on_unarmed_op_is_type_b(self):
        """The armed-only stamping law (calculus B6): an offset carried by an
        op with no wrap slots exists in the plain forward alone — a
        train/deploy split by construction."""
        repr_, _, _, host = _seam_repr(arm=False)
        host.output_value_offset = torch.tensor(0.3)
        ledger = _audit(repr_)
        hits = [c for c in ledger.type_b if "unarmed" in c.note]
        assert hits

    def test_wire_value_twin_divergence_is_type_b(self):
        """A corrupted SNW DECODE gauge (per_source_scales) makes the emitted
        wire composition disagree with the trained value twin — the
        deployed-side half of the B10 transparency contract."""
        repr_, _, _, host = _seam_repr()
        assert host.per_source_scales is not None
        host.per_source_scales = [
            torch.as_tensor(s) * 2.0 for s in host.per_source_scales
        ]
        ledger = _audit(repr_)
        hits = [c for c in ledger.type_b if c.kind == "host_twin"]
        assert hits

    def test_output_gauge_corruption_surfaces_as_currency_not_twin_split(self):
        """Under one-writer coherence (calculus §11.2) the table FOLLOWS an
        armed op's output gauge, so corrupting it cannot split the twins —
        it moves the whole currency, caught at the consumer's stamps."""
        repr_, _, _, host = _seam_repr()
        assert host.output_scale is not None
        host.output_scale = torch.as_tensor(host.output_scale) * 2.0
        ledger = _audit(repr_)
        assert not [c for c in ledger.type_b if c.kind == "host_twin"]
        assert [c for c in ledger.type_b if c.kind == "currency"]

    def test_starved_currency_is_type_c_with_high_oob(self):
        """theta far below the seam's band saturates the encode: capacity
        starvation must read C with a large out-of-band fraction, not B."""
        repr_, _, _, _ = _seam_repr(theta1=0.25)
        ledger = _audit(repr_)
        seam = [
            c for c in ledger.by_kind("boundary")
            if c.classification == "C" and c.oob_fraction > 0.25
        ]
        assert seam
        assert not [c for c in ledger.type_b if c.kind == "boundary"]

    def test_broken_kernel_is_type_b(self):
        """An activation that is not its own count staircase (bias well above
        one grid step) breaks the A2 train-side identity."""

        class _Biased(nn.Module):
            def __init__(self, lif):
                super().__init__()
                self.lif = lif

            def forward(self, x):
                return self.lif(x) + 0.2

        repr_, p1, _, _ = _seam_repr()
        p1.activation = _Biased(p1.activation)
        ledger = _audit(repr_)
        hits = [c for c in ledger.type_b if c.kind == "kernel"]
        assert hits


class TestLedgerShape:
    def test_certificates_carry_gauges_and_bounds(self):
        repr_, _, _, _ = _seam_repr()
        ledger = _audit(repr_)
        for c in ledger.certificates:
            assert c.classification in ("B", "C", "G", "ok")
            assert c.grid_bound >= 0.0
            assert c.kappa > 0.0
        boundary = ledger.by_kind("boundary")
        assert all(
            abs(c.grid_bound - c.kappa / (2 * T)) < 1e-9 for c in boundary
        )
