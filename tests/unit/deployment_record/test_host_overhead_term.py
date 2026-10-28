"""[R2] The host wall has a per-invocation cost; the rate alone understates it."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import (
    apply_overrides,
    get_platform_physics,
)
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue


def _quantities(**values):
    return Quantities({
        key: QuantityValue(float(value), "static") for key, value in values.items()
    })


def _physics(**extra):
    overrides = {
        "host_macs_per_s": {"nominal": 10.0, "unit": "G/s",
                            "evidence_kind": "estimated", "note": "n"},
        "p_host": {"nominal": 20.0, "unit": "W",
                   "evidence_kind": "estimated", "note": "n"},
    }
    overrides.update(extra)
    return apply_overrides(get_platform_physics("loihi"), overrides)


_BASE = dict(synaptic_events=1e6, host_macs=1_000_000, cores_physical=8,
             latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1)


class TestTheComposition:
    def test_declared_overhead_adds_per_invocation_time(self):
        """1M MACs at 10 G/s = 100 us of work; 3 ops x 500 us of dispatch
        dwarf it — the shape the study measured (~2000x on tiny ops)."""
        physics = _physics(t_host_op_overhead={
            "nominal": 500.0, "unit": "us",
            "evidence_kind": "measured", "note": "n"})
        with_ops = price_absolute(
            _quantities(**_BASE, compute_op_count=3), physics)
        term = {t.name: t for t in with_ops.terms}["latency_host_s"]
        assert term.value == pytest.approx(100e-6 + 3 * 500e-6, rel=1e-6)
        assert "t_host_op_overhead" in term.source

    def test_an_undeclared_overhead_prices_the_rate_and_says_so(self):
        """The rate-only figure still prices — refusing e2e over an optional
        refinement would lose the chip-side terms — but the source NAMES the
        unpriced dispatch, so the number cannot read as complete."""
        pricing = price_absolute(
            _quantities(**_BASE, compute_op_count=3), _physics())
        term = {t.name: t for t in pricing.terms}["latency_host_s"]
        assert term.value == pytest.approx(100e-6, rel=1e-6)
        assert "unpriced: t_host_op_overhead" in term.source

    def test_a_measured_wall_never_double_charges_the_overhead(self):
        """The record plane's measured wall already CONTAINS the dispatch —
        the overhead term is candidate-plane only."""
        physics = _physics(t_host_op_overhead={
            "nominal": 500.0, "unit": "us",
            "evidence_kind": "measured", "note": "n"})
        pricing = price_absolute(
            _quantities(**_BASE, compute_op_count=3, host_ops_s=0.002), physics)
        term = {t.name: t for t in pricing.terms}["latency_host_s"]
        assert term.value == pytest.approx(0.002, rel=1e-6)
        assert "t_host_op_overhead" not in term.source


_NO_HOST_MACS = dict(_BASE, host_macs=0)


class TestDispatchWithoutHostMacs:
    """Host ReLU, add, pool, softmax and LayerNorm ops carry no MACs but still dispatch."""

    def test_a_declared_overhead_prices_the_dispatch(self):
        physics = _physics(t_host_op_overhead={
            "nominal": 500.0, "unit": "us", "evidence_kind": "measured", "note": "n"})
        pricing = price_absolute(_quantities(**_NO_HOST_MACS, compute_op_count=3), physics)
        term = {t.name: t for t in pricing.terms}["latency_host_s"]
        assert term.value == pytest.approx(3 * 500e-6, rel=1e-12)
        assert "t_host_op_overhead x compute_op_count" in term.source

    def test_an_undeclared_overhead_is_named_unpriced(self):
        pricing = price_absolute(_quantities(**_NO_HOST_MACS, compute_op_count=3), _physics())
        terms = {t.name: t for t in pricing.terms}
        assert "latency_host_s" not in terms
        assert "unpriced: t_host_op_overhead(per-invocation dispatch)" in terms["e2e_latency_s"].source
        assert "p_host x t_host_op_overhead(per-invocation dispatch)" in terms["energy_per_inference_mj"].source

    def test_host_macs_keep_their_behaviour(self):
        pricing = price_absolute(_quantities(**_BASE, compute_op_count=3), _physics())
        assert {t.name: t for t in pricing.terms}["latency_host_s"].value == pytest.approx(100e-6, rel=1e-6)

    def test_a_measured_wall_without_host_macs_keeps_an_empty_host_time(self):
        """ODIN declares no host_compute_rate: a record's measured wall must not turn into a refusal. [U-C2 G-2] ODIN
        prices its compute on the event census of one core, so the census here is one ODIN core's."""
        odin = get_platform_physics("odin")
        pricing = price_absolute(
            _quantities(**dict(_NO_HOST_MACS, cores_physical=1), compute_op_count=3, host_ops_s=0.002), odin)
        assert "latency_host_s" not in {t.name for t in pricing.terms}
        assert not any("host" in r.reason for r in pricing.refusals)
        assert "t_host_op_overhead" not in {t.name: t for t in pricing.terms}["e2e_latency_s"].source

    def test_no_dispatch_no_host_term(self):
        pricing = price_absolute(_quantities(**_NO_HOST_MACS, compute_op_count=0), _physics())
        assert "unpriced: t_host_op_overhead" not in {t.name: t for t in pricing.terms}["e2e_latency_s"].source


class TestTheCandidateCountsItsOps:
    def test_the_flow_walk_counts_host_invocations(self):
        from unit.search.test_candidate_fragments_live_path import (
            _candidate,
            _physics_cfg,
            _problem,
        )

        problem = _problem(_physics_cfg(), ["e2e_latency_s"])
        q = problem.candidate_layout(_candidate(problem)).view.quantities
        assert q.has("compute_op_count")
        assert q.get("compute_op_count").value >= 1


class TestTheCalibrationBlock:
    def test_the_measured_band_is_percentiles_not_a_point(self):
        from mimarsinan.deployment_record.platform_physics.host_calibration import (
            calibration_overrides,
        )

        block = calibration_overrides(
            macs_per_s=1e9, op_overhead_s=(10e-6, 20e-6, 40e-6),
            p_host_w=None, identity="h")
        overhead = block["t_host_op_overhead"]
        assert (overhead["low"], overhead["nominal"], overhead["high"]) == (
            10.0, 20.0, 40.0)
        assert overhead["evidence_kind"] == "measured"
