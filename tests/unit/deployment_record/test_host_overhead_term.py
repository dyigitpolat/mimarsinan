"""[R2] The host wall has a per-invocation cost; the rate alone understates it."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import (
    apply_overrides,
    get_platform_physics,
)
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

import tempfile
from pathlib import Path

from unit.deployment_record.registered_hosts import declared_rows, registered

from unit.deployment_record.printed_names import unpriced_names


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
    # [U-HOST2] the test numbers as a test host's own description, registered for the resolve
    with tempfile.TemporaryDirectory() as directory, registered(Path(directory), "test_overhead_host", overrides):
        return apply_overrides(get_platform_physics("loihi"), declared_rows("test_overhead_host", overrides))


_BASE = dict(synaptic_events=1e6, host_macs=1_000_000, cores_physical=8,
             latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1)


class TestTheComposition:
    def test_declared_overhead_adds_per_invocation_time(self):
        """1M MACs at 10 G/s = 100 us of work; 3 ops x 500 us of dispatch
        dwarf it — the shape the study measured (~2000x on tiny ops)."""
        physics = _physics(t_host_op_overhead={
            "nominal": 500.0, "unit": "us",
            "evidence_kind": "estimated", "note": "n"})  # [U-HOST2] a test number
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

    def test_a_measured_wall_prices_nothing(self):
        """[U-HOST] The development machine's measured wall is a diagnostic: the declared host's rate and dispatch
        price the host time on every plane, with or without a wall."""
        physics = _physics(t_host_op_overhead={
            "nominal": 500.0, "unit": "us",
            "evidence_kind": "estimated", "note": "n"})  # [U-HOST2] a test number
        pricing = price_absolute(
            _quantities(**_BASE, compute_op_count=3, host_ops_s=0.002), physics)
        term = {t.name: t for t in pricing.terms}["latency_host_s"]
        assert term.value == pytest.approx(100e-6 + 3 * 500e-6, rel=1e-6)
        assert "t_host_op_overhead" in term.source and "host_ops_s" not in term.source


_NO_HOST_MACS = dict(_BASE, host_macs=0)


class TestDispatchWithoutHostMacs:
    """Host ReLU, add, pool, softmax and LayerNorm ops carry no MACs but still dispatch."""

    def test_a_declared_overhead_prices_the_dispatch(self):
        physics = _physics(t_host_op_overhead={
            "nominal": 500.0, "unit": "us", "evidence_kind": "estimated", "note": "n"})  # [U-HOST2] a test number
        pricing = price_absolute(_quantities(**_NO_HOST_MACS, compute_op_count=3), physics)
        term = {t.name: t for t in pricing.terms}["latency_host_s"]
        assert term.value == pytest.approx(3 * 500e-6, rel=1e-12)
        assert "t_host_op_overhead x compute_op_count" in term.source

    def test_an_undeclared_overhead_is_named_unpriced(self):
        pricing = price_absolute(_quantities(**_NO_HOST_MACS, compute_op_count=3), _physics())
        terms = {t.name: t for t in pricing.terms}
        assert "latency_host_s" not in terms
        # [U-ENERGY class 2] read among the unpriced names: a row over a census the view does not produce precedes it
        assert "t_host_op_overhead(per-invocation dispatch)" in unpriced_names(terms["e2e_latency_s"].source)
        assert "p_host x t_host_op_overhead(per-invocation dispatch)" in terms["energy_per_inference_mj"].source

    def test_host_macs_keep_their_behaviour(self):
        pricing = price_absolute(_quantities(**_BASE, compute_op_count=3), _physics())
        assert {t.name: t for t in pricing.terms}["latency_host_s"].value == pytest.approx(100e-6, rel=1e-6)

    def test_a_measured_wall_without_host_macs_prices_nothing(self):
        """[U-HOST] A record's measured wall without host MACs neither prices the host nor turns into a refusal: the
        dispatch is named unpriced (no host is declared). [U-C2 G-2] ODIN prices its compute on the event census of
        one core, so the census here is one ODIN core's."""
        odin = get_platform_physics("odin")
        pricing = price_absolute(
            _quantities(**dict(_NO_HOST_MACS, cores_physical=1), compute_op_count=3, host_ops_s=0.002), odin)
        assert "latency_host_s" not in {t.name for t in pricing.terms}
        assert not any("host" in r.reason for r in pricing.refusals)
        source = {t.name: t for t in pricing.terms}["e2e_latency_s"].source
        assert "t_host_op_overhead(per-invocation dispatch)" in source.split("unpriced: ", 1)[-1]
        assert "host_ops_s" not in source

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
        """[U-HOST] The calibration's rows are the development machine's diagnostic, measured percentiles."""
        from mimarsinan.deployment_record.platform_physics.host_calibration import (
            calibration_diagnostic,
        )

        block = calibration_diagnostic(
            macs_per_s=1e9, op_overhead_s=(10e-6, 20e-6, 40e-6),
            p_host_w=None, identity="h")
        overhead = block["t_host_op_overhead"]
        assert (overhead["low"], overhead["nominal"], overhead["high"]) == (
            10.0, 20.0, 40.0)
        assert overhead["evidence_kind"] == "measured"
