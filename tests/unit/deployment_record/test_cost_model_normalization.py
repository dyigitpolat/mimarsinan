"""The latency decomposition must add commensurable quantities.

``compute_sim_time_s`` is a PER-SAMPLE census (sample 0); host-op walls are
accumulated over every invocation of the run. Summing them raw overstates the
per-sample latency by the sample factor and inverts into a wrong throughput.
[U-HOST] The host wall is the development machine's: the measured plane carries it
per inference (``stage_timing.host_wall``: calls x samples per call) as a labelled
diagnostic term beside its chip-side total, never in it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from unit.deployment_record.record_fixtures import make_full_record, with_partition

from mimarsinan.deployment_record.cost import DeploymentCostModel


def _with_host_walls(record, *, wall_s_total, invocations, samples_per_call=1):
    # [U-HOST2] the record states its partition census (its host op, a max pool, carries no host MACs: dispatch
    # only, so the chip-side total stands with the dispatch named unpriced beside it)
    record = with_partition(record)
    stages = tuple(
        replace(s, wall_s_total=wall_s_total, invocations=invocations, samples_per_call=samples_per_call)
        if hasattr(s, "op_type") else s
        for s in record.schedule.stages
    )
    schedule = replace(record.schedule, stages=stages)
    latency = replace(
        record.timing.latency,
        host_ops_s_raw_total=wall_s_total,
        host_ops_s_per_pass=(
            None if invocations in (None, 0) or not samples_per_call
            else wall_s_total / (invocations * samples_per_call)
        ),
    )
    timing = replace(record.timing, latency=latency)
    return replace(record, schedule=schedule, timing=timing)


def _chip_side(terms):
    return (terms["compute_s"].value + terms["programming_s"].value + terms["core_init_s"].value
            + terms["sync_s"].value)


def test_host_wall_is_a_per_inference_diagnostic_never_in_the_total():
    # 40 one-sample calls of a 0.8 s total = 0.02 s per inference; one 25-sample call of 0.8 s = 0.032 s. The
    # measured plane's total is the chip side either way: the wall is the development machine's.
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=40)
    terms = {t.name: t for t in DeploymentCostModel().latency(record)}
    assert terms["host_wall_diagnostic_s"].value == pytest.approx(0.02)
    assert "never in a total" in terms["host_wall_diagnostic_s"].source
    assert terms["total_s"].value == pytest.approx(_chip_side(terms))
    batched = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=1, samples_per_call=25)
    terms = {t.name: t for t in DeploymentCostModel().latency(batched)}
    assert terms["host_wall_diagnostic_s"].value == pytest.approx(0.032)
    assert terms["total_s"].value == pytest.approx(_chip_side(terms))


def test_a_wall_without_its_count_is_withheld_by_name():
    # A measured total with no call or sample count cannot be put per inference; guessing a divisor would be a
    # proxy presented as a measurement: the diagnostic is withheld and the report names why.
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=None, samples_per_call=None)
    model = DeploymentCostModel()
    assert "host_wall_diagnostic_s" not in {t.name for t in model.latency(record)}
    notes = model.evaluate(record).notes
    assert any("host wall diagnostic withheld" in n and "samples per call" in n for n in notes)


def test_throughput_inverts_the_normalized_latency():
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=40)
    model = DeploymentCostModel()
    total = {t.name: t for t in model.latency(record)}["total_s"]
    tp = {t.name: t for t in model.throughput(record)}["samples_per_s"]
    assert tp.value == pytest.approx(1.0 / total.value)


def test_the_basis_states_the_per_sample_normalization():
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=40)
    total = {t.name: t for t in DeploymentCostModel().latency(record)}["total_s"]
    assert "per" in (total.band.basis or "").lower()
    assert "sample" in (total.band.basis or "").lower() or "pass" in (total.band.basis or "").lower()


def _all_terms(record):
    model = DeploymentCostModel()
    latency = model.latency(record)
    return (
        list(model.energy(record)) + list(latency)
        + list(model.area(record)) + list(model._throughput(latency))
    )


def test_every_terms_value_is_its_band_nominal():
    # A consumer reading .value expects the nominal corner; a term whose value
    # silently reports a band edge would skew every downstream comparison.
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=40)
    for term in _all_terms(record):
        if term.band is not None:
            assert term.value == pytest.approx(term.band.nominal), term.name


def test_the_term_set_per_group_is_exact():
    # Pins the published surface: a spurious or vanished term is a contract
    # change, not a silently tolerated difference.
    record = _with_host_walls(make_full_record(), wall_s_total=0.8, invocations=40)
    model = DeploymentCostModel()
    latency = model.latency(record)
    assert {t.name for t in latency} == {
        "programming_s", "core_init_s", "compute_s", "host_wall_diagnostic_s", "sync_s", "total_s",
    }
    assert {t.name for t in model._throughput(latency)} == {"samples_per_s"}
    assert {t.name for t in model.energy(record)} == {
        "measured_sanafe_total_mj", "measured_total_mj", "modeled_core_init_mj",
        "modeled_programming_mj", "modeled_sync_mj", "total_mj",
    }
    assert {t.name for t in model.area(record)} == {
        "cores_used", "cell_occupancy", "cell_waste_fraction",
        "fragmentation_pct", "unusable_space_cells", "unused_area_cells",
    }


def test_owner_signed_coefficients_match_the_schema_doc():
    # docs/deployment_record_schema.md section 7 is the owner-review surface;
    # code drifting from the blessed band must fail, not pass silently.
    from mimarsinan.deployment_record.cost.coefficients import (
        BYTES_PER_CONNECTIVITY_ENTRY,
        PROGRAMMING_BANDWIDTH_BYTES_PER_S,
        SYNC_BARRIER_S,
    )
    assert (BYTES_PER_CONNECTIVITY_ENTRY.low,
            BYTES_PER_CONNECTIVITY_ENTRY.nominal,
            BYTES_PER_CONNECTIVITY_ENTRY.high) == (4.0, 8.0, 16.0)
    assert PROGRAMMING_BANDWIDTH_BYTES_PER_S.nominal == pytest.approx(12.8e9)
    assert (PROGRAMMING_BANDWIDTH_BYTES_PER_S.low,
            PROGRAMMING_BANDWIDTH_BYTES_PER_S.high) == pytest.approx((1e9, 256e9))
    assert SYNC_BARRIER_S.nominal == pytest.approx(8.0e-8)
    assert (SYNC_BARRIER_S.low, SYNC_BARRIER_S.high) == pytest.approx((6.4e-8, 6.4e-7))
