"""The contract: a sealed run compared against the candidate view of its own config."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.fidelity import FIDELITY_FILENAME
from mimarsinan.deployment_record.fidelity_build import (
    fidelity_report_for_record,
    emit_fidelity_report,
)
from mimarsinan.deployment_record.objectives.views import CandidateStaticView
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities import CandidateQuantityContext
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)

from unit.deployment_record.record_fixtures import (
    make_full_record,
    make_identity,
    make_layout,
)


def _record(profile="truenorth"):
    platform = build_platform_constraints_resolved({
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
        "platform_physics_profile": profile,
    })
    return replace(make_full_record(), identity=replace(make_identity(), platform=platform))


def _candidate():
    return CandidateStaticView(
        layout=make_layout(),
        chip_param_capacity=20 * 256 * 256.0,
        total_params=1000.0,
        host_side_segment_count=1,
        physics=get_platform_physics("truenorth"),
        quantity_context=CandidateQuantityContext(
            timesteps=32, latency_steps=32, activity_factor=0.05, weight_bits=8,
            cores_physical=20, neurons_physical=5120, axons_physical=5120,
            host_macs=0, onchip_macs=100000,
        ),
    )


class TestTheComparison:
    def test_it_compares_the_axes_both_sides_can_answer(self):
        report = fidelity_report_for_record(_record(), _candidate())
        keys = {axis.key for axis in report.axes}
        assert "chip_area_mm2" in keys, "both sides price area from the same physics"

    def test_noc_hops_zip_modeled_against_measured(self):
        """[N4] The traffic axis rides fidelity automatically: the candidate's
        wireload estimate against the sealed census, one axis, both sides."""
        candidate = replace(
            _candidate(),
            quantity_context=replace(
                _candidate().quantity_context,
                cores_per_tile=1, tile_mesh_height=1,
            ),
            noc_fragments=SimpleNamespace(
                pass_placements=(((0, 0), (1, 1)),),
                census=SimpleNamespace(
                    pair_wires={(0, 1): 4}, input_wires=(2, 0), on_wires=(0, 0),
                ),
            ),
        )
        report = fidelity_report_for_record(_record(), candidate)
        hops = [a for a in report.axes if a.key == "noc_total_hops"]
        assert hops, "the traffic axis must ride the fidelity zip"
        assert hops[0].predicted is not None and hops[0].predicted > 0.0
        assert hops[0].measured is not None

    def test_area_agrees_exactly_because_both_price_the_declared_chip(self):
        """The strongest structural check available: area depends on the DECLARED
        chip, which the candidate and the record describe identically."""
        report = fidelity_report_for_record(_record(), _candidate())
        area = [a for a in report.axes if a.key == "chip_area_mm2"][0]
        assert area.predicted == pytest.approx(area.measured)
        assert area.relative_error == pytest.approx(0.0)

    def test_the_predicted_band_comes_from_the_candidates_own_priced_term(self):
        """Without the band there is no in-band verdict at all, so the whole
        correlation collapses to a bare number pair."""
        report = fidelity_report_for_record(_record(), _candidate())
        area = [a for a in report.axes if a.key == "chip_area_mm2"][0]
        assert area.predicted_band is not None
        low, high = area.predicted_band
        assert low <= area.predicted <= high
        assert area.in_band is True

    def test_a_static_axis_carries_no_band(self):
        """Only the vendor-priced axes are banded; a census is a count."""
        report = fidelity_report_for_record(_record(), _candidate())
        capacity = [a for a in report.axes if a.key == "total_param_capacity"][0]
        assert capacity.predicted_band is None
        assert capacity.in_band is None

    def test_the_structural_pass_count_is_compared(self):
        report = fidelity_report_for_record(_record(), _candidate())
        keys = {axis.key for axis in report.axes}
        assert "pass_count" in keys

    def test_an_axis_only_the_record_answers_records_the_measurement_alone(self):
        report = fidelity_report_for_record(_record(), _candidate())
        measured_only = [
            a for a in report.axes
            if a.measured is not None and a.predicted is None
        ]
        assert measured_only, "a sealed run measures axes no candidate can predict"

    def test_the_report_carries_the_runs_identity(self):
        record = _record()
        report = fidelity_report_for_record(record, _candidate())
        assert report.cell_key == record.identity.cell_key
        assert report.run_dir == record.identity.run_dir

    def test_axes_are_reported_in_catalog_order(self):
        from mimarsinan.deployment_record.objectives import OBJECTIVES

        report = fidelity_report_for_record(_record(), _candidate())
        order = list(OBJECTIVES.keys())
        indices = [order.index(axis.key) for axis in report.axes]
        assert indices == sorted(indices)


class TestEmission:
    def test_a_run_without_a_candidate_view_writes_nothing(self, tmp_path):
        """Nothing to compare against is not a fidelity of zero."""
        assert emit_fidelity_report(_record(), None, str(tmp_path)) is None
        assert not (tmp_path / FIDELITY_FILENAME).exists()

    def test_a_run_with_one_writes_the_report(self, tmp_path):
        path = emit_fidelity_report(_record(), _candidate(), str(tmp_path))
        assert path is not None
        assert (tmp_path / FIDELITY_FILENAME).exists()

    def test_the_written_report_reloads(self, tmp_path):
        from mimarsinan.deployment_record.fidelity import load_fidelity_report

        emit_fidelity_report(_record(), _candidate(), str(tmp_path))
        report = load_fidelity_report(str(tmp_path / FIDELITY_FILENAME))
        assert report.compared_count > 0
