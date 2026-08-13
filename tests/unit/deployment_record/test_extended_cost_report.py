"""The extended report: absolute terms appended, measured plane untouched."""

import json
from dataclasses import replace

import pytest

from mimarsinan.deployment_record.cost.absolute import (
    ABSOLUTE_TERM_NAMES,
    absolute_pricing_for_record,
    candidate_cost_report,
    declared_physics_of,
    price_absolute,
    report_with_absolute_terms,
)
from mimarsinan.deployment_record.cost.model import DeploymentCostModel
from mimarsinan.deployment_record.cost.terms import (
    DeploymentCostReport,
    find_term,
    find_term_or_none,
)
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)

from unit.deployment_record.record_fixtures import make_full_record, make_identity


def _record_with_physics(profile="truenorth"):
    record = make_full_record()
    platform = build_platform_constraints_resolved({
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
        "weight_bits": 8,
        "platform_physics_profile": profile,
    })
    return replace(record, identity=replace(make_identity(), platform=platform))


def _quantities(**values):
    return Quantities({
        key: QuantityValue(float(v), "static") for key, v in values.items()
    })


def test_a_record_without_physics_declares_none():
    assert declared_physics_of(make_full_record()) is None
    assert absolute_pricing_for_record(make_full_record()) is None


def test_the_records_own_physics_prices_it():
    pricing = absolute_pricing_for_record(_record_with_physics())
    assert pricing is not None
    names = {term.name for term in pricing.terms}
    assert "chip_area_mm2" in names
    # 20 declared cores x 0.0936 mm2 + 46.6 mm2 global.
    area = next(t for t in pricing.terms if t.name == "chip_area_mm2")
    assert area.value == pytest.approx(20 * 0.0936 + 46.6)


def test_extension_appends_and_never_rewrites_the_measured_plane():
    record = _record_with_physics()
    base = DeploymentCostModel().evaluate(record)
    pricing = absolute_pricing_for_record(record)
    assert pricing is not None
    extended = report_with_absolute_terms(base, pricing)
    assert extended.energy[: len(base.energy)] == base.energy
    assert extended.latency[: len(base.latency)] == base.latency
    assert extended.segments == base.segments
    assert find_term(extended.area, "chip_area_mm2").unit == "mm^2"
    assert any("parallel PREDICTION" in note for note in extended.notes)


def test_refusals_ride_the_notes_by_name():
    record = _record_with_physics()
    pricing = absolute_pricing_for_record(record)
    assert pricing is not None
    extended = report_with_absolute_terms(DeploymentCostModel().evaluate(record), pricing)
    assert any(
        note.startswith("refused energy_per_inference_mj:") for note in extended.notes
    ), "no event census is sealed: the energy refusal must be visible in the report"


def test_the_extended_report_round_trips():
    record = _record_with_physics()
    pricing = absolute_pricing_for_record(record)
    extended = report_with_absolute_terms(DeploymentCostModel().evaluate(record), pricing)
    payload = json.loads(json.dumps(extended.to_dict()))
    assert DeploymentCostReport.from_dict(payload) == extended


def test_candidate_report_carries_no_segments_and_the_same_term_names():
    physics = get_platform_physics("truenorth")
    report = candidate_cost_report(
        _quantities(cores_physical=20, latency_steps=64, synaptic_events=1e6,
                    host_macs=0),
        physics,
    )
    assert report.segments == ()
    assert find_term(report.area, "chip_area_mm2").name in ABSOLUTE_TERM_NAMES
    assert find_term(report.latency, "e2e_latency_s").name in ABSOLUTE_TERM_NAMES
    assert find_term(report.energy, "energy_per_inference_mj").value == pytest.approx(26e-3)
    assert find_term(report.throughput, "throughput_inferences_s").value == pytest.approx(
        1 / 0.064
    )


def test_find_term_or_none_answers_where_find_term_raises():
    physics = get_platform_physics("truenorth")
    report = candidate_cost_report(_quantities(cores_physical=20, host_macs=0), physics)
    assert find_term_or_none(report.area, "chip_area_mm2") is not None
    assert find_term_or_none(report.energy, "energy_per_inference_mj") is None
    with pytest.raises(KeyError):
        find_term(report.energy, "energy_per_inference_mj")


def test_the_fold_appends_so_measured_positions_never_shift():
    """A consumer indexing the measured plane must not be moved by prediction.

    The record's own quantities refuse energy (no sealed event census), so this
    folds a pricing that DOES carry energy terms — otherwise the append order is
    never exercised on a populated group.
    """
    record = _record_with_physics()
    base = DeploymentCostModel().evaluate(record)
    assert base.energy, "the fixture carries a measured energy plane"
    pricing = price_absolute(
        _quantities(cores_physical=20, latency_steps=64, synaptic_events=1e6,
                    host_macs=0),
        get_platform_physics("truenorth"),
    )
    priced_energy = [t.name for t in pricing.terms if t.unit == "mJ"]
    assert "energy_per_inference_mj" in priced_energy

    extended = report_with_absolute_terms(base, pricing)
    assert extended.energy[: len(base.energy)] == base.energy
    assert [t.name for t in extended.energy[len(base.energy):]] == priced_energy
