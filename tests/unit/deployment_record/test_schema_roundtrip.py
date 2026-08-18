"""JSON round-trip identity, unknown-field rejection, wrong-version rejection."""

from __future__ import annotations

import json

import pytest

from mimarsinan.deployment_record.schema import (
    Band,
    DeploymentRecord,
    ModeledValue,
    Provenance,
    RecordIdentity,
    load_deployment_record,
    save_deployment_record,
)
from unit.deployment_record.record_fixtures import (
    make_accuracy,
    make_adaptation,
    make_adaptation_summary,
    make_band,
    make_compute_op,
    make_core,
    make_crossbar,
    make_energy,
    make_full_record,
    make_identity,
    make_layout,
    make_placement,
    make_provenance,
    make_read,
    make_schedule,
    make_segment,
    make_softcore_placement,
    make_timing,
    make_traffic,
    make_utilization,
)

FRAGMENT_FIXTURES = [
    make_provenance(),
    make_band(),
    ModeledValue(value=1e-3, band=make_band()),
    make_identity(),
    make_compute_op(),
    make_core(),
    make_segment(),
    make_schedule(),
    make_softcore_placement(),
    make_placement(),
    make_placement(with_floorplan=False),
    make_crossbar(),
    make_layout(),
    make_utilization(),
    make_traffic(),
    make_traffic(with_boundaries=False, with_noc=False),
    make_timing(),
    make_timing(with_per_segment=False),
    make_energy(),
    make_read(),
    make_accuracy(),
    make_adaptation(),
    make_adaptation_summary(),
    make_full_record(),
]


@pytest.mark.parametrize(
    "fragment", FRAGMENT_FIXTURES, ids=lambda f: type(f).__name__
)
def test_json_roundtrip_identity(fragment):
    wire = json.loads(json.dumps(fragment.to_dict()))
    assert type(fragment).from_dict(wire) == fragment


@pytest.mark.parametrize(
    "fragment", FRAGMENT_FIXTURES, ids=lambda f: type(f).__name__
)
def test_unknown_field_rejected(fragment):
    wire = json.loads(json.dumps(fragment.to_dict()))
    wire["field_from_the_future"] = 1
    with pytest.raises(ValueError, match="unknown fields.*field_from_the_future"):
        type(fragment).from_dict(wire)


def test_wrong_version_rejected_on_record_load():
    wire = make_full_record().to_dict()
    wire["format_version"] = 2
    with pytest.raises(ValueError, match="format_version 2"):
        DeploymentRecord.from_dict(wire)


def test_wrong_version_rejected_on_identity_load():
    wire = make_identity().to_dict()
    wire["format_version"] = 0
    with pytest.raises(ValueError, match="format_version 0"):
        RecordIdentity.from_dict(wire)


def test_wrong_version_rejected_at_construction():
    import dataclasses

    with pytest.raises(ValueError, match="format_version"):
        dataclasses.replace(make_full_record(), format_version=2)


def test_save_load_roundtrip(tmp_path):
    record = make_full_record()
    path = save_deployment_record(record, str(tmp_path))
    assert path.endswith("deployment_record.json")
    assert load_deployment_record(path) == record
    assert json.loads(open(path).read())["format_version"] == 1


def test_provenance_kind_validated():
    with pytest.raises(ValueError, match="Provenance.kind"):
        Provenance(kind="guessed", producer="p", step="s")


def test_band_ordering_and_basis_validated():
    with pytest.raises(ValueError, match="low <= nominal <= high"):
        Band(low=2.0, nominal=1.0, high=3.0, basis="b")
    with pytest.raises(ValueError, match="basis"):
        Band(low=1.0, nominal=2.0, high=3.0, basis="")


def test_segment_pass_reason_consistency_validated():
    with pytest.raises(ValueError, match="capacity_overflow"):
        make_segment(pass_index=1, pass_reason="initial")


def test_schedule_stage_discriminator_required():
    wire = make_schedule().to_dict()
    del wire["stages"][0]["stage_kind"]
    with pytest.raises(ValueError, match="stage_kind"):
        make_schedule().__class__.from_dict(wire)
