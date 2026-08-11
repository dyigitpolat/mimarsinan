"""The builder's attach-once slots, required-fragment matrix, and seal cross-checks."""

from __future__ import annotations

import dataclasses

import pytest

from mimarsinan.deployment_record.build import DeploymentRecordBuilder
from mimarsinan.deployment_record.schema import DeploymentRecord
from unit.deployment_record.record_fixtures import (
    PARAMS_PROGRAMMED_TOTAL,
    PlanView,
    make_accuracy,
    make_adaptation,
    make_energy,
    make_identity,
    make_placement,
    make_provenance,
    make_schedule,
    make_timing,
    make_traffic,
    make_utilization,
)

ALWAYS_GROUPS = ("identity", "schedule", "placement", "utilization", "accuracy", "timing")


def _fragment(group, **kwargs):
    makers = {
        "identity": make_identity,
        "schedule": make_schedule,
        "placement": make_placement,
        "utilization": make_utilization,
        "accuracy": make_accuracy,
        "timing": make_timing,
        "traffic": make_traffic,
        "energy": make_energy,
        "adaptation": make_adaptation,
    }
    return makers[group](**kwargs)


def make_builder(
    groups=ALWAYS_GROUPS, *, declare_totals=True, overrides=None
) -> DeploymentRecordBuilder:
    builder = DeploymentRecordBuilder()
    overrides = overrides or {}
    for group in groups:
        fragment = overrides.get(group, _fragment(group))
        builder.attach(group, fragment, make_provenance())
    if declare_totals:
        builder.declare_weight_programming_totals(
            params_programmed=PARAMS_PROGRAMMED_TOTAL
        )
    return builder


def test_minimal_plan_seals():
    record = make_builder().seal(PlanView(), ["Hard Core Mapping"])
    assert isinstance(record, DeploymentRecord)
    assert record.traffic is None
    assert record.energy is None
    assert record.adaptation is None
    assert set(record.provenance) == set(ALWAYS_GROUPS)


@pytest.mark.parametrize("missing", ALWAYS_GROUPS)
def test_each_always_required_fragment_enforced(missing):
    builder = make_builder([g for g in ALWAYS_GROUPS if g != missing])
    with pytest.raises(ValueError, match=f"required fragment '{missing}'"):
        builder.seal(PlanView(), [])


def test_double_attach_raises():
    builder = make_builder()
    with pytest.raises(ValueError, match="'schedule' attached twice"):
        builder.attach("schedule", make_schedule(), make_provenance())


def test_unknown_group_raises():
    with pytest.raises(ValueError, match="unknown fragment group 'physics'"):
        DeploymentRecordBuilder().attach("physics", make_timing(), make_provenance())


def test_wrong_fragment_type_raises():
    with pytest.raises(TypeError, match="'schedule' requires ScheduleRecord"):
        DeploymentRecordBuilder().attach("schedule", make_timing(), make_provenance())


def test_missing_provenance_raises():
    builder = DeploymentRecordBuilder()
    for group in ALWAYS_GROUPS:
        provenance = None if group == "utilization" else make_provenance()
        builder.attach(group, _fragment(group), provenance)
    builder.declare_weight_programming_totals(params_programmed=PARAMS_PROGRAMMED_TOTAL)
    with pytest.raises(ValueError, match=r"\['utilization'\] have no provenance"):
        builder.seal(PlanView(), [])


def _sanafe_builder(overrides=None) -> DeploymentRecordBuilder:
    return make_builder(ALWAYS_GROUPS + ("energy", "traffic"), overrides=overrides)


SANAFE = PlanView(enable_sanafe_simulation=True)


def test_sanafe_plan_seals_when_complete():
    record = _sanafe_builder().seal(SANAFE, [])
    assert record.energy is not None
    assert record.traffic is not None and record.traffic.noc is not None
    assert record.placement.floorplan is not None
    assert record.timing.per_segment


def test_sanafe_requires_energy():
    builder = make_builder(ALWAYS_GROUPS + ("traffic",))
    with pytest.raises(ValueError, match="required fragment 'energy'"):
        builder.seal(SANAFE, [])


def test_sanafe_requires_per_segment_timing():
    builder = _sanafe_builder(overrides={"timing": make_timing(with_per_segment=False)})
    with pytest.raises(ValueError, match="timing.per_segment empty"):
        builder.seal(SANAFE, [])


def test_sanafe_requires_noc_traffic():
    builder = _sanafe_builder(overrides={"traffic": make_traffic(with_noc=False)})
    with pytest.raises(ValueError, match=r"traffic\.noc missing"):
        builder.seal(SANAFE, [])


def test_sanafe_requires_floorplan():
    builder = _sanafe_builder(
        overrides={"placement": make_placement(with_floorplan=False)}
    )
    with pytest.raises(ValueError, match=r"placement\.floorplan missing"):
        builder.seal(SANAFE, [])


def test_counts_gate_requires_boundaries():
    plan = PlanView(counts_observable=True, spike_count_gate_armed=True)
    with pytest.raises(ValueError, match="required fragment 'traffic'"):
        make_builder().seal(plan, [])
    builder = make_builder(
        ALWAYS_GROUPS + ("traffic",),
        overrides={"traffic": make_traffic(with_boundaries=False)},
    )
    with pytest.raises(ValueError, match=r"traffic\.boundaries missing"):
        builder.seal(plan, [])


def test_counts_without_armed_gate_requires_nothing():
    record = make_builder().seal(PlanView(counts_observable=True), [])
    assert record.traffic is None


def test_nevresim_requires_nevresim_read():
    builder = make_builder(overrides={"accuracy": make_accuracy(with_nevresim=False)})
    with pytest.raises(ValueError, match="no 'nevresim' read"):
        builder.seal(PlanView(nevresim_applies=True), [])
    make_builder().seal(PlanView(nevresim_applies=True), [])


def test_tuner_steps_require_adaptation():
    plan = PlanView(tuner_hosting_step_names=frozenset({"LIF Adaptation"}))
    with pytest.raises(ValueError, match="required fragment 'adaptation'"):
        make_builder().seal(plan, ["Pretraining", "LIF Adaptation"])
    record = make_builder().seal(plan, ["Pretraining"])
    assert record.adaptation is None
    record = make_builder(ALWAYS_GROUPS + ("adaptation",)).seal(
        plan, ["Pretraining", "LIF Adaptation"]
    )
    assert record.adaptation is not None


def test_pass_count_cross_check():
    builder = make_builder(
        overrides={"utilization": make_utilization(schedule_pass_count=5)}
    )
    with pytest.raises(ValueError, match="cross-check pass_count"):
        builder.seal(PlanView(), [])


def test_params_programmed_cross_check():
    builder = make_builder(declare_totals=False)
    builder.declare_weight_programming_totals(
        params_programmed=PARAMS_PROGRAMMED_TOTAL + 1
    )
    with pytest.raises(ValueError, match="cross-check params_programmed"):
        builder.seal(PlanView(), [])


def test_undeclared_weight_programming_totals_raises():
    builder = make_builder(declare_totals=False)
    with pytest.raises(ValueError, match="never declared"):
        builder.seal(PlanView(), [])


def test_double_declared_weight_programming_totals_raises():
    builder = make_builder()
    with pytest.raises(ValueError, match="declared twice"):
        builder.declare_weight_programming_totals(params_programmed=1)


def test_cores_allocated_cross_check():
    builder = make_builder(
        overrides={"utilization": make_utilization(cores_allocated=99)}
    )
    with pytest.raises(ValueError, match="cross-check cores_allocated"):
        builder.seal(PlanView(), [])


def test_sealed_record_roundtrips():
    record = _sanafe_builder().seal(
        dataclasses.replace(SANAFE, nevresim_applies=True), []
    )
    assert DeploymentRecord.from_dict(record.to_dict()) == record
