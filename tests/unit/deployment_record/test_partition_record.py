"""The on-chip/host split as a sealed fact (§8b item 4) — additive, never breaking."""

import json

import pytest

from mimarsinan.deployment_record.schema import (
    ComputePartitionRecord,
    UtilizationRecord,
)

from unit.deployment_record.record_fixtures import make_utilization


def _partition(**over):
    kwargs = dict(
        onchip_params=90, host_params=10, total_params=100,
        onchip_macs=900, host_macs=100, total_macs=1000,
    )
    kwargs.update(over)
    return ComputePartitionRecord(**kwargs)


def test_partition_round_trips_through_json():
    partition = _partition()
    payload = json.loads(json.dumps(partition.to_dict()))
    assert ComputePartitionRecord.from_dict(payload) == partition


def test_partition_rejects_unknown_fields():
    payload = _partition().to_dict()
    payload["surprise"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        ComputePartitionRecord.from_dict(payload)


def test_partition_fractions_are_defined_and_guard_zero_totals():
    partition = _partition()
    assert partition.param_fraction == pytest.approx(0.9)
    assert partition.mac_fraction == pytest.approx(0.9)
    empty = _partition(onchip_params=0, host_params=0, total_params=0)
    assert empty.param_fraction == 0.0


def test_utilization_carries_the_partition_when_attached():
    utilization = UtilizationRecord(
        crossbar=make_utilization().crossbar,
        layout=make_utilization().layout,
        relay_cores_inserted=1,
        partition=_partition(),
    )
    payload = json.loads(json.dumps(utilization.to_dict()))
    restored = UtilizationRecord.from_dict(payload)
    assert restored.partition == _partition()


def test_a_pre_partition_utilization_json_still_loads():
    """The additive-optional contract: old sealed records lack the field entirely and
    must load with partition=None — no format-version bump, no migration."""
    legacy = make_utilization().to_dict()
    legacy.pop("partition", None)
    restored = UtilizationRecord.from_dict(legacy)
    assert restored.partition is None


def test_the_fixture_default_stays_partitionless():
    """Every existing seal-matrix/roundtrip test keeps its exact inputs."""
    assert make_utilization().partition is None
