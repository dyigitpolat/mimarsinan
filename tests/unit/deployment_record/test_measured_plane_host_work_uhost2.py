"""[U-HOST2] The measured cost plane refuses host work it does not price by the absolute plane's refusal (verify m-6).

The measured plane's total is the chip side: it prices no host row, and the development machine's wall is a
diagnostic. On a record that holds host work, a host-inclusive total there would make moving work host-side look free
(the subsume rig the absolute plane refuses by name). So where the record's host work is unpriced the plane withholds
its total and its throughput and names the absolute plane's own refusal, judged by the same function
(``hosts.host_work``); host work the absolute plane names unpriced without refusing (dispatch only) is named the same
way beside the total; a record with no host ComputeOp keeps its total.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mimarsinan.deployment_record.cost import DeploymentCostModel
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.objectives import OBJECTIVES, DeploymentRecordView
from mimarsinan.deployment_record.platform_physics import available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
from mimarsinan.deployment_record.schema import ComputeOpRecord
from mimarsinan.deployment_record.schema.utilization import ComputePartitionRecord

from unit.deployment_record.g3_fixtures import quantities
from unit.deployment_record.record_fixtures import make_full_record


def _with_partition(record, host_macs):
    partition = ComputePartitionRecord(onchip_params=100, host_params=10, total_params=110, onchip_macs=1000,
                                       host_macs=host_macs, total_macs=1000 + host_macs)
    return replace(record, utilization=replace(record.utilization, partition=partition))


def _without_host_ops(record):
    stages = tuple(s for s in record.schedule.stages if not isinstance(s, ComputeOpRecord))
    return replace(record, schedule=replace(record.schedule, stages=stages, compute_op_count=0))


def _absolute_refusal(**view):
    """The absolute plane's refusal of its host-inclusive headline on a physics that declares no host row."""
    name = next(p for p in available_profiles()
                if not [k for k in get_platform_physics(p).constants if spec_for(k).group == HOST])
    pricing = price_absolute(quantities(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4,
                                        **view), get_platform_physics(name))
    return next(r.reason for r in pricing.refusals if r.name == "e2e_latency_s")


def _names(terms):
    return {t.name for t in terms}


class TestHostWorkTheMeasuredPlaneDoesNotPrice:
    def test_host_macs_withhold_the_total_and_the_throughput_by_the_absolute_planes_refusal(self):
        record = _with_partition(make_full_record(), host_macs=5000)
        report = DeploymentCostModel().evaluate(record)
        assert "total_s" not in _names(report.latency) and "samples_per_s" not in _names(report.throughput)
        reason = _absolute_refusal(host_macs=5000, compute_op_count=1)
        assert any(reason in note for note in report.notes), (reason, report.notes)
        assert "throughput_samples_per_s" not in {s.key for s in OBJECTIVES.available_for(
            DeploymentRecordView(record=record))}

    def test_an_unknown_host_share_withholds_them_by_the_same_refusal(self):
        record = make_full_record()
        assert record.utilization.partition is None
        report = DeploymentCostModel().evaluate(record)
        assert "total_s" not in _names(report.latency)
        reason = _absolute_refusal(compute_op_count=1)
        assert any(reason in note for note in report.notes), (reason, report.notes)

    def test_dispatch_only_host_work_keeps_the_total_and_names_it_unpriced(self):
        record = _with_partition(make_full_record(), host_macs=0)
        report = DeploymentCostModel().evaluate(record)
        assert "total_s" in _names(report.latency) and "samples_per_s" in _names(report.throughput)
        assert any("unpriced" in note and "t_host_op_overhead" in note for note in report.notes), report.notes

    def test_a_record_without_host_work_keeps_its_total(self):
        record = _without_host_ops(make_full_record())
        report = DeploymentCostModel().evaluate(record)
        assert "total_s" in _names(report.latency) and "samples_per_s" in _names(report.throughput)
        assert OBJECTIVES.get("throughput_samples_per_s").value(DeploymentRecordView(record=record)) == pytest.approx(
            next(t.value for t in report.throughput if t.name == "samples_per_s"))
