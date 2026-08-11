"""Mirror-fidelity: record types stay field-identical to the mapping-report sources."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

from mimarsinan.deployment_record.schema import (
    CrossbarUtilizationRecord,
    LayoutStatsRecord,
)
from mimarsinan.mapping.crossbar_utilization import CrossbarUtilizationReport
from mimarsinan.mapping.verification.layout_verification_types import (
    LayoutVerificationStats,
)


def test_layout_stats_record_mirrors_layout_verification_stats():
    source = [f.name for f in fields(LayoutVerificationStats)]
    mirror = [f.name for f in fields(LayoutStatsRecord)]
    assert mirror == source


def _tiny_report() -> CrossbarUtilizationReport:
    core = SimpleNamespace(
        axons_per_core=256,
        neurons_per_core=256,
        available_axons=156,
        available_neurons=206,
        unusable_space=3,
    )
    return CrossbarUtilizationReport.from_hard_cores([core, core], weight_bits=8)


def test_crossbar_record_mirrors_report_to_dict_keys():
    report_dict = _tiny_report().to_dict()
    mirror = [f.name for f in fields(CrossbarUtilizationRecord)]
    assert mirror == list(report_dict.keys())


def test_crossbar_record_accepts_a_real_report_dict():
    report_dict = _tiny_report().to_dict()
    record = CrossbarUtilizationRecord.from_dict(report_dict)
    assert record.to_dict() == report_dict


def test_layout_stats_record_accepts_a_real_stats_dict():
    stats = LayoutVerificationStats(
        feasible=True,
        total_cores=2,
        total_softcores=2,
        total_hw_cores=2,
        total_wasted_axons_pct=0.0,
        total_wasted_neurons_pct=0.0,
        mapped_params_pct=100.0,
        per_core_wasted_axons_pct_min=0.0,
        per_core_wasted_axons_pct_avg=0.0,
        per_core_wasted_axons_pct_max=0.0,
        per_core_wasted_neurons_pct_min=0.0,
        per_core_wasted_neurons_pct_avg=0.0,
        per_core_wasted_neurons_pct_max=0.0,
        per_core_mapped_params_pct_min=100.0,
        per_core_mapped_params_pct_avg=100.0,
        per_core_mapped_params_pct_max=100.0,
        coalesced_cores=0,
        split_cores=0,
    )
    record = LayoutStatsRecord.from_dict(stats.to_dict())
    assert record.to_dict() == stats.to_dict()
