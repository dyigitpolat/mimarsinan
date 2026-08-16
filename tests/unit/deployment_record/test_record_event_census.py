"""[H1] The sealed arrival census makes record-plane energy priceable.

Before this, ``energy_per_inference_mj`` REFUSED on every sealed record ("the
synaptic_events census is absent") — the candidate's modeled energy was never
checked against a deployment. The census rides the energy fragment as an
additive-optional field, so records sealed before it load unchanged and stay
honestly absent.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mimarsinan.deployment_record.build.from_simulators import (
    energy_record_from_sanafe,
)
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.schema import DeploymentRecord

from unit.deployment_record.record_fixtures import make_full_record


def _with_events(record: DeploymentRecord, events) -> DeploymentRecord:
    assert record.energy is not None
    return replace(record, energy=replace(record.energy, synaptic_events=events))


class TestTheClaim:
    def test_a_sealed_census_is_claimed_measured(self):
        quantities = from_record(_with_events(make_full_record(), 21577.0))
        assert quantities.get("synaptic_events").value == 21577.0
        assert quantities.get("synaptic_events").provenance == "measured"

    def test_a_pre_census_record_stays_honestly_absent(self):
        """The E-series law: absence survives, never a zero."""
        quantities = from_record(make_full_record())
        assert not quantities.has("synaptic_events")

    def test_an_old_record_json_loads_without_the_field(self):
        data = make_full_record().to_dict()
        data["energy"].pop("synaptic_events", None)
        record = DeploymentRecord.from_dict(data)
        assert record.energy is not None
        assert record.energy.synaptic_events is None


class TestRecordPlaneEnergyGoesLive:
    def test_the_energy_headline_prices_on_a_censused_record(self):
        """The hole this stage closes: measured multiplicand x the same
        constants the candidate priced with — the middle of the triangle.
        The record needs its partition census too (the host share), which
        every real emission seals; the fixture gains one here."""
        from dataclasses import replace as _replace

        from mimarsinan.deployment_record.schema import ComputePartitionRecord

        record = _with_events(make_full_record(), 21577.0)
        record = _replace(record, utilization=_replace(
            record.utilization,
            partition=ComputePartitionRecord(
                onchip_params=900, host_params=0, total_params=900,
                onchip_macs=68096, host_macs=0, total_macs=68096,
            ),
        ))
        quantities = from_record(record)
        pricing = price_absolute(quantities, get_platform_physics("loihi"))
        term = {t.name: t for t in pricing.terms}["energy_per_inference_mj"]
        assert term.value is not None and term.value > 0.0
        assert "compute: e_mac" in term.source
        assert not any(r.name == "energy_per_inference_mj"
                       for r in pricing.refusals)

    def test_the_dynamic_term_needs_no_host_declaration(self):
        """Switching energy stands alone: the census alone lights it up."""
        quantities = from_record(_with_events(make_full_record(), 21577.0))
        pricing = price_absolute(quantities, get_platform_physics("loihi"))
        term = {t.name: t for t in pricing.terms}["energy_dynamic_mj"]
        assert term.value is not None and term.value > 0.0

    def test_a_pre_census_record_still_refuses_by_name(self):
        pricing = price_absolute(
            from_record(make_full_record()), get_platform_physics("loihi"))
        reasons = {r.name: r.reason for r in pricing.refusals}
        assert "synaptic_events" in reasons["energy_per_inference_mj"]


class TestTheSnapshotThreading:
    def _snapshot(self, events, samples=1):
        return {
            "aggregate": {
                "total_energy_mj": 2.0, "sample_count": samples,
                "energy_breakdown_j": {}, "total_spikes": 40,
                "total_synaptic_events": events,
            },
            "per_sample": [{"segments": [
                {"stage_index": 0, "timesteps_executed": 5, "sim_time_s": 0.1,
                 "per_core": []},
            ]} for _ in range(samples)],
        }

    def test_multi_sample_totals_seal_the_per_inference_mean(self):
        """The same rule ``mj_per_sample`` follows — one estimand family."""
        record = energy_record_from_sanafe(self._snapshot(3000, samples=4))
        assert record.synaptic_events == pytest.approx(750.0)

    def test_a_refused_stage_seals_no_census(self):
        record = energy_record_from_sanafe(self._snapshot(None, samples=2))
        assert record.synaptic_events is None
