"""[U-S S-2] The record seals a declared event census as ``EnergyRecord.synaptic_events``; nothing moves without it.

Without the declaration the energy fragment and its provenance are SANA-FE's, as sealed before (or absent on a run
without SANA-FE), and the legacy ``cost_record.json`` is written exactly when SANA-FE ran. With it:

- on a run without SANA-FE, the energy fragment carries the census alone: ``synaptic_events`` (the mean per
  inference) and ``sample_count``, the SANA-FE measured fields None, never a zero. The record plane's energy axis
  then prices from it, while the measured cost plane, the SANA-FE objectives and the legacy projection refuse or
  stay absent by name;
- on a run with SANA-FE, SANA-FE's measured fragment keeps every field but ``synaptic_events``, which the census
  replaces; the provenance names both;
- the seal requires the census in the energy fragment, so a declared census can never go missing silently."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mimarsinan.chip_simulation.cost_extraction import COST_RECORD_FILENAME
from mimarsinan.deployment_record.build.builder import DeploymentRecordBuilder
from mimarsinan.deployment_record.build.from_simulators import energy_record_from_sanafe
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.legacy_projection import cost_record_from_deployment_record
from mimarsinan.deployment_record.cost.model import DeploymentCostModel
from mimarsinan.deployment_record.objectives.extractors import energy_field
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.schema import (
    DEPLOYMENT_RECORD_FILENAME,
    ComputePartitionRecord,
    DeploymentRecord,
    EnergyRecord,
    Provenance,
    load_deployment_record,
)
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import energy_with_event_census
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_step import DeploymentRecordStep

from unit.deployment_record.record_fixtures import PlanView, make_full_record, make_sanafe_snapshot
from unit.pipelining.pipeline_steps import test_deployment_record_step as step_fixtures

CENSUS = {"samples": 3, "split": "validation", "per_sample": [10, 12, 14], "mean_per_inference": 12.0,
          "simulation_steps": 4, "spiking_mode": "lif", "cells_used": 100, "activity": 0.03}
SANAFE_PROVENANCE = Provenance(kind="measured", producer="SANA-FE energy trace", step="SANA-FE Simulation", detail="")


def _census_only() -> EnergyRecord:
    energy, _ = energy_with_event_census(None, CENSUS)
    assert energy is not None
    return energy


def _with_partition(record: DeploymentRecord) -> DeploymentRecord:
    return replace(record, utilization=replace(record.utilization, partition=ComputePartitionRecord(
        onchip_params=900, host_params=0, total_params=900, onchip_macs=68096, host_macs=0, total_macs=68096)))


class TestNothingMovesWithoutTheDeclaration:
    def test_sanafes_fragment_and_provenance_pass_through_unchanged(self):
        energy = energy_record_from_sanafe(make_sanafe_snapshot())
        sealed, provenance = energy_with_event_census(energy, None)
        assert sealed is energy
        assert provenance == SANAFE_PROVENANCE

    def test_no_simulator_and_no_census_seals_no_energy(self):
        assert energy_with_event_census(None, None)[0] is None


class TestTheCensusOnlyFragment:
    def test_it_carries_the_mean_and_count_and_no_sanafe_figure(self):
        energy = _census_only()
        assert energy.synaptic_events == 12.0 and energy.sample_count == 3
        assert energy.total_energy_mj is None and energy.mj_per_sample is None
        assert energy.total_spikes is None and energy.energy_proxy_neuron_steps is None
        assert energy.breakdown == ()
        assert EnergyRecord.from_dict(energy.to_dict()) == energy

    def test_its_provenance_names_the_census_and_its_inputs(self):
        _, provenance = energy_with_event_census(None, CENSUS)
        assert provenance.kind == "measured" and provenance.step == "Event Census"
        assert "census" in provenance.producer and "3 validation samples" in provenance.detail

    def test_the_record_plane_prices_energy_from_it(self):
        record = _with_partition(replace(make_full_record(), energy=_census_only()))
        quantities = from_record(record)
        assert quantities.get("synaptic_events").value == 12.0
        assert quantities.get("synaptic_events").provenance == "measured"
        assert not quantities.has("total_spikes")
        pricing = price_absolute(quantities, get_platform_physics("loihi"))
        term = {t.name: t for t in pricing.terms}["energy_per_inference_mj"]
        assert term.value is not None and term.value > 0.0

    def test_the_sanafe_planes_refuse_or_stay_absent_by_name(self):
        record = replace(make_full_record(), energy=_census_only())
        with pytest.raises(ValueError, match="event census"):
            DeploymentCostModel().energy(record)
        with pytest.raises(ValueError, match="event census"):
            cost_record_from_deployment_record(record)
        view = SimpleNamespace(record=record)
        assert energy_field("mj_per_sample").reader(view) is None
        assert energy_field("total_spikes").reader(view) is None


class TestBesideSanafe:
    def test_the_census_replaces_only_synaptic_events(self):
        energy = energy_record_from_sanafe(make_sanafe_snapshot())
        sealed, provenance = energy_with_event_census(energy, CENSUS)
        assert sealed is not None and sealed.synaptic_events == 12.0
        assert replace(sealed, synaptic_events=energy.synaptic_events) == energy
        assert provenance.step == "SANA-FE Simulation" and "census" in provenance.producer
        assert "3 validation samples" in provenance.detail


class TestTheSeal:
    def _builder(self, energy):
        record = make_full_record()
        builder = DeploymentRecordBuilder()
        for group in ("identity", "schedule", "placement", "utilization", "accuracy", "timing"):
            builder.attach(group, getattr(record, group), Provenance("derived", "fixture", "test"))
        if energy is not None:
            builder.attach("energy", energy, Provenance("measured", "fixture", "test"))
        builder.declare_weight_programming_totals(
            params_programmed=sum(s.params_programmed for s in record.schedule.segments()))
        return builder

    def test_a_declared_census_must_reach_the_energy_fragment(self):
        with pytest.raises(ValueError, match="event census"):
            self._builder(None).seal(PlanView(event_census=True), [])
        no_events = replace(_census_only(), synaptic_events=None)
        with pytest.raises(ValueError, match="synaptic_events"):
            self._builder(no_events).seal(PlanView(event_census=True), [])

    def test_the_sealed_census_passes(self):
        record = self._builder(_census_only()).seal(PlanView(event_census=True), [])
        assert record.energy is not None and record.energy.synaptic_events == 12.0


class TestTheRecordStep:
    def _run(self, tmp_path, *, sanafe: bool):
        pipeline, _ = step_fixtures._build(tmp_path, sanafe=sanafe)
        pipeline.config["event_census_samples"] = CENSUS["samples"]
        pipeline.seed("event_census", dict(CENSUS))
        step = DeploymentRecordStep(pipeline)
        pipeline.prepare_step(step)
        step.run()
        record = load_deployment_record(os.path.join(pipeline.working_directory, DEPLOYMENT_RECORD_FILENAME))
        return pipeline, record

    def test_without_sanafe_the_record_seals_the_census_and_writes_no_cost_record(self, tmp_path):
        pipeline, record = self._run(tmp_path, sanafe=False)
        assert record.energy is not None and record.energy.synaptic_events == 12.0
        assert record.energy.total_energy_mj is None
        assert record.provenance["energy"].step == "Event Census"
        assert not os.path.exists(os.path.join(pipeline.working_directory, COST_RECORD_FILENAME))

    def test_beside_sanafe_the_record_keeps_its_energy_and_writes_its_cost_record(self, tmp_path):
        pipeline, record = self._run(tmp_path, sanafe=True)
        assert record.energy is not None and record.energy.synaptic_events == 12.0
        assert record.energy.mj_per_sample == 2.0
        assert os.path.exists(os.path.join(pipeline.working_directory, COST_RECORD_FILENAME))
