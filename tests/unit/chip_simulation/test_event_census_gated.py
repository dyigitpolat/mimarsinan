"""[U-LIF2 W13] The event census of integrate then emit comes from the executor of record.

The census (U-S) reads the Hard Core Mapping executor's spike record: per core, the arrivals it integrated, times its
used columns. Under integrate then emit the record is the gated executor's, so each core's recorded arrivals are its
window's input counts (a producer's emitted count, the stage input's count, T on an always-on axon), the census is those
counts times the fan-out, and the record seals it as the energy fragment's synaptic events with no external simulator,
so the energy headline is priced from it. The flows state integrate then emit explicitly (their two-window flag)."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.chip_simulation.synaptic_events import census_by_unit, census_from_run_record
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import energy_with_event_census

from unit.models import gated_window_programs as G


def _closed_form_arrivals(flow, stage, train, closed):
    """Per core and axon, the window's input counts the closed form integrates (on-axons at T)."""
    seg = flow._get_segment_tensors(stage, train.device)
    out = {}
    for i, core in enumerate(seg["cores"]):
        if core.latency is None:
            continue
        window = torch.zeros(train.shape[1], max(int(core.axons_per_core - core.available_axons), 1),
                             dtype=COMPUTE_DTYPE)
        empty = torch.zeros(train.shape[1], 1, dtype=COMPUTE_DTYPE)
        buffers = [closed.get(k, empty) for k in range(len(seg["cores"]))]
        seg["axon_fill_plans"][i].apply(window, input_spikes=train.sum(0).to(COMPUTE_DTYPE), buffers=buffers,
                                        on_value=float(flow.simulation_length))
        out[i] = window
    return out


@pytest.mark.parametrize("retimed", [False, True], ids=["fused", "retimed"])
@pytest.mark.parametrize("hardware_bias", [True, False], ids=["hardware_bias", "on_axon_bias"])
def test_w13_the_census_of_an_integrate_then_emit_run_is_its_window_counts_times_the_fan_out(retimed, hardware_bias):
    instance = G.layered_instance(G.INPUT_AT_DEPTH, timesteps=6, seed=41, batch=1, hardware_bias=hardware_bias)
    mapping = G.identity_program(instance, retimed=retimed)
    flow = G.ite_flow(instance, mapping)
    runs = G.stage_runs(flow, G.inputs(instance))
    with torch.no_grad():
        _out, record = flow.forward_with_recording(G.inputs(instance))
    want = 0
    stage_of = {stage.name: (stage, train) for stage, train in runs}
    for segment in record.segments.values():
        stage, train = stage_of[segment.stage_name]
        closed, _ = G.window_counts(flow, stage, train, "closed_form")
        arrivals = _closed_form_arrivals(flow, stage, train, closed)
        for core in segment.cores:
            assert core.output_spike_count.tolist() == closed[core.core_index][0].to(torch.int64).tolist()
            assert core.input_spike_count.tolist() == arrivals[core.core_index][0].to(torch.int64).tolist()
            columns = CoreOccupancy.from_hard_core(stage.hard_core_mapping.cores[core.core_index]).neurons_used
            want += int(arrivals[core.core_index].sum()) * int(columns)
    assert census_from_run_record(record, mapping) == want > 0
    assert sum(census_by_unit(record, mapping).values()) == want


def test_w13_the_record_seals_the_census_and_prices_the_energy_without_an_external_simulator():
    instance = G.layered_instance(G.SKIP, timesteps=6, seed=42, batch=3)
    mapping = G.identity_program(instance, retimed=False)
    flow = G.ite_flow(instance, mapping)
    with torch.no_grad():
        _out, records = flow.forward_with_recordings(G.inputs(instance))
    per_sample = [census_from_run_record(record, mapping) for record in records]
    census = {"samples": len(per_sample), "split": "validation", "simulation_steps": instance.timesteps,
              "mean_per_inference": sum(per_sample) / len(per_sample)}
    energy, provenance = energy_with_event_census(None, census)
    assert energy is not None and energy.synaptic_events == census["mean_per_inference"] > 0
    assert energy.total_energy_mj is None and "SANA-FE" not in provenance.producer
    def dynamic(events):
        priced = price_absolute(Quantities({"synaptic_events": QuantityValue(events, "measured")}),
                                get_platform_physics("odin"))
        return [term.value for term in priced.terms if term.name == "energy_dynamic_mj"]

    (one,), (two,) = dynamic(energy.synaptic_events), dynamic(2 * energy.synaptic_events)
    assert two > one > 0                              # priced from the sealed census, no external simulator
    # the gated executor's window counts, never another function's: each sample's census is its closed form's
    for i, record in enumerate(records):
        single = G.ite_flow(instance, mapping)
        (stage, train), = G.stage_runs(single, G.inputs(instance)[i:i + 1])
        closed, _ = G.window_counts(single, stage, train, "closed_form")
        arrivals = _closed_form_arrivals(single, stage, train, closed)
        columns = [CoreOccupancy.from_hard_core(c).neurons_used for c in stage.hard_core_mapping.cores]
        assert per_sample[i] == sum(int(arrivals[k].sum()) * int(columns[k]) for k in arrivals)
