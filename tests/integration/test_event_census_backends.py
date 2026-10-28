"""[U-S S-3] The executed event census against SANA-FE's and nevresim's, on tier-0 LIF programs (built SANA-FE, g++).

The tier-0 templates' programs at random weights, activated to fire (``templates.activate``, the residency tests'
vehicles), three seeded samples each. The census of the Hard Core Mapping run record (``census_from_run_record``:
per core, the recorded input counts times the used columns) equals:

- SANA-FE's sealed census (``SanafeRunRecord.total_synaptic_events``: its trace groups' arrivals times occupied
  columns, ``census_from_trace_groups``), stage by stage, on t0_45 (streamed, one segment), t0_50 (streamed LeNet-5,
  three segments between host ops) and t0_05 (synchronized, per-hop level stages);
- on t0_45, the census of nevresim's ``SPKREC`` record of the same program and segment input, joined through the
  program's axon spans (``synaptic_event_census``: every producer's window emissions OUT, and the input and
  always-on arrivals IN of the rows that read them).

nevresim's IN of a core-sourced row is not the signal the core integrated: the recorder samples
``get_axon_input`` after the cycle's compute (``spiking_execution.hpp``), so it reads the producer's next output. Its
OUT agrees neuron for neuron; the consumer-side reading of IN is pinned strict-xfail (finding of this unit)."""

from __future__ import annotations

import contextlib
import io
import tempfile

import numpy as np
import pytest
import torch

from integration.parity_harness import ensure_nevresim_ready, ensure_sanafe
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import enumerate_execution_stages
from mimarsinan.chip_simulation.synaptic_events import (
    census_by_unit,
    census_from_run_record,
    synaptic_event_census,
)
from mimarsinan.pipelining.core.simulation_factory import (
    build_deployment_contract,
    build_hybrid_mapping_for_pipeline,
    build_spiking_hybrid_flow,
)
from unit.mapping.residency.executed import templates as TV

pytestmark = [pytest.mark.slow, pytest.mark.integration]

GAIN = 32.0
SAMPLES = 3


def _program(name: str):
    vehicle = TV.template_vehicle(name)
    config = vehicle.config
    TV.activate(vehicle.ir, GAIN, bits=int(config.get("weight_bits", 8)) if config.get("weight_quantization") else 0)
    with contextlib.redirect_stdout(io.StringIO()):
        mapping = build_hybrid_mapping_for_pipeline(vehicle.ir, config, pipeline_config=config)
    flow = build_spiking_hybrid_flow(vehicle.pipeline(), mapping).eval()
    xs = torch.rand(SAMPLES, *config["input_shape"], generator=torch.Generator().manual_seed(0))
    return vehicle, mapping, flow, xs


def neural_execution_units(mapping):
    return [(i, unit) for i, _stage, unit in enumerate_execution_stages(mapping.stages) if unit.kind == "neural"]


def _records(flow, xs):
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        return flow.forward_with_recordings(xs)[1]


@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", ["t0_45", "t0_50", "t0_05"])
def test_the_census_equals_sanafes_stage_by_stage(name):
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    ensure_sanafe()
    vehicle, mapping, flow, xs = _program(name)
    T = int(vehicle.config["simulation_steps"])
    runner = SanafeRunner(mapping=mapping, simulation_length=T, contract=build_deployment_contract(vehicle.pipeline()))
    for b, record in enumerate(_records(flow, xs)):
        with contextlib.redirect_stdout(io.StringIO()):
            sanafe = runner.run(xs[b:b + 1].numpy().reshape(1, -1), sample_index=b)
        events = census_from_run_record(record, mapping)
        assert events > 0
        assert events == sanafe.total_synaptic_events
        assert sorted(sanafe.segments) == [index for index, _ in neural_execution_units(mapping)]
        assert census_by_unit(record, mapping) == {i: s.synaptic_events for i, s in sanafe.segments.items()}


def _nevresim_records(vehicle, mapping, records):
    """nevresim's SPKREC of the program's one neural unit, per sample, on the segment input the HCM run read."""
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.mapping.latency.chip import ChipLatency

    ensure_nevresim_ready()
    (index, unit), = neural_execution_units(mapping)
    segment = unit.hard_core_mapping
    config = vehicle.config
    T = int(config["simulation_steps"])
    out = []
    for record in records:
        rates = record.segments[index].seg_input_rates
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            driver = NevresimDriver(rates.shape[1], segment, tmp,
                                    int if config.get("weight_quantization") else float,
                                    spike_generation_mode=config["spike_generation_mode"],
                                    firing_mode=config["firing_mode"], thresholding_mode=config["thresholding_mode"],
                                    spiking_mode=config["spiking_mode"], verbose=False)
            _raw, spk = driver.predict_spiking_raw_with_records([(rates[0], np.zeros(1))], T,
                                                                ChipLatency(segment).calculate())
        out.append(spk[0])
    return index, segment, out


def _boundary_arrivals(core, spkrec_in) -> int:
    rows = [(span.dst_start, span.dst_end) for span in core.get_axon_source_spans() if span.kind in ("input", "on")]
    return int(sum(int(np.sum(spkrec_in[lo:hi])) for lo, hi in rows))


@pytest.mark.timeout(900)
def test_the_census_equals_nevresims_spkrec_joined_through_the_spans():
    vehicle, mapping, flow, xs = _program("t0_45")
    records = _records(flow, xs)
    index, segment, spkrecs = _nevresim_records(vehicle, mapping, records)
    for record, spk in zip(records, spkrecs):
        events = census_from_run_record(record, mapping)
        assert events > 0
        assert events == synaptic_event_census(
            segment.cores,
            emissions_of=lambda c: np.asarray(spk[c]["out"], dtype=np.int64) if c in spk else None,
            boundary_arrivals_of=lambda c: _boundary_arrivals(segment.cores[c], spk[c]["in"]),
        )
        for core in record.segments[index].cores:     # the emissions agree neuron for neuron
            used = len(core.output_spike_count)
            assert np.array_equal(np.asarray(spk[core.core_index]["out"][:used]), core.output_spike_count)


@pytest.mark.timeout(900)
@pytest.mark.xfail(strict=True, reason="U-S finding: nevresim's SPKREC IN samples a core-sourced row after the "
                   "cycle's compute (spiking_execution.hpp), one cycle late; the census joins its OUT instead")
def test_nevresims_spkrec_in_is_the_signal_each_core_integrated():
    vehicle, mapping, flow, xs = _program("t0_45")
    records = _records(flow, xs)
    index, _segment, spkrecs = _nevresim_records(vehicle, mapping, records)
    for record, spk in zip(records, spkrecs):
        for core in record.segments[index].cores:
            used = len(core.input_spike_count)
            assert np.array_equal(np.asarray(spk[core.core_index]["in"][:used]), core.input_spike_count)
