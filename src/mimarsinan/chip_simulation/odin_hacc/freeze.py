"""Freeze a multi-core network into a host-mediated DEPLOYMENT BUNDLE.

The shipped NC=1 bitstream holds ONE ODIN core, so a multi-core network runs as
one host-mediated PASS per core: program the core, run the sample, read the
counts back, transcode them into the next core's axon slots, run again.

FOUR GOLDEN GATES, and no bundle is written unless all four hold:
  1. the witness's counts reproduce the cycle-accurate twin's at EVERY cycle of
     every sample — the frozen expectations are MEASURED, not asserted;
  2. every stimulus the shipped bundle reader builds is byte-identical to the
     one the repository's own encoder builds, per core and per sample;
  3. the capture of one pass fits the shipped fabric's capture RAM;
  4. the program plus its stimulus fits an NC=1 kernel's program RAM.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle
from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    SHIPPED_CAPTURE_EVENTS,
)
from mimarsinan.chip_simulation.odin_hacc.pass_build import PassBuild, used_neurons
from mimarsinan.chip_simulation.odin_hacc.witness import PassWitness
from mimarsinan.chip_simulation.odin_rtl.reference import simulate_cycles

#: The replay document a fake pyxrt is armed with; sealed like the bundle.
CAPTURE_SCHEMA = "odin_hacc_deployment_capture/1"


class BundleRefusal(RuntimeError):
    """A golden gate did not hold; no bundle is written."""


def _require_witness_matches_twin(name, index, counts, traces) -> None:
    for sample, trace in enumerate(traces):
        for cycle in range(trace.total_cycles):
            want = tuple(int(v) for v in trace.outputs[cycle][index])
            got = tuple(
                int(counts.get((sample, cycle, 0, neuron), 0))
                for neuron in range(len(want)))
            if got != want:
                raise BundleRefusal(
                    f"{name}: pass {index} sample {sample} cycle {cycle} "
                    f"witnessed {got} against the cycle-accurate twin's "
                    f"{want}. The bundle's expectations would not be a "
                    f"measurement of this network")


def _require_reader_agrees(name, build, traces, plan, cycles) -> None:
    for sample, trace in enumerate(traces):
        mirrored = bundle.pass_stimulus_tokens(
            plan, [trace.inputs[cycle][build.index]
                   for cycle in range(trace.total_cycles)],
            sample=0, cycles_per_sample=cycles)
        reference = list(build.reference_stimulus(trace))
        if mirrored != reference:
            raise BundleRefusal(
                f"{name}: the shipped bundle reader builds a DIFFERENT stimulus "
                f"for pass {build.index} sample {sample} than the repository's "
                f"own encoder ({len(mirrored)} vs {len(reference)} tokens). The "
                f"board would stimulate a network nobody assembled")


def _require_capture_fits(name, build, events: int) -> None:
    """The ONE fabric depth a pass can exhaust: the capture RAM.

    The op stream has no fabric depth to exhaust -- it is streamed, never
    stored -- so its length is checked for nothing here on purpose.
    """
    if events >= SHIPPED_CAPTURE_EVENTS:
        raise BundleRefusal(
            f"{name}: pass {build.index} produced {events} capture events "
            f"against the shipped fabric's {SHIPPED_CAPTURE_EVENTS}-record RAM; "
            f"a board run would refuse as truncated")


def cycle_traces(mapping, *, rasters, soma_law, simulation_length, membrane_init,
                 chip_latency) -> List[Any]:
    """One cycle-accurate trace per sample — THE reference every gate compares."""
    return [
        simulate_cycles(
            mapping, soma_law=soma_law, input_counts=raster,
            simulation_length=int(simulation_length),
            membrane_init=int(membrane_init), chip_latency=int(chip_latency))
        for raster in rasters
    ]


def build_bundle(
    *,
    name: str,
    title: str,
    description: str,
    mapping: Any,
    rasters: Sequence[Sequence[Sequence[int]]],
    labels: Sequence[int],
    simulation_length: int,
    chip_latency: int,
    soma_law: Any,
    weight_bits: int,
    effective_max_axons: int,
    weight_sign_granularity: str,
    membrane_init: int,
    readout_core: int,
    certification: Sequence[int],
    provenance: Dict[str, Any],
    kernel_table: Dict[str, Any],
    model: Dict[str, Any],
    witness: PassWitness,
    traces: Sequence[Any] | None = None,
) -> Any:
    """Freeze one network + sample set into a sealed, board-executable bundle."""
    if len(labels) != len(rasters):
        raise BundleRefusal(
            f"{name}: {len(rasters)} sample(s) against {len(labels)} label(s)")
    if provenance.get("derivation") != witness.derivation:
        raise BundleRefusal(
            f"{name}: the provenance claims\n  {provenance.get('derivation')!r}\n"
            f"but the {witness.name!r} witness produced these counts, whose "
            f"honest derivation is\n  {witness.derivation!r}.\nA bundle whose "
            f"provenance names a producer that did not produce it is worse than "
            f"no bundle")
    if traces is None:
        traces = cycle_traces(
            mapping, rasters=rasters, soma_law=soma_law,
            simulation_length=simulation_length, membrane_init=membrane_init,
            chip_latency=chip_latency)
    if len(traces) != len(rasters):
        raise BundleRefusal(
            f"{name}: {len(traces)} trace(s) against {len(rasters)} sample(s)")
    cycles = traces[0].total_cycles
    builds = [
        PassBuild(mapping, index, traces[0], weight_bits=weight_bits,
                  effective_max_axons=effective_max_axons, soma_law=soma_law,
                  weight_sign_granularity=weight_sign_granularity,
                  membrane_init=membrane_init)
        for index in range(len(mapping.cores))
    ]

    cores: List[Dict[str, Any]] = []
    replay: List[Dict[str, Any]] = []
    for build in builds:
        plan = build.plan_document(traces)
        _require_reader_agrees(name, build, traces, plan, cycles)
        measured = witness.measure(build, traces)
        _require_witness_matches_twin(name, build.index, measured.counts, traces)
        per_sample_events = [
            sum(value for (sample, _c, _k, _n), value in measured.counts.items()
                if sample == index)
            for index in range(len(traces))
        ]
        _require_capture_fits(
            name, build, max(per_sample_events, default=0))
        plan["cosim_events_per_sample"] = [int(v) for v in per_sample_events]
        cores.append(plan)
        replay.extend(_replay_runs(build, traces, measured, cycles))

    windows = [
        {index: [int(v) for v in trace.window_counts()[index]]
         for index in range(len(mapping.cores))}
        for trace in traces
    ]
    document = _document(
        name=name, title=title, description=description,
        provenance=dict(provenance),
        kernel_table=kernel_table, model=model, soma_law=soma_law,
        weight_bits=weight_bits, effective_max_axons=effective_max_axons,
        weight_sign_granularity=weight_sign_granularity,
        membrane_init=membrane_init, simulation_length=simulation_length,
        chip_latency=chip_latency, cycles=cycles, builds=builds, cores=cores,
        mapping=mapping, readout_core=readout_core, labels=labels,
        rasters=rasters, traces=traces, certification=certification,
        windows=windows)
    sealed = bundle.seal(document)
    capture = bundle.seal({
        "schema": CAPTURE_SCHEMA,
        "bundle": sealed["name"],
        "bundle_self_hash": sealed["self_hash"],
        "provenance": dict(sealed["provenance"]),
        "note": witness.capture_note,
        "runs": replay,
    })
    return sealed, capture


def _document(*, name, title, description, provenance, kernel_table, model,
              soma_law, weight_bits, effective_max_axons,
              weight_sign_granularity, membrane_init, simulation_length,
              chip_latency, cycles, builds, cores, mapping, readout_core,
              labels, rasters, traces, certification, windows
              ) -> Dict[str, Any]:
    """The unsealed bundle document — one place the schema is written down."""
    document = {
        "schema": bundle.SCHEMA,
        "name": name,
        "title": title,
        "description": description,
        "provenance": dict(provenance),
        "kernel": dict(kernel_table),
        "model": dict(model),
        "chip_config": {
            "soma_law": {
                "firing_mode": str(soma_law.firing_mode),
                "thresholding_mode": str(soma_law.thresholding_mode),
                "firing_granularity": str(soma_law.firing_granularity),
                "membrane_arithmetic": str(soma_law.membrane_arithmetic),
                "membrane_bits": int(soma_law.membrane_bits),
                "bias_slot": str(soma_law.bias_slot),
            },
            "weight_bits": int(weight_bits),
            "effective_max_axons": int(effective_max_axons),
            "weight_sign_granularity": str(weight_sign_granularity),
            "membrane_init": int(membrane_init),
            "simulation_length": int(simulation_length),
            "chip_latency": int(chip_latency),
        },
        "cycles_per_sample": int(cycles),
        "pass_order": [build.index for build in builds],
        "cores": cores,
        "readout": {
            "core": int(readout_core),
            "rule": bundle.READOUT_ARGMAX,
            "neurons": list(range(used_neurons(mapping.cores[readout_core]))),
        },
        "samples": [
            {
                "index": index,
                "label": int(labels[index]),
                "entry_raster": [[int(v) for v in row] for row in rasters[index]],
                "stimulus": bundle.encode_payload(
                    bundle.tokens_to_bytes(builds[0].reference_stimulus(trace))),
            }
            for index, trace in enumerate(traces)
        ],
        "certification": {
            "samples": [int(index) for index in certification],
            "windows": {
                str(int(index)): {str(core): [row]
                                  for core, row in windows[int(index)].items()}
                for index in certification
            },
        },
    }
    document["expected"] = {
        "final": [_final_expectation(document, windows[index], int(labels[index]),
                                    index)
                  for index in range(len(traces))],
    }
    return document


def _replay_runs(build, traces, measured, cycles: int) -> List[Dict[str, Any]]:
    """One replayable capture image per (core, sample), keyed by its stimulus."""
    runs: List[Dict[str, Any]] = []
    for sample, trace in enumerate(traces):
        tokens = build.reference_stimulus(trace)
        events: List[List[int]] = []
        for cycle in range(cycles):
            tag = bundle.tag_of(0, cycle, cycles)
            for neuron in range(build.neurons):
                for _spike in range(int(
                        measured.counts.get((sample, cycle, 0, neuron), 0))):
                    events.append([tag, cycle, 0, int(neuron)])
        runs.append({
            "core": build.index,
            "sample": int(sample),
            "stimulus_sha256": hashlib.sha256(
                bundle.tokens_to_bytes(tokens)).hexdigest(),
            "device_cycles": int(measured.device_cycles),
            "events": events,
        })
    return runs


def _final_expectation(document, window, label: int, index: int) -> Dict[str, Any]:
    scores = bundle.readout_scores(document, window)
    return {
        "sample": int(index),
        "label": int(label),
        "scores": [int(value) for value in scores],
        "predicted": int(bundle.predicted_label(document, scores)),
    }
