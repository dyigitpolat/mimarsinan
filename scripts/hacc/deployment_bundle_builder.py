#!/usr/bin/env python3
"""Freeze a multi-core network into a host-mediated DEPLOYMENT BUNDLE.

The shipped NC=1 bitstream holds ONE ODIN core, so a multi-core network runs as
one host-mediated PASS per core: program the core, run the sample, read the
counts back, transcode them into the next core's axon slots, run again. This
module is the repository-side half that FREEZES such a program — it may import
anything; the module it freezes for (``chip_simulation/odin_deployment_bundle``)
may not.

A pass core is the original crossbar with every host-fed axon source rewritten
to an INPUT line: on the chip nothing routes between cores, so a consumer's
slots are delivered by the host either way, and the rewrite is what makes a
one-core segment graph a DAG the exporter's emission bound is defined on.
Always-on rows keep their source, because the exporter injects those itself.

FOUR GOLDEN GATES, and no bundle is written unless all four hold:
  1. the pass cosimulation reproduces the cycle-accurate twin's counts at EVERY
     cycle of every sample — the frozen expectations are MEASURED, not asserted;
  2. every stimulus the shipped bundle reader builds is byte-identical to the
     one the repository's own encoder builds, per core and per sample;
  3. the capture of one pass fits the shipped fabric's capture RAM;
  4. the program plus its stimulus fits an NC=1 kernel's program RAM.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle  # noqa: E402
from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (  # noqa: E402
    SHIPPED_CAPTURE_EVENTS,
    SHIPPED_PROGRAM_WORDS_PER_CORE,
    WORD_BYTES,
    stimulus_base_word,
)
from mimarsinan.chip_simulation.odin_fpga.payload import (  # noqa: E402
    payload_bytes,
    program_plan,
    run_plan,
    stimulus_ops,
)
from mimarsinan.chip_simulation.odin_rtl.cosim import run_cosim  # noqa: E402
from mimarsinan.chip_simulation.odin_rtl.program_ops import (  # noqa: E402
    slot_rows_from_inject,
    stages_of_kind,
)
from mimarsinan.chip_simulation.odin_rtl.reference import (  # noqa: E402
    CycleTrace,
    simulate_cycles,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import OP_TAG, encode_ops  # noqa: E402
from mimarsinan.code_generation.cpp_chip_model import SpikeSource  # noqa: E402
from mimarsinan.mapping.export.odin.exporter import export_odin  # noqa: E402
from mimarsinan.mapping.export.odin.program import STAGE_INJECT  # noqa: E402
from mimarsinan.mapping.latency.chip import ChipLatency  # noqa: E402
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping  # noqa: E402


class BundleRefusal(RuntimeError):
    """A golden gate did not hold; no bundle is written."""


def used_neurons(core: Any) -> int:
    return max(int(core.neurons_per_core) - int(core.available_neurons or 0), 1)


def pass_mapping(core: Any) -> HardCoreMapping:
    """The one-core mapping a pass programs: every host-fed slot is an INPUT."""
    sources = [
        source if (getattr(source, "is_always_on_", False)
                   or getattr(source, "is_off_", False))
        else SpikeSource(-2, slot, is_input=True)
        for slot, source in enumerate(core.axon_sources)
    ]
    values = np.asarray(core.core_matrix, dtype=np.float64)
    replica = HardCore(
        axons_per_core=values.shape[0], neurons_per_core=values.shape[1],
        has_bias_capability=False)
    replica.core_matrix = values
    replica.axon_sources = sources
    replica.threshold = float(core.threshold)
    replica.available_axons = 0
    replica.available_neurons = int(core.available_neurons or 0)
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [replica]
    mapping.output_sources = np.asarray(
        [SpikeSource(0, index) for index in range(used_neurons(core))], dtype=object)
    ChipLatency(mapping).calculate()
    return mapping


class PassBuild:
    """One core's pass: its export, its frozen program, and its stimulus recipe."""

    def __init__(self, mapping: Any, index: int, trace: CycleTrace, *,
                 weight_bits: int, effective_max_axons: int, soma_law: Any,
                 weight_sign_granularity: str, membrane_init: int) -> None:
        self.index = int(index)
        self.core = mapping.cores[index]
        self.latency = int(trace.latencies[index])
        self.neurons = int(self.core.neurons_per_core)
        self.used = used_neurons(self.core)
        self.routes = bundle.core_routes(self.core)
        self.export = export_odin(
            pass_mapping(self.core), soma_law=soma_law, weight_bits=int(weight_bits),
            weight_sign_granularity=str(weight_sign_granularity),
            effective_max_axons=int(effective_max_axons),
            membrane_init=int(membrane_init))
        self.program = program_plan(self.export)
        self.program_bytes = payload_bytes(self.program.ops)
        inject = stages_of_kind(self.export.program, STAGE_INJECT)
        self.slot_rows = slot_rows_from_inject(inject[0])[0] if inject else {}

    def per_cycle(self, trace: CycleTrace) -> List[Dict[int, Tuple[int, ...]]]:
        """The pass's own injection plan: this core's gathered slots, per cycle."""
        return [{0: trace.inputs[cycle][self.index]}
                for cycle in range(trace.total_cycles)]

    def reference_stimulus(self, trace: CycleTrace) -> Tuple[int, ...]:
        """The stimulus the REPOSITORY encoder builds for one sample of this pass."""
        full = run_plan(
            self.export, [self.per_cycle(trace)], latencies=[self.latency])
        return encode_ops(stimulus_ops(self.program, full))

    def plan_document(self, traces: Sequence[CycleTrace]) -> Dict[str, Any]:
        full = run_plan(
            self.export, [self.per_cycle(traces[0])], latencies=[self.latency])
        ops = stimulus_ops(self.program, full)
        head = next(index for index, op in enumerate(ops) if op.code == OP_TAG)
        barrier = full.barrier_cycles
        return {
            "core": self.index,
            "latency": self.latency,
            "neurons": self.neurons,
            "used_neurons": self.used,
            "barrier_cycles": int(barrier),
            "routes": [[int(kind), int(index)] for kind, index in self.routes],
            "slot_rows": [[int(slot), [int(row) for row in rows]]
                          for slot, rows in sorted(self.slot_rows.items())],
            # The GATE-on / CLEAR / GATE-off head of every sample, frozen: it is
            # sample-independent, and the reader appends the cycles to it.
            "clear_prefix": [int(word) for word in encode_ops(ops[:head])[:-1]],
            # The widest stimulus any shipped sample needs, so the executor
            # allocates ONE buffer object per core and rewrites it; a live
            # stimulus over this bound is a transcode that left the twin.
            "max_stimulus_words": max(
                len(self.reference_stimulus(trace)) for trace in traces),
            "program": bundle.encode_payload(self.program_bytes),
        }


def _require_cosim_matches_twin(name: str, index: int, counts, traces) -> None:
    for sample, trace in enumerate(traces):
        for cycle in range(trace.total_cycles):
            want = tuple(int(v) for v in trace.outputs[cycle][index])
            got = tuple(
                int(counts.get((sample, cycle, 0, neuron), 0))
                for neuron in range(len(want)))
            if got != want:
                raise BundleRefusal(
                    f"{name}: pass {index} sample {sample} cycle {cycle} "
                    f"cosimulated {got} against the cycle-accurate twin's "
                    f"{want}. The bundle's expectations would not be a "
                    f"measurement of this network")


def _require_reader_agrees(name: str, build: PassBuild, traces, plan, cycles) -> None:
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


def _require_capacities(name: str, build: PassBuild, tokens: int, events: int) -> None:
    if events >= SHIPPED_CAPTURE_EVENTS:
        raise BundleRefusal(
            f"{name}: pass {build.index} produced {events} capture events "
            f"against the shipped fabric's {SHIPPED_CAPTURE_EVENTS}-record RAM; "
            f"a board run would refuse as truncated")
    needed = stimulus_base_word(len(build.program_bytes) // WORD_BYTES) + tokens
    if needed > SHIPPED_PROGRAM_WORDS_PER_CORE:
        raise BundleRefusal(
            f"{name}: pass {build.index} needs {needed} program words but an "
            f"NC=1 kernel's program RAM holds {SHIPPED_PROGRAM_WORDS_PER_CORE}")


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
) -> Dict[str, Any]:
    """Freeze one network + sample set into a sealed, board-executable bundle."""
    if len(labels) != len(rasters):
        raise BundleRefusal(
            f"{name}: {len(rasters)} sample(s) against {len(labels)} label(s)")
    traces = [
        simulate_cycles(
            mapping, soma_law=soma_law, input_counts=raster,
            simulation_length=int(simulation_length),
            membrane_init=int(membrane_init), chip_latency=int(chip_latency))
        for raster in rasters
    ]
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
        measured = run_cosim(
            build.export, [build.per_cycle(trace) for trace in traces],
            latencies=[build.latency])
        _require_cosim_matches_twin(name, build.index, measured.counts, traces)
        per_sample_events = [
            sum(value for (sample, _c, _k, _n), value in measured.counts.items()
                if sample == index)
            for index in range(len(traces))
        ]
        _require_capacities(
            name, build, len(build.reference_stimulus(traces[0])),
            max(per_sample_events, default=0))
        plan["cosim_events_per_sample"] = [int(v) for v in per_sample_events]
        cores.append(plan)
        replay.extend(_replay_runs(build, traces, measured, cycles))

    windows = [
        {index: [int(v) for v in trace.window_counts()[index]]
         for index in range(len(mapping.cores))}
        for trace in traces
    ]
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
    sealed = bundle.seal(document)
    capture = bundle.seal({
        "schema": CAPTURE_SCHEMA,
        "bundle": sealed["name"],
        "bundle_self_hash": sealed["self_hash"],
        "provenance": dict(provenance),
        "note": (
            "What the cosimulated fabric DMA'd back, one entry per (core, "
            "sample) pass, keyed by the sha256 of the stimulus that produced "
            "it. A fake pyxrt replays these; a host that sends a stimulus not "
            "listed here gets no verdict, which is how a wrong stimulus is "
            "caught rather than answered. device_cycles is the free-running "
            "counter of the WHOLE multi-sample cosimulation of that pass, not "
            "a per-sample figure."),
        "runs": replay,
    })
    return sealed, capture


#: The replay document a fake pyxrt is armed with; sealed like the bundle.
CAPTURE_SCHEMA = "odin_hacc_deployment_capture/1"


def _replay_runs(build: PassBuild, traces, measured, cycles: int
                 ) -> List[Dict[str, Any]]:
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
            "device_cycles": int(measured.capture.cycles),
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
