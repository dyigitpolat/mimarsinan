"""One core of a network as ONE host-mediated pass: its export and its stimulus."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle
from mimarsinan.chip_simulation.odin_fpga.payload import (
    payload_bytes,
    program_plan,
    run_plan,
    stimulus_ops,
)
from mimarsinan.chip_simulation.odin_rtl.program_ops import (
    slot_rows_from_inject,
    stages_of_kind,
)
from mimarsinan.chip_simulation.odin_rtl.reference import CycleTrace
from mimarsinan.chip_simulation.odin_rtl.stimulus import OP_TAG, encode_ops
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.exporter import export_odin
from mimarsinan.mapping.export.odin.program import STAGE_INJECT
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping


def used_neurons(core: Any) -> int:
    return max(int(core.neurons_per_core) - int(core.available_neurons or 0), 1)


def pass_mapping(core: Any) -> HardCoreMapping:
    """The one-core mapping a pass programs: every host-fed slot is an INPUT.

    On the chip nothing routes between cores (``SPI_OPEN_LOOP``), so a
    consumer's slots are delivered by the host either way, and the rewrite is
    what makes a one-core segment graph a DAG the exporter's emission bound is
    defined on. Always-on rows keep their source: the exporter injects those.
    """
    sources = [
        source if (getattr(source, "is_always_on_", False)
                   or getattr(source, "is_off_", False))
        else SpikeSource(-2, slot, is_input=True)
        for slot, source in enumerate(core.axon_sources)
    ]
    values = np.asarray(core.get_core_matrix(), dtype=np.float64)
    replica = HardCore(
        axons_per_core=values.shape[0], neurons_per_core=values.shape[1],
        has_bias_capability=False)
    replica.core_matrix = values
    replica.axon_sources = sources
    replica.threshold = float(core.threshold)
    # The replica's UNUSED tail is the original's: a pass rewrites who drives a
    # slot, never how many slots the mapper actually filled, and the exporter's
    # fan-in gate is defined on the filled ones.
    replica.available_axons = int(core.available_axons or 0)
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
        # One encode per (pass, sample): rebuilding it costs the whole CONFIG
        # prefix (a full 256x256 crossbar image) every time, and the freezer
        # asks for the same stimulus four times per sample.
        self._stimulus: Dict[int, Tuple[int, ...]] = {}

    def per_cycle(self, trace: CycleTrace) -> List[Dict[int, Tuple[int, ...]]]:
        """The pass's own injection plan: this core's gathered slots, per cycle."""
        return [{0: trace.inputs[cycle][self.index]}
                for cycle in range(trace.total_cycles)]

    def reference_stimulus(self, trace: CycleTrace) -> Tuple[int, ...]:
        """The stimulus the REPOSITORY encoder builds for one sample of this pass."""
        cached = self._stimulus.get(id(trace))
        if cached is None:
            full = run_plan(
                self.export, [self.per_cycle(trace)], latencies=[self.latency])
            cached = encode_ops(stimulus_ops(self.program, full))
            self._stimulus[id(trace)] = cached
        return cached

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
