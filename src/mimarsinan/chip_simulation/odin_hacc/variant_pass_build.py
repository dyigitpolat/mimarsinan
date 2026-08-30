"""One core of a network as ONE host-mediated pass on a GENERATED fabric.

The stock sibling (``pass_build.py``) programs the vendored crossbar over SPI
and words an axon event as ``{physical row, 0x07}``. A generated core is the
same pass with two things changed, and nothing else: it is programmed through
the direct configuration port (``OP_PROG``, one write per clock), and its
AER-in word IS the axon slot — one row per logical slot, because the synapse
cell signs itself.

Both encodings come from ``odin_rtl/stimulus.py`` and both programming tables
from ``odin_gen/packer.py``; nothing is re-derived here. What this file owns is
the SPLIT a bundle needs: the program payload, the per-sample CLEAR prefix, and
the per-cycle stimulus the shipped reader must rebuild byte for byte.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle
from mimarsinan.chip_simulation.odin_fpga.payload import payload_bytes
from mimarsinan.chip_simulation.odin_hacc.pass_build import pass_mapping, used_neurons
from mimarsinan.chip_simulation.odin_rtl.gen_cosim import (
    SETTLE_CYCLES,
    clear_ops,
    gate_ops,
    injection_ops,
    program_ops,
)
from mimarsinan.chip_simulation.odin_rtl.program_ops import barrier_stage_ops, tag_op
from mimarsinan.chip_simulation.odin_rtl.reference import CycleTrace, core_routes
from mimarsinan.chip_simulation.odin_rtl.stimulus import Op, encode_ops
from mimarsinan.mapping.export.odin.feasibility import (
    check_fan_in,
    check_membrane_init,
    check_theta_ceiling,
)
from mimarsinan.mapping.export.odin_gen.packer import build_variant_core_image
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec

#: The pass runs one core, and the packed image is that core's replica at 0.
PASS_CORE_INDEX = 0


class VariantPassBuild:
    """One generated core's pass: its OP_PROG image, its CLEAR, its stimulus."""

    def __init__(self, mapping: Any, index: int, trace: CycleTrace, *,
                 spec: CoreSpec, membrane_init: int) -> None:
        self.index = int(index)
        self.core = mapping.cores[index]
        self.spec = spec
        self.latency = int(trace.latencies[index])
        self.neurons = int(self.core.neurons_per_core)
        self.used = used_neurons(self.core)
        self.routes = core_routes(self.core)
        self.theta = check_theta_ceiling(
            self.core.threshold, membrane_bits=int(spec.membrane_bits),
            core_index=self.index)
        initial = check_membrane_init(
            membrane_init, theta=self.theta, core_index=self.index)
        check_fan_in(
            int(self.core.axons_per_core) - int(self.core.available_axons or 0),
            effective_max_axons=int(spec.max_axons) - 1, core_index=self.index)
        self.image = build_variant_core_image(
            pass_mapping(self.core).cores[PASS_CORE_INDEX], spec=spec,
            core_index=PASS_CORE_INDEX, theta=self.theta, membrane_init=initial)
        # Every declared slot is one physical row: the sign lives in the cell,
        # so nothing expands and the identity map IS the slot->row table.
        self.slot_rows = {
            slot: (slot,) for slot in range(int(self.core.axons_per_core))}
        self.program_bytes = payload_bytes(self._program_ops())
        self._stimulus: Dict[int, Tuple[int, ...]] = {}

    def _program_ops(self) -> List[Op]:
        """Gate, write both memories, ungate — the cosimulation's own prefix."""
        cores = [PASS_CORE_INDEX]
        return (gate_ops(cores, on=True) + program_ops(self.image)
                + gate_ops(cores, on=False))

    def _clear_ops(self) -> List[Op]:
        """The per-sample CLEAR: the membrane state, and nothing else."""
        cores = [PASS_CORE_INDEX]
        return (gate_ops(cores, on=True) + clear_ops(self.image)
                + gate_ops(cores, on=False))

    def _cycle_ops(self, trace: CycleTrace, cycle: int) -> List[Op]:
        """One cycle: TAG, this core's gathered slots, the settling wait."""
        ops = [tag_op(bundle.tag_of(0, cycle, trace.total_cycles))]
        if cycle >= self.latency:
            ops.extend(injection_ops(
                PASS_CORE_INDEX, trace.inputs[cycle][self.index], spec=self.spec))
        ops.extend(barrier_stage_ops({"cycles": SETTLE_CYCLES}))
        return ops

    def per_cycle(self, trace: CycleTrace) -> List[Dict[int, Tuple[int, ...]]]:
        """The pass's own injection plan: this core's gathered slots, per cycle."""
        return [{PASS_CORE_INDEX: trace.inputs[cycle][self.index]}
                for cycle in range(trace.total_cycles)]

    def reference_stimulus(self, trace: CycleTrace) -> Tuple[int, ...]:
        """The stimulus the REPOSITORY encoder builds for one sample of this pass."""
        cached = self._stimulus.get(id(trace))
        if cached is None:
            ops = list(self._clear_ops())
            for cycle in range(trace.total_cycles):
                ops.extend(self._cycle_ops(trace, cycle))
            cached = encode_ops(ops)
            self._stimulus[id(trace)] = cached
        return cached

    def plan_document(self, traces: Sequence[CycleTrace]) -> Dict[str, Any]:
        """The per-core plan a bundle carries, in the schema's own shape."""
        return {
            "core": self.index,
            "latency": self.latency,
            "neurons": self.neurons,
            "used_neurons": self.used,
            "barrier_cycles": int(SETTLE_CYCLES),
            "routes": [[int(kind), int(index)] for kind, index in self.routes],
            "slot_rows": [[int(slot), [int(row) for row in rows]]
                          for slot, rows in sorted(self.slot_rows.items())],
            "clear_prefix": [int(word) for word in encode_ops(self._clear_ops())[:-1]],
            "max_stimulus_words": max(
                len(self.reference_stimulus(trace)) for trace in traces),
            "program": bundle.encode_payload(self.program_bytes),
        }
