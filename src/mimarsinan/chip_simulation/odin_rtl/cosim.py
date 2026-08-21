"""The ODIN RTL cosimulation instrument: an ``OdinExport`` becomes measured counts.

This is NOT a `BACKEND_REGISTRY` backend. It is the instrument that discharges
plan gate R11a: it programs the byte-identical vendored core over SPI exactly as
the exported sequencer program says, drives the canonical event order over the
AER-in link, and reports the per-neuron per-cycle counts the crossbar actually
produced. The physical backend (XRT) is P7's work and joins the registry there.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from mimarsinan.chip_simulation.odin_rtl.program_ops import (
    STAGE_CLEAR,
    STAGE_CONFIG,
    STAGE_INJECT,
    STAGE_TREF,
    barrier_stage_ops,
    clear_stage_ops,
    config_stage_ops,
    gate_stage_ops,
    inject_ops,
    neuron_readback_ops,
    plan_cycle_injection,
    shadow_ops,
    slot_rows_from_inject,
    stages_of_kind,
    synapse_readback_ops,
    tag_op,
    tref_stage_ops,
)
from mimarsinan.chip_simulation.odin_rtl.capture import (
    CaptureResult,
    parse_capture,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import (
    Op,
    StimulusError,
    op_summary,
    write_stimulus,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    SimulationRun,
    TestbenchBuild,
    build_testbench,
    run_testbench,
)
from mimarsinan.mapping.export.odin.program import STAGE_BARRIER

#: Tag 0 is what the testbench starts with, so windows are numbered from 1 and
#: anything captured outside a window (there should be nothing) stays visible.
FIRST_TAG = 1


@dataclass(frozen=True)
class CosimPlan:
    """The token program plus the bookkeeping needed to read its capture back."""

    ops: Tuple[Op, ...]
    n_cores: int
    cycles_per_sample: int
    samples: int
    barrier_cycles: int

    def tag_of(self, sample: int, cycle: int) -> int:
        return FIRST_TAG + sample * self.cycles_per_sample + cycle

    def decode_tag(self, tag: int) -> Tuple[int, int]:
        offset = int(tag) - FIRST_TAG
        return divmod(offset, self.cycles_per_sample)


@dataclass(frozen=True)
class CosimResult:
    """Everything one cosimulation run measured."""

    counts: Dict[Tuple[int, int, int, int], int]
    capture: CaptureResult
    plan: CosimPlan
    build: TestbenchBuild
    run: SimulationRun
    token_count: int
    program_summary: Mapping[str, int]

    def cycle_counts(self, sample: int, cycle: int, core: int, n_neurons: int
                     ) -> Tuple[int, ...]:
        """The per-neuron counts one core emitted in one cycle of one sample."""
        return tuple(
            self.counts.get((sample, cycle, core, neuron), 0)
            for neuron in range(n_neurons)
        )

    def window_counts(self, *, latencies: Sequence[int], simulation_length: int,
                      neurons: Sequence[int]) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        """Per-sample per-core per-neuron counts over each core's own window."""
        totals = []
        for sample in range(self.plan.samples):
            per_core = []
            for core, count in enumerate(neurons):
                row = [0] * count
                for cycle in range(self.plan.cycles_per_sample):
                    local = cycle - int(latencies[core])
                    if not 0 <= local < int(simulation_length):
                        continue
                    for neuron in range(count):
                        row[neuron] += self.counts.get(
                            (sample, cycle, core, neuron), 0)
                per_core.append(tuple(row))
            totals.append(tuple(per_core))
        return tuple(totals)


def _barrier_cycles(program: Any) -> int:
    barriers = stages_of_kind(program, STAGE_BARRIER)
    if not barriers:
        raise StimulusError(
            "the sequencer program carries no BARRIER stage: without the "
            "deterministic drain bound the harness has no defined moment at "
            "which a cycle's output is complete")
    return max(int(payload["cycles"]) for payload in barriers)


def build_cosim_ops(
    export: Any,
    per_cycle_inputs: Sequence[Sequence[Mapping[int, Sequence[int]]]],
    *,
    latencies: Sequence[int],
    readback: bool = False,
    shadow: bool = False,
    barrier_cycles: int | None = None,
) -> CosimPlan:
    """Translate the export plus a per-sample per-cycle injection plan into ops.

    ``per_cycle_inputs[sample][cycle][core]`` is that core's FULL per-slot count
    vector (always-on slots included, as the gather delivers them); a core is
    injected only from its own latency onward, mirroring the latency gate the
    software twins apply.
    """
    program = export.program
    configs = stages_of_kind(program, STAGE_CONFIG)
    injects = stages_of_kind(program, STAGE_INJECT)
    clears = stages_of_kind(program, STAGE_CLEAR)
    trefs = stages_of_kind(program, STAGE_TREF)
    core_indices = [int(payload["core_index"]) for payload in configs]
    if len(core_indices) != len(export.cores):
        raise StimulusError(
            f"{len(configs)} CONFIG stages for {len(export.cores)} core images")
    slot_rows = {
        int(payload["core_index"]): slot_rows_from_inject(payload)[0]
        for payload in injects
    }
    bound = _barrier_cycles(program) if barrier_cycles is None else int(barrier_cycles)

    ops: List[Op] = []
    for payload in configs:
        ops.extend(config_stage_ops(payload))
    if shadow:
        for image in export.cores:
            ops.extend(shadow_ops(
                image.core_index, [
                    {"address": write.address, "value": write.value}
                    for write in image.register_writes
                ],
                gate_on=True))
    if readback:
        for image in export.cores:
            ops.extend(neuron_readback_ops(image.core_index, image.neuron_words))
            ops.extend(synapse_readback_ops(image.core_index, image.synapse_words))
    ops.extend(gate_stage_ops(core_indices, on=False))

    cycles_per_sample = len(per_cycle_inputs[0]) if per_cycle_inputs else 0
    for sample, cycles in enumerate(per_cycle_inputs):
        if len(cycles) != cycles_per_sample:
            raise StimulusError(
                "every sample must run the same number of cycles; sample "
                f"{sample} has {len(cycles)} against {cycles_per_sample}")
        # The per-sample CLEAR: the membrane state bytes are rewritten with the
        # network gated, and NOTHING else is reprogrammed (plan §7 row 15).
        ops.extend(gate_stage_ops(core_indices, on=True))
        for payload in clears:
            ops.extend(clear_stage_ops(payload))
        ops.extend(gate_stage_ops(core_indices, on=False))
        for cycle, per_core in enumerate(cycles):
            ops.append(tag_op(FIRST_TAG + sample * cycles_per_sample + cycle))
            for core in core_indices:
                if cycle < int(latencies[core]):
                    continue
                counts = per_core.get(core)
                if counts is None:
                    raise StimulusError(
                        f"sample {sample} cycle {cycle} has no slot counts for "
                        f"core {core}")
                ops.extend(inject_ops(
                    core, plan_cycle_injection(slot_rows[core], counts)))
            for payload in trefs:
                ops.extend(tref_stage_ops(payload, core_indices))
            ops.extend(barrier_stage_ops({"cycles": bound}))
    return CosimPlan(
        ops=tuple(ops), n_cores=len(core_indices),
        cycles_per_sample=cycles_per_sample, samples=len(per_cycle_inputs),
        barrier_cycles=bound,
    )


def run_cosim(
    export: Any,
    per_cycle_inputs: Sequence[Sequence[Mapping[int, Sequence[int]]]],
    *,
    latencies: Sequence[int],
    readback: bool = False,
    shadow: bool = False,
    overlay: bool = False,
    engine: str | None = None,
    workdir: Path | None = None,
    timeout_s: float = 3600.0,
) -> CosimResult:
    """Program, inject, drain and read back one export on the vendored RTL."""
    plan = build_cosim_ops(
        export, per_cycle_inputs, latencies=latencies,
        readback=readback, shadow=shadow)
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(workdir) if workdir is not None else Path(scratch)
        root.mkdir(parents=True, exist_ok=True)
        stimulus = root / "odin_stim.hex"
        token_count = write_stimulus(stimulus, plan.ops)
        build = build_testbench(
            n_cores=plan.n_cores, token_count=token_count,
            overlay=overlay, engine=engine)
        run = run_testbench(build, stimulus, timeout_s=timeout_s)
    capture = parse_capture(run.stdout)
    counts: Dict[Tuple[int, int, int, int], int] = {}
    for event in capture.events:
        sample, cycle = plan.decode_tag(event.tag)
        key = (sample, cycle, event.core, event.neuron)
        counts[key] = counts.get(key, 0) + 1
    return CosimResult(
        counts=counts, capture=capture, plan=plan, build=build, run=run,
        token_count=token_count, program_summary=op_summary(plan.ops),
    )
