"""Cosimulating a GENERATED variant core: the P5 instrument, one program apart.

The stock harness (``cosim.py``) executes the exporter's sequencer program over
SPI. A generated variant is the same instrument with the same capture, the same
reference twin and the same count comparison — only the PROGRAMMING changes, to
the direct synchronous port the generator's statement of changes declares.

One thing the variant gets for free that the stock core does not: its AER-in
ACK rises only when the whole sweep (and every output handshake it produced) is
finished, so a window's completion is OBSERVABLE ON THE WIRE. The stock core
pushes into a scheduler FIFO and therefore needs the exported BARRIER's cycle
bound; here the barrier is a settling margin, and the gate is the counts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from mimarsinan.chip_simulation.odin_rtl.capture import parse_capture
from mimarsinan.chip_simulation.odin_rtl.cosim import FIRST_TAG, CosimPlan, CosimResult
from mimarsinan.chip_simulation.odin_rtl.stimulus import (
    OP_AER,
    OP_PROG,
    OP_TAG,
    OP_WAIT,
    Op,
    StimulusError,
    op_summary,
    variant_axon_event,
    variant_tref_event,
    write_stimulus,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    build_testbench,
    materialize_sources,
    run_testbench,
)
from mimarsinan.mapping.export.odin_gen.packer import ProgWrite, VariantCoreImage, gate_write
from mimarsinan.mapping.export.odin_gen.render import spec_flags_word
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec
from mimarsinan.mapping.platform.event_order import drain_events

VARIANT_TB = "tb_odin_gen_core"

#: Clocks the testbench idles after a window; the variant's completion is
#: handshake-observable, so this is a settling margin and never a drain bound.
SETTLE_CYCLES = 8


def prog_op(core_index: int, write: ProgWrite) -> Op:
    """One configuration-port write as a testbench token."""
    return Op(OP_PROG, (int(core_index), int(write.sel), int(write.addr),
                        int(write.data)))


def gate_ops(core_indices: Sequence[int], *, on: bool) -> List[Op]:
    """Gate (or ungate) activity on every core, exactly as the stock GATE stage."""
    return [prog_op(index, gate_write(on=on)) for index in core_indices]


def program_ops(image: VariantCoreImage) -> List[Op]:
    """One core's whole memory image, thresholds first then synapses."""
    return [
        prog_op(image.core_index, write)
        for write in tuple(image.threshold_writes) + tuple(image.synapse_writes)
    ]


def clear_ops(image: VariantCoreImage) -> List[Op]:
    """The per-sample CLEAR: the membrane state, and nothing else, rewritten."""
    return [prog_op(image.core_index, write) for write in image.membrane_writes]


def injection_ops(core_index: int, counts: Sequence[int], *, spec: CoreSpec
                  ) -> List[Op]:
    """One cycle's events for one core, in the CANONICAL order.

    Ascending slots with each slot's multiplicity ADJACENT — the wire contract
    the whole equivalence rests on, taken from ``platform.event_order`` rather
    than re-derived here.
    """
    bits = spec.axon_address_bits
    ops: List[Op] = []
    for slot, multiplicity in drain_events(list(counts)):
        if slot >= spec.max_axons:
            raise StimulusError(
                f"slot {slot} is outside the generated core's {spec.max_axons} "
                f"axon rows")
        word = variant_axon_event(slot, axon_address_bits=bits)
        ops.extend([Op(OP_AER, (int(core_index), word))] * int(multiplicity))
    return ops


def build_variant_ops(
    images: Sequence[VariantCoreImage],
    per_cycle_inputs: Sequence[Sequence[Mapping[int, Sequence[int]]]],
    *,
    spec: CoreSpec,
    latencies: Sequence[int],
) -> CosimPlan:
    """The whole token program: program, then one CLEAR-and-run per sample."""
    core_indices = [int(image.core_index) for image in images]
    ops: List[Op] = []
    ops.extend(gate_ops(core_indices, on=True))
    for image in images:
        ops.extend(program_ops(image))
    ops.extend(gate_ops(core_indices, on=False))

    cycles_per_sample = len(per_cycle_inputs[0]) if per_cycle_inputs else 0
    for sample, cycles in enumerate(per_cycle_inputs):
        if len(cycles) != cycles_per_sample:
            raise StimulusError(
                "every sample must run the same number of cycles; sample "
                f"{sample} has {len(cycles)} against {cycles_per_sample}")
        ops.extend(gate_ops(core_indices, on=True))
        for image in images:
            ops.extend(clear_ops(image))
        ops.extend(gate_ops(core_indices, on=False))
        for cycle, per_core in enumerate(cycles):
            ops.append(Op(OP_TAG, (FIRST_TAG + sample * cycles_per_sample + cycle,)))
            for core in core_indices:
                if cycle < int(latencies[core]):
                    continue
                counts = per_core.get(core)
                if counts is None:
                    raise StimulusError(
                        f"sample {sample} cycle {cycle} has no slot counts for "
                        f"core {core}")
                ops.extend(injection_ops(core, counts, spec=spec))
            if not spec.per_event:
                # The sync-fire law fires at the time reference: the window's
                # whole charge is accumulated first, then compared once.
                tref = variant_tref_event(
                    axon_address_bits=spec.axon_address_bits)
                for core in core_indices:
                    if cycle < int(latencies[core]):
                        continue
                    ops.append(Op(OP_AER, (core, tref)))
            ops.append(Op(OP_WAIT, (SETTLE_CYCLES,)))
    return CosimPlan(
        ops=tuple(ops), n_cores=len(core_indices),
        cycles_per_sample=cycles_per_sample, samples=len(per_cycle_inputs),
        barrier_cycles=SETTLE_CYCLES,
    )


def variant_testbench_params(spec: CoreSpec) -> Dict[str, int]:
    """The testbench parameters that must MATCH the generated file's own spec."""
    return {
        "AW": spec.axon_address_bits,
        "NW": spec.neuron_address_bits,
        "AXONS": spec.max_axons,
        "NEURONS": spec.max_neurons,
        "MBITS": spec.membrane_bits,
        "WBITS": spec.weight_bits,
        "FLAGS": spec_flags_word(spec),
    }


def run_variant_cosim(
    generated: Any,
    images: Sequence[VariantCoreImage],
    per_cycle_inputs: Sequence[Sequence[Mapping[int, Sequence[int]]]],
    *,
    latencies: Sequence[int],
    engine: str | None = None,
    workdir: Path | None = None,
    timeout_s: float = 3600.0,
) -> CosimResult:
    """Program, inject and read back one generated core on its own RTL."""
    spec = generated.spec
    plan = build_variant_ops(
        images, per_cycle_inputs, spec=spec, latencies=latencies)
    sources = materialize_sources(generated.files, spec.spec_key())
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(workdir) if workdir is not None else Path(scratch)
        root.mkdir(parents=True, exist_ok=True)
        stimulus = root / "odin_gen_stim.hex"
        token_count = write_stimulus(stimulus, plan.ops)
        build = build_testbench(
            n_cores=plan.n_cores, token_count=token_count, engine=engine,
            tb_name=VARIANT_TB, rtl_sources=sources,
            extra_params=variant_testbench_params(spec),
        )
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
