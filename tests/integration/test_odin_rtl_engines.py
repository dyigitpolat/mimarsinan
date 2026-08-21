"""[ODIN5] The one testbench source runs, and agrees, on both simulators.

The cosimulation gates run on the compiled engine because the interpreted one
executes the vendored core about a hundred times slower (measured: ~6.6k
cycles/s against ~750k), and one FULL memory programming pass is ~6M cycles per
core. That speed choice must not become an unexamined dependency on one
simulator's semantics, so this gate drives the SAME `hw/tb/tb_odin_core.v` and
the SAME token program under both and requires the same capture.

The program here is deliberately reduced to what this run TOUCHES -- every
neuron word (the controller's pop sweeps all 256) and the synapse words of the
physical rows that are injected -- because those memories have no reset and the
interpreted engine, unlike the compiled one, propagates X out of anything left
uninitialised. That is exactly the property that makes it worth running.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from integration.odin_rtl_harness import (
    export_of,
    hard_core,
    mapping_of,
    require_simulator,
    timed,
    traces_for,
)

from mimarsinan.chip_simulation.odin_rtl.capture import parse_capture
from mimarsinan.chip_simulation.odin_rtl.program_ops import (
    UNMASKED,
    config_register_ops,
    gate_stage_ops,
    inject_ops,
    neuron_image_ops,
    plan_cycle_injection,
    slot_rows_from_inject,
    stages_of_kind,
    tag_op,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import (
    OP_SPI_W,
    OP_WAIT,
    SYNAPSE_BYTES_PER_WORD,
    Op,
    masked_byte_data,
    synapse_address_field,
    word_bytes,
    write_stimulus,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    ENGINE_COMPILED,
    ENGINE_INTERPRETED,
    build_testbench,
    find_tool,
    run_testbench,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.program import STAGE_INJECT
from mimarsinan.mapping.latency.chip import ChipLatency

pytestmark = [pytest.mark.slow, pytest.mark.integration]

#: One physical row's dendritic tree spans 32 synapse words (doc Sec.3.1).
WORDS_PER_ROW = 32
BARRIER = 4096


def _mapping():
    matrix = np.array([[5.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    core = hard_core(
        matrix, threshold=4.0,
        sources=[SpikeSource(-2, 0, is_input=True),
                 SpikeSource(-3, 0, is_always_on=True)],
    )
    mapping = mapping_of([core], [SpikeSource(0, 0), SpikeSource(0, 1)])
    ChipLatency(mapping).calculate()
    return mapping


def _touched_synapse_ops(image, rows):
    """Every synapse byte of the physical ROWS this run drives, and no others."""
    ops = []
    for row in sorted(rows):
        for word_addr in range(row * WORDS_PER_ROW, (row + 1) * WORDS_PER_ROW):
            for byte_addr, value in enumerate(
                    word_bytes(int(image.synapse_words[word_addr]),
                               SYNAPSE_BYTES_PER_WORD)):
                ops.append(Op(OP_SPI_W, (
                    image.core_index,
                    synapse_address_field(word_addr, byte_addr, write=True),
                    masked_byte_data(value, UNMASKED),
                )))
    return ops


def _reduced_program(export, per_cycle):
    image = export.cores[0]
    payload = stages_of_kind(export.program, STAGE_INJECT)[0]
    slot_rows, _bias = slot_rows_from_inject(payload)
    driven = {row for rows in slot_rows.values() for row in rows}
    ops = list(config_register_ops(
        image.core_index,
        [{"address": w.address, "value": w.value} for w in image.register_writes]))
    ops += neuron_image_ops(image.core_index, image.neuron_words)
    ops += _touched_synapse_ops(image, driven)
    ops += gate_stage_ops([image.core_index], on=False)
    for cycle, counts in enumerate(per_cycle):
        ops.append(tag_op(cycle + 1))
        ops += inject_ops(
            image.core_index, plan_cycle_injection(slot_rows, counts[0]))
        ops.append(Op(OP_WAIT, (BARRIER,)))
    return ops


def _run(engine, ops, tmp_path):
    stimulus = Path(tmp_path) / f"engines_{engine}.hex"
    tokens = write_stimulus(stimulus, ops)
    build = build_testbench(n_cores=1, token_count=tokens, engine=engine)
    with timed(f"engine {engine}") as clock:
        run = run_testbench(build, stimulus, timeout_s=3600.0)
    capture = parse_capture(run.stdout)
    print(f"[odin-rtl] engine={engine} build={build.build_seconds:.1f}s "
          f"cached={build.cached} sim={run.seconds:.1f}s cycles={capture.cycles} "
          f"events={len(capture.events)} wall={clock.seconds:.1f}s")
    return capture


class TestBothSimulatorsExecuteTheSameTestbench:
    def test_the_captures_agree_event_for_event(self, tmp_path):
        require_simulator()
        if find_tool("iverilog") is None or find_tool("vvp") is None:
            pytest.skip(
                "the interpreted engine (iverilog+vvp) is not present in the "
                "configured simulator directory, so the cross-simulator check "
                "cannot run and is NOT being reported as passing")
        mapping = _mapping()
        export = export_of(mapping)
        samples = traces_for(mapping, [[[1], [1], [1]]], simulation_length=3)
        ops = _reduced_program(export, list(samples[0].per_cycle))

        compiled = _run(ENGINE_COMPILED, ops, tmp_path)
        interpreted = _run(ENGINE_INTERPRETED, ops, tmp_path)

        assert compiled.events, "the compiled engine captured nothing to compare"
        assert [(e.core, e.neuron, e.tag) for e in interpreted.events] == \
            [(e.core, e.neuron, e.tag) for e in compiled.events]

    def test_the_captured_counts_are_the_folds_counts(self, tmp_path):
        require_simulator()
        mapping = _mapping()
        export = export_of(mapping)
        samples = traces_for(mapping, [[[1], [1], [1]]], simulation_length=3)
        ops = _reduced_program(export, list(samples[0].per_cycle))
        capture = _run(ENGINE_COMPILED, ops, tmp_path)
        counts = capture.counts_by_tag()
        for cycle, per_core in enumerate(samples[0].trace.outputs):
            for neuron, expected in enumerate(per_core[0]):
                assert counts.get((cycle + 1, 0, neuron), 0) == int(expected), \
                    (cycle, neuron)
