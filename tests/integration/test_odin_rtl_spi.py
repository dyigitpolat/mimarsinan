"""[ODIN5, plan §7 row 17] What the SPI bus actually holds after programming.

Two claims, and one deliberate mutant that proves the first has teeth:

  * the FULL neuron and synapse memories -- 256x16 + 8192x4 = 36,864 bytes per
    core -- read back over SPI byte-for-byte equal to the exporter's images;
  * every configuration register holds what the CONFIG stage wrote. Those
    registers have NO readback path and no reset value (upstream doc Sec.4), so
    the testbench asserts them HIERARCHICALLY inside the DUT. That tap is a
    simulation-only instrument: the runtime cannot read them back and therefore
    reprograms them on every session start (plan §5.4).
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

from mimarsinan.chip_simulation.odin_rtl.cosim import build_cosim_ops, run_cosim
from mimarsinan.chip_simulation.odin_rtl.program_ops import (
    FULL_PROGRAM_TRANSACTIONS,
    neuron_readback_ops,
    synapse_readback_ops,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import (
    OP_SPI_R,
    Op,
    config_write_address,
    encode_ops,
    write_stimulus,
)
from mimarsinan.chip_simulation.odin_rtl.capture import parse_capture
from mimarsinan.chip_simulation.odin_rtl.toolchain import build_testbench, run_testbench
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.registers import config_register
from mimarsinan.mapping.latency.chip import ChipLatency

pytestmark = [pytest.mark.slow, pytest.mark.integration]

S = 2


def _mapping():
    matrix = np.array(
        [[2.0, -3.0, 0.0, 7.0],
         [0.0, 1.0, -7.0, 0.0],
         [1.0, 1.0, 4.0, -1.0],
         [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    core = hard_core(
        matrix, threshold=6.0,
        sources=[SpikeSource(-2, i, is_input=True) for i in range(3)]
        + [SpikeSource(-3, 0, is_always_on=True)],
    )
    mapping = mapping_of([core], [SpikeSource(0, i) for i in range(4)])
    ChipLatency(mapping).calculate()
    return mapping


@pytest.fixture(scope="module")
def readback():
    require_simulator()
    mapping = _mapping()
    export = export_of(mapping)
    samples = traces_for(mapping, [[[1, 1, 1]] * S], simulation_length=S)
    with timed("SPI readback + shadow") as clock:
        result = run_cosim(
            export, [list(samples[0].per_cycle)],
            latencies=samples[0].trace.latencies, readback=True, shadow=True)
    print(f"[odin-rtl] readback engine={result.build.engine} "
          f"sim={result.run.seconds:.1f}s cycles={result.capture.cycles} "
          f"reads={result.capture.reads} shadow={result.capture.shadow_checks} "
          f"wall={clock.seconds:.1f}s")
    return export, result


class TestTheMemoriesReadBackAsTheExporterWroteThem:
    def test_every_byte_of_both_memories_matches(self, readback):
        _export, result = readback
        assert result.capture.read_failures == 0, \
            result.capture.read_failure_lines[:10]

    def test_the_readback_covered_the_FULL_memories_not_a_sample(self, readback):
        export, result = readback
        assert result.capture.reads == FULL_PROGRAM_TRANSACTIONS * len(export.cores)
        assert result.capture.reads == 256 * 16 + 8192 * 4

    def test_the_run_still_produced_the_folds_counts(self, readback):
        _export, result = readback
        assert result.capture.events


class TestTheConfigurationRegistersHoldWhatWasWritten:
    def test_every_register_and_every_sign_word_is_asserted(self, readback):
        export, result = readback
        assert result.capture.shadow_checks == 26 * len(export.cores)

    def test_none_of_them_failed(self, readback):
        _export, result = readback
        assert result.capture.shadow_failures == 0, \
            result.capture.shadow_failure_lines[:10]


class TestTheReadbackGateHasTeeth:
    """A mutated expectation must be REPORTED, not absorbed."""

    def test_a_single_wrong_expected_byte_is_caught(self, tmp_path):
        require_simulator()
        mapping = _mapping()
        export = export_of(mapping)
        samples = traces_for(mapping, [[[1, 1, 1]] * S], simulation_length=S)
        plan = build_cosim_ops(
            export, [list(samples[0].per_cycle)],
            latencies=samples[0].trace.latencies)
        image = export.cores[0]
        clean = (
            neuron_readback_ops(0, image.neuron_words)[:64]
            + synapse_readback_ops(0, image.synapse_words)[:64]
        )
        mutated = list(clean)
        target = next(i for i, op in enumerate(mutated) if op.args[2] != 0)
        mutated[target] = Op(
            OP_SPI_R, (0, mutated[target].args[1], mutated[target].args[2] ^ 0xFF))
        # The memories are only SPI-accessible while the gate is asserted, so
        # the mutated reads go in just before the CONFIG phase releases it.
        gate = config_write_address(config_register("SPI_GATE_ACTIVITY").address)
        ops = list(plan.ops)
        insert = next(i for i, op in enumerate(ops)
                      if op.args[1:] == (gate, 0))
        ops = ops[:insert] + mutated + ops[insert:]

        stimulus = Path(tmp_path) / "mutant.hex"
        tokens = write_stimulus(stimulus, ops)
        build = build_testbench(n_cores=1, token_count=tokens)
        with timed("readback mutant"):
            run = run_testbench(build, stimulus)
        capture = parse_capture(run.stdout)
        assert capture.reads == len(clean)
        assert capture.read_failures == 1, capture.read_failure_lines
        assert capture.read_failure_lines, "a mismatch must be REPORTED, not counted"

    def test_the_encoded_program_length_is_what_the_tb_array_holds(self, readback):
        _export, result = readback
        assert result.token_count == len(encode_ops(result.plan.ops))
        assert result.token_count <= result.build.program_words
