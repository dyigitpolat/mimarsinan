"""[ODIN5] The BRAM overlay is the upstream-mandated substitution, not a change.

Upstream (doc/README.md Sec.5, and the comments at `neuron_core.v:300` /
`synaptic_core.v:153`) instructs the implementer to replace the two behavioural
memory models with SRAM macros or Block RAM. `hw/fpga/mem/` does exactly that,
under the same module names and ports, selected by SOURCE-FILE ORDER so that
`hw/vendor/odin` is never edited.

This gate runs the SAME memory testbench twice -- once with the vendored
behavioural declarations, once with the overlay in front of them -- and requires
the two transcripts to be byte-identical, which pins read latency, the CS hold
semantics and the read-before-write ordering the controller depends on. It then
runs the FULL core under the overlay and requires the fold's counts unchanged.
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

from mimarsinan.chip_simulation.odin_rtl.cosim import run_cosim
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    ENGINE_COMPILED,
    SimulatorBuildError,
    build_testbench,
    overlay_sources,
    run_testbench,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.latency.chip import ChipLatency

pytestmark = [pytest.mark.slow, pytest.mark.integration]

S = 3


def _memory_transcript(tmp_path: Path, *, overlay: bool) -> list[str]:
    """The testbench's OWN lines only; the simulator's footer carries a wall time."""
    build = build_testbench(
        n_cores=1, token_count=1, overlay=overlay,
        engine=ENGINE_COMPILED, tb_name="tb_mem_overlay", parameterised=False)
    stimulus = tmp_path / f"unused_{int(overlay)}.hex"
    stimulus.write_text("00000000\n")
    stdout = run_testbench(build, stimulus).stdout
    return [line for line in stdout.splitlines() if line.startswith("MEM")]


def _mapping():
    matrix = np.array(
        [[3.0, -2.0, 0.0], [0.0, 5.0, -4.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    core = hard_core(
        matrix, threshold=5.0,
        sources=[SpikeSource(-2, 0, is_input=True), SpikeSource(-2, 1, is_input=True),
                 SpikeSource(-3, 0, is_always_on=True)],
    )
    mapping = mapping_of([core], [SpikeSource(0, i) for i in range(3)])
    ChipLatency(mapping).calculate()
    return mapping


class TestTheOverlayBehavesLikeTheModelItReplaces:
    def test_the_two_memory_transcripts_are_byte_identical(self, tmp_path):
        engine = require_simulator()
        if engine != ENGINE_COMPILED:
            pytest.skip(
                "the overlay is selected by source-file order, which only the "
                "compiled engine implements; iverilog rejects the duplicate "
                "module declaration outright")
        with timed("memory overlay equivalence"):
            vendored = _memory_transcript(tmp_path, overlay=False)
            overlaid = _memory_transcript(tmp_path, overlay=True)
        assert vendored[-1].startswith("MEMDONE")
        assert overlaid[-1].startswith("MEMDONE")
        assert len(vendored) > 4000
        assert vendored == overlaid

    def test_the_overlay_really_was_in_the_source_list(self):
        names = sorted(path.name for path in overlay_sources())
        assert names == ["SRAM_256x128_wrapper.v", "SRAM_8192x32_wrapper.v"]

    def test_an_overlay_build_on_the_interpreted_engine_is_refused_by_name(self):
        with pytest.raises(SimulatorBuildError, match="source-file order"):
            build_testbench(n_cores=1, token_count=1, overlay=True,
                            engine="iverilog")


class TestTheWholeCoreRunsUnchangedOnTheOverlay:
    def test_the_counts_are_the_folds_counts_with_the_bram_memories(self):
        engine = require_simulator()
        if engine != ENGINE_COMPILED:
            pytest.skip("overlay selection needs the compiled engine")
        mapping = _mapping()
        export = export_of(mapping)
        samples = traces_for(mapping, [[[1, 1]] * S], simulation_length=S)
        with timed("overlay full-core cosim"):
            overlaid = run_cosim(
                export, [list(samples[0].per_cycle)],
                latencies=samples[0].trace.latencies, overlay=True)
            vendored = run_cosim(
                export, [list(samples[0].per_cycle)],
                latencies=samples[0].trace.latencies, overlay=False)
        assert overlaid.counts == vendored.counts
        for cycle, per_core in enumerate(samples[0].trace.outputs):
            expected = tuple(int(v) for v in per_core[0])
            assert overlaid.cycle_counts(0, cycle, 0, len(expected)) == expected
        assert overlaid.capture.events, "the overlay run produced no spikes at all"
