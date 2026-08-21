"""[ODIN5.5a] The LOCAL synthesizability gate: the stock core through yosys.

Plan §7 row 19, local half. The vendored core plus the `hw/fpga/mem/` BRAM
overlay is pushed through `synth_xilinx -family xcup -top ODIN` (xcup is the
UltraScale+ family the U55C belongs to). The gate asserts three things and hides
none of them: the synthesis reports zero errors, BOTH memories land in block RAM
rather than in distributed RAM or flip-flops, and the resource census is
bit-for-bit the one committed in `hw/fpga/synth_resources.json` (regenerate it
deliberately with `scripts/hw_tests/regen_synth_report.py` when the RTL moves).

The control run -- the SAME script over the vendored tree ALONE -- is the local
witness for plan finding F17: the stock memories are behavioural models, and
without the overlay they do not reach block RAM.

yosys-synthesizability is NECESSARY, NOT SUFFICIENT for Vivado closure on the
U55C shell; that is P7's HACC build.
"""

from __future__ import annotations

import pytest

from integration.odin_rtl_harness import timed

from mimarsinan.chip_simulation.odin_rtl.synth_artifacts import (
    RESOURCES_JSON,
    build_record,
    gated_view,
    load_committed,
)
from mimarsinan.chip_simulation.odin_rtl.synth_census import memory_inferences
from mimarsinan.chip_simulation.odin_rtl.synthesis import (
    SynthesisUnavailable,
    run_synthesis,
    synthesis_unavailable_reason,
    yosys_version,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _require_yosys() -> str:
    try:
        return yosys_version()
    except SynthesisUnavailable:
        pytest.skip(synthesis_unavailable_reason())
        raise  # pragma: no cover - pytest.skip raises


@pytest.fixture(scope="module")
def synthesis(tmp_path_factory):
    """One overlay run and one vendor-only control run, shared by every check."""
    version = _require_yosys()
    root = tmp_path_factory.mktemp("odin_synth")
    with timed("yosys synthesis (vendor + BRAM overlay)"):
        overlay = run_synthesis(overlay=True, workdir=root / "overlay")
    with timed("yosys synthesis (vendored memories alone, F17 control)"):
        control = run_synthesis(overlay=False, workdir=root / "control")
    return version, overlay, control


class TestTheStockCoreSynthesizesForUltraScalePlus:
    def test_the_synthesis_reports_zero_errors(self, synthesis):
        _, overlay, _ = synthesis
        assert overlay.succeeded, (
            f"yosys refused the vendor+overlay design: {overlay.error_line}\n"
            f"{overlay.log_tail}")
        assert overlay.error_line is None
        assert overlay.stat is not None

    def test_the_core_is_the_top_and_the_census_is_not_empty(self, synthesis):
        _, overlay, _ = synthesis
        table = overlay.resource_table()
        assert table.total_cells > 0
        assert table.luts > 0 and table.flip_flops > 0
        print(f"[odin-rtl] per-core resources: {table.as_record()}")


class TestBothMemoriesReachBlockRam:
    def test_neither_memory_falls_back_to_distributed_ram_or_registers(
            self, synthesis):
        _, overlay, _ = synthesis
        inferences = memory_inferences(overlay.require_stat())
        findings = [m.finding() for m in inferences if m.finding() is not None]
        assert not findings, "\n".join(findings)

    def test_the_vendored_memories_alone_do_not_reach_block_ram(self, synthesis):
        _, _, control = synthesis
        if not control.succeeded:
            print(f"[odin-rtl] F17 control: yosys refused the vendored tree "
                  f"outright -- {control.error_line}")
            return
        reached = [m.module for m in memory_inferences(control.require_stat())
                   if m.inferred_as_bram]
        assert not reached, (
            f"the vendored behavioural memories reached block RAM without the "
            f"overlay ({reached}); plan finding F17 and the whole "
            f"`hw/fpga/mem/` overlay need re-deriving")


class TestTheResourceTableIsTheCommittedOne:
    def test_the_fresh_census_matches_the_committed_artifact_exactly(
            self, synthesis):
        version, overlay, control = synthesis
        fresh = build_record(
            attempt=overlay, control=control, tool_version=version)
        committed = load_committed()
        if gated_view(fresh) != gated_view(committed):
            differing = sorted(
                key for key, value in gated_view(fresh).items()
                if gated_view(committed).get(key) != value)
            pytest.fail(
                f"the synthesis no longer produces the committed numbers in "
                f"{RESOURCES_JSON} (sections {differing}).\n"
                f"committed tool: {committed['tool']['version']}\n"
                f"this tool:      {version}\n"
                f"fresh per-core: {fresh['per_core']}\n"
                f"committed:      {committed['per_core']}\n"
                f"Regenerate DELIBERATELY with "
                f"scripts/hw_tests/regen_synth_report.py when the RTL changed.")
