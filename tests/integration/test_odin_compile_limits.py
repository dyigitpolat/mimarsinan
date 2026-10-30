"""[ODIN P8] The compile-limits gate: every studied configuration, re-synthesized.

Plan §8, P8, synthesis half. The same pattern as the P5.5a gate one level up:
`synth_xilinx -family xcup` is run again over EVERY non-stock configuration of
the study -- the three cosim-proven generated variants and the two kernel-wrapper
points -- and the resulting record must match `hw/fpga/compile_limits.json`
section for section. The stock row is not re-measured here; it is the P5.5a
artifact, and the fast test asserts the study carries it unchanged.

Three things this gate deliberately re-derives rather than trusts: that the
generated cores still hold their synapse memory IN block RAM, exactly one tile
per 1,024 declared words (the finding the whole bound rests on); that the
capture RAM still costs tiles and ZERO flip-flops per word (the finding that
says the wrapper can ship at its declared depth); and that the wrapper overhead
the bounds reserve was measured at that shipped depth and not at a stand-in.

yosys-synthesizability is NECESSARY, NOT SUFFICIENT for Vivado closure on the
U55C shell, and no bound in the study is a claim about a card until P7b's
`report_utilization` exists.
"""

from __future__ import annotations

import pytest

from integration.odin_rtl_harness import timed

from mimarsinan.chip_simulation.odin_rtl.limits.artifacts import (
    LIMITS_JSON,
    gated_view,
    load_committed,
    measure_all,
)
from mimarsinan.chip_simulation.odin_rtl.limits.configurations import STOCK_KEY
from mimarsinan.chip_simulation.odin_rtl.limits.tables import (
    tile_arrays,
    tiles_for,
)
from mimarsinan.chip_simulation.odin_rtl.synthesis import (
    SynthesisUnavailable,
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
def fresh(tmp_path_factory):
    """One synthesis of every studied configuration, shared by every check."""
    version = _require_yosys()
    root = tmp_path_factory.mktemp("odin_limits")
    with timed("yosys compile-limits sweep (all studied configurations)"):
        record = measure_all(workdir=root, tool_version=version)
    for row in record["configurations"]:
        census = row["census"]
        print(f"[odin-limits] {row['key']}: LUTeq={census['lut_equivalent']:,} "
              f"FF={census['flip_flops']:,} CARRY={census['carry']:,} "
              f"LUTRAM={census['lutram']:,} BRAM36={census['bram36']:,} "
              f"BRAM18={census['bram18']:,} URAM={census['uram']:,}")
    return record


class TestEveryStudiedConfigurationStillSynthesizes:
    def test_every_configuration_produced_a_census(self, fresh):
        for row in fresh["configurations"]:
            assert row["census"]["total_cells"] > 0, row["key"]
            assert row["census"]["lut_equivalent"] > 0, row["key"]

    def test_only_the_stock_row_comes_from_the_p5_5a_artifact(self, fresh):
        origins = {row["key"]: row["measured_by"] for row in fresh["configurations"]}
        assert origins.pop(STOCK_KEY).endswith("synth_resources.json")
        assert set(origins.values()) == {"this run"}


class TestTheFindingsTheBoundsRestOnAreStillTrue:
    def test_every_generated_variant_holds_its_synapse_memory_in_block_ram(
            self, fresh):
        """The registered, address-ahead read of `syn_mem`, re-derived. The
        tile count must be the array's declared DEPTH exactly: one tile short
        and part of the crossbar went back to SLICEM LUTs, one tile long and
        something else moved into the tiles without the study noticing."""
        for row in fresh["configurations"]:
            if row["kind"] != "generated":
                continue
            synapse = next(
                m for m in row["memories"] if m["array"] == "syn_mem")
            assert row["census"]["bram36"] == tiles_for(synapse), (
                f"{row['key']} holds {row['census']['bram36']} RAMB36E2 where "
                f"its {synapse['words']}-word synapse memory needs "
                f"{tiles_for(synapse)}; the bounds in "
                f"docs/odin_fpga_compile_limits_study.md are derived from that "
                f"tile count and must be re-derived, not patched")
            assert row["census"]["lutram"] * 512 < synapse["bits"], (
                f"{row['key']} put synapse-sized distributed RAM back on the "
                f"fabric: the LUT bound is stale")

    def test_the_capture_ram_costs_tiles_and_not_flip_flops_per_word(self, fresh):
        """The single-write-port capture RAM, re-derived. A per-word flip-flop
        cost coming back means the header writes escaped the arbiter and the
        wrapper cannot be built at its shipped depth again."""
        slope = fresh["capture_ram_slope"]["per_capture_word"]
        assert slope["flip_flops"] == 0.0, (
            f"the capture RAM costs {slope['flip_flops']} flip-flops per word "
            f"again; the wrapper's shipped-depth verdict in the study is "
            f"derived from that being zero")
        assert slope["bram36"] == pytest.approx(1 / 1024)

    def test_the_wrapper_is_measured_at_the_depth_it_ships_with(self, fresh):
        """The overhead the bounds reserve is a real kernel's, not a stand-in's."""
        overhead = fresh["wrapper_overhead"]
        wrapper = next(row for row in fresh["configurations"]
                       if row["key"] == overhead["wrapper"])
        assert (overhead["at_parameters"]["CAP_WORDS"]
                == wrapper["geometry"]["shipped_cap_words"])
        assert overhead["delta_costs"]["flip_flops"] > 0
        assert overhead["delta_costs"]["bram36"] == sum(
            tiles_for(memory) for memory in tile_arrays(wrapper))

    def test_the_binding_resource_is_read_off_the_record_and_is_one_of_two(
            self, fresh):
        """The stock core still binds on block RAM; the generated variants no
        longer bind as a family, and the study's verdict section says which."""
        binding = {entry["configuration"]: entry["binding_resource"]
                   for entry in fresh["bounds"]}
        assert binding[STOCK_KEY] == "bram36"
        assert set(binding.values()) <= {"bram36", "lut_sites"}
        generated = {row["key"] for row in fresh["configurations"]
                     if row["kind"] == "generated"}
        assert {binding[key] for key in generated} == {"bram36", "lut_sites"}


class TestTheStudyIsTheCommittedOne:
    def test_the_fresh_sweep_matches_the_committed_record_exactly(self, fresh):
        committed = load_committed()
        if gated_view(fresh) != gated_view(committed):
            differing = sorted(
                key for key, value in gated_view(fresh).items()
                if gated_view(committed).get(key) != value)
            pytest.fail(
                f"the sweep no longer produces the committed numbers in "
                f"{LIMITS_JSON} (sections {differing}).\n"
                f"committed tool: {committed['tool']['version']}\n"
                f"this tool:      {fresh['tool']['version']}\n"
                f"Regenerate DELIBERATELY with "
                f"scripts/hw_tests/regen_compile_limits.py when the RTL, the "
                f"generator or the studied configuration set changed.")
