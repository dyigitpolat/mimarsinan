"""[ODIN P8] The compile-limits gate: every studied configuration, re-synthesized.

Plan §8, P8, synthesis half. The same pattern as the P5.5a gate one level up:
`synth_xilinx -family xcup` is run again over EVERY non-stock configuration of
the study -- the three cosim-proven generated variants and the two kernel-wrapper
points -- and the resulting record must match `hw/fpga/compile_limits.json`
section for section. The stock row is not re-measured here; it is the P5.5a
artifact, and the fast test asserts the study carries it unchanged.

Two things this gate deliberately re-derives rather than trusts: that the
generated cores still keep their synapse memory OUT of block RAM (the finding
the whole LUT bound rests on), and that the capture RAM still costs 32
flip-flops per word (the finding that says the wrapper cannot ship as written).

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
    def test_no_generated_variant_reaches_block_ram_for_its_synapse_memory(
            self, fresh):
        for row in fresh["configurations"]:
            if row["kind"] != "generated":
                continue
            synapse_bits = next(
                m["bits"] for m in row["memories"] if m["array"] == "syn_mem")
            held_in_bram = row["census"]["bram36"] * 36 * 1024
            assert row["census"]["lutram"] > 0, row["key"]
            assert held_in_bram < synapse_bits, (
                f"{row['key']} now holds its synapse memory in block RAM; the "
                f"LUT-bound analysis in docs/odin_fpga_compile_limits_study.md "
                f"is stale and must be re-derived, not patched")

    def test_the_capture_ram_still_costs_flip_flops_per_word(self, fresh):
        slope = fresh["capture_ram_slope"]["per_capture_word"]
        assert slope["flip_flops"] == 32.0, (
            f"the capture RAM's per-word flip-flop cost moved to "
            f"{slope['flip_flops']}; the wrapper's shipped-depth verdict in the "
            f"study is derived from it")
        assert slope["bram36"] == 0.0

    def test_the_stock_core_is_the_one_that_binds_on_block_ram(self, fresh):
        binding = {entry["configuration"]: entry["binding_resource"]
                   for entry in fresh["bounds"]}
        assert binding[STOCK_KEY] == "bram36"
        assert set(binding.values()) - {"bram36"} == {"lut_sites"}


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
