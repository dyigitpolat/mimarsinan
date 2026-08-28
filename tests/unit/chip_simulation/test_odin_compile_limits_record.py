"""[ODIN P8] The compile-limits record: the bound function, the arithmetic, the doc.

The FAST half of the compile-limits study: no yosys. These pin how a census
becomes a per-core cost, that the packing bound is a FUNCTION rather than a
headline number, that a distributed-RAM cell the study has no conversion for is
REFUSED rather than rounded away, and that the two committed artifacts under
`hw/fpga/` and `docs/` still describe the configurations that are in the repo.
Running the synthesizer is the [slow] gate in `scripts/hw_tests/`.
"""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.odin_rtl.limits.artifacts import (
    LIMITS_JSON,
    SCHEMA,
    STUDY_MD,
    gated_view,
    load_committed,
)
from mimarsinan.chip_simulation.odin_rtl.limits.configurations import (
    STOCK_KEY,
    WRAPPER_CAP_WORDS,
    WRAPPER_FIFO_WORDS,
    configuration_named,
    configurations,
    wrapper_shipped_depths,
)
from mimarsinan.chip_simulation.odin_rtl.limits.device import (
    ASSUMED_SHELL_FRACTION,
    AVAILABILITY_SCENARIOS,
    DEVICE_TOTALS,
    RESOURCE_CLASSES,
    binding_bound,
    census_costs,
    lutram_lut_sites,
    packing_bound,
)
from mimarsinan.chip_simulation.odin_rtl.limits.report import render_study
from mimarsinan.chip_simulation.odin_rtl.synth_artifacts import (
    load_committed as load_synth_record,
)
from mimarsinan.chip_simulation.odin_rtl.synth_census import resource_table
from mimarsinan.mapping.export.odin_gen.variants import PROVEN_VARIANTS


@pytest.fixture(scope="module")
def record():
    return load_committed()


class TestACensusBecomesAPerCoreCost:
    def test_distributed_ram_is_charged_to_the_lut_sites_it_occupies(self):
        census = resource_table({"LUT6": 100, "RAM64M8": 10}).as_record()
        assert lutram_lut_sites(census) == 80
        assert census_costs(census)["lut_sites"] == 100 + 80

    def test_a_half_tile_block_ram_is_half_a_tile_of_cost(self):
        census = resource_table({"RAMB36E2": 3, "RAMB18E2": 1}).as_record()
        assert census_costs(census)["bram36"] == pytest.approx(3.5)

    def test_a_distributed_cell_with_no_conversion_is_refused_not_rounded(self):
        census = resource_table({"RAM128X1D": 4}).as_record()
        with pytest.raises(ValueError, match="RAM128X1D"):
            census_costs(census)


class TestThePackingBoundIsAFunction:
    def test_each_resource_class_gets_its_own_answer(self):
        bounds = packing_bound(
            per_core={"lut_sites": 1_000, "flip_flops": 1_000, "bram36": 10,
                      "uram": 0},
            fixed_overhead={resource: 0.0 for resource in RESOURCE_CLASSES},
            availability=AVAILABILITY_SCENARIOS[0])
        by_resource = {bound.resource: bound.n_max for bound in bounds}
        assert by_resource["lut_sites"] == DEVICE_TOTALS["luts"]["value"] // 1_000
        assert by_resource["bram36"] == DEVICE_TOTALS["bram36_tiles"]["value"] // 10
        assert binding_bound(bounds).resource == "bram36"

    def test_a_class_the_design_never_uses_has_no_bound_rather_than_infinity(self):
        bounds = packing_bound(
            per_core={"lut_sites": 1_000}, fixed_overhead={},
            availability=AVAILABILITY_SCENARIOS[0])
        assert {b.resource: b.n_max for b in bounds}["uram"] is None
        assert binding_bound(bounds).resource == "lut_sites"

    def test_the_fixed_overhead_is_taken_off_the_top_once(self):
        naked = packing_bound(
            per_core={"bram36": 10}, fixed_overhead={},
            availability=AVAILABILITY_SCENARIOS[0])
        reserved = packing_bound(
            per_core={"bram36": 10}, fixed_overhead={"bram36": 100.0},
            availability=AVAILABILITY_SCENARIOS[0])
        assert binding_bound(naked).n_max - binding_bound(reserved).n_max == 10

    def test_the_shell_scenario_is_an_assumption_that_says_so(self):
        datasheet, conservative = AVAILABILITY_SCENARIOS
        assert datasheet.shell_fraction == 0.0
        assert conservative.shell_fraction == ASSUMED_SHELL_FRACTION
        assert "ASSUMPTION" in conservative.status
        assert "P7b" in conservative.status
        assert (conservative.available()["luts"]
                < datasheet.available()["luts"] == DEVICE_TOTALS["luts"]["value"])

    def test_every_device_total_carries_its_provenance(self):
        for column, entry in DEVICE_TOTALS.items():
            assert entry["published"] and entry["derivation"], column
            assert entry["value"] > 0


class TestTheStudiedConfigurationsAreTheProvenOnes:
    def test_every_cosim_proven_variant_is_costed(self):
        keys = {configuration.key for configuration in configurations()}
        assert {variant.name for variant in PROVEN_VARIANTS} <= keys

    def test_the_wrapper_is_measured_at_two_capture_depths_and_nothing_else(self):
        wrappers = [c for c in configurations() if c.kind == "wrapper"]
        assert len(wrappers) == len(WRAPPER_CAP_WORDS) >= 2
        assert {c.geometry["cap_words"] for c in wrappers} == set(WRAPPER_CAP_WORDS)
        assert {c.geometry["fifo_words"] for c in wrappers} == {WRAPPER_FIFO_WORDS}

    def test_both_wrapper_depths_are_read_from_the_rtl_not_assumed(self):
        """And BOTH are MEASURED, not extrapolated to: nothing is shrunk any
        more. The shipped `CAP_WORDS` is itself one of the two synthesized
        points and the shipped `FIFO_WORDS` is the depth both carry, so the
        wrapper overhead the bounds reserve is the overhead of the kernel that
        actually ships."""
        depths = wrapper_shipped_depths()
        assert depths["CAP_WORDS"] == min(WRAPPER_CAP_WORDS)
        assert depths["FIFO_WORDS"] == WRAPPER_FIFO_WORDS

    def test_a_configuration_the_study_does_not_carry_is_refused(self):
        with pytest.raises(KeyError, match="not a studied configuration"):
            configuration_named("gen_a1024n1024_mb32")

    def test_a_generated_variant_declares_the_three_arrays_the_template_has(self):
        variant = configuration_named(PROVEN_VARIANTS[0].name)
        assert [memory["array"] for memory in variant.memories] == [
            "syn_mem", "thr_arr", "vmem_arr"]
        assert variant.memory_bits == sum(m["bits"] for m in variant.memories)


class TestTheCommittedArtifactsDescribeThisTree:
    def test_the_json_is_the_schema_the_gate_reads(self, record):
        assert record["schema"] == SCHEMA
        assert [row["key"] for row in record["configurations"]] == [
            configuration.key for configuration in configurations()]

    def test_the_stock_row_is_the_p5_5a_artifact_and_not_a_second_measurement(
            self, record):
        stock = next(r for r in record["configurations"] if r["key"] == STOCK_KEY)
        committed = load_synth_record()
        assert stock["cells_by_type"] == committed["cells_by_type"]
        assert stock["script"] == committed["script"]
        assert stock["census"] == committed["per_core"]

    def test_every_row_census_is_its_own_cell_census(self, record):
        for row in record["configurations"]:
            assert resource_table(row["cells_by_type"]).as_record() == row["census"]

    def test_the_capture_slope_is_a_subtraction_of_two_recorded_rows(self, record):
        slope = record["capture_ram_slope"]
        rows = {row["key"]: row for row in record["configurations"]}
        low, high = (rows[key] for key in slope["points"])
        words = (high["geometry"]["cap_words"] - low["geometry"]["cap_words"])
        assert words == slope["extra_capture_words"]
        assert (high["census"]["flip_flops"] - low["census"]["flip_flops"]
                == slope["delta"]["flip_flops"])
        assert (slope["per_capture_word"]["flip_flops"]
                == slope["delta"]["flip_flops"] / words)

    def test_every_bound_is_the_bound_function_over_the_recorded_censuses(
            self, record):
        rows = {row["key"]: row for row in record["configurations"]}
        scenarios = {scenario.key: scenario for scenario in AVAILABILITY_SCENARIOS}
        for entry in record["bounds"]:
            fixed = (record["wrapper_overhead"]["delta_costs"]
                     if entry["wrapper_overhead_reserved"] else {})
            bounds = packing_bound(
                per_core=census_costs(rows[entry["configuration"]]["census"]),
                fixed_overhead={k: max(0.0, v) for k, v in fixed.items()},
                availability=scenarios[entry["scenario"]])
            assert [b.as_record() for b in bounds] == entry["bounds"]
            assert binding_bound(bounds).n_max == entry["n_max"]

    def test_no_bound_is_reported_without_both_availability_scenarios(self, record):
        seen = {(entry["configuration"], entry["scenario"])
                for entry in record["bounds"]}
        cores = [row["key"] for row in record["configurations"]
                 if row["kind"] != "wrapper"]
        for key in cores:
            for scenario in AVAILABILITY_SCENARIOS:
                assert (key, scenario.key) in seen

    def test_the_gate_compares_the_numbers_and_ignores_the_tool_banner(self, record):
        view = gated_view(record)
        assert "tool" not in view
        louder = {**record, "tool": {"name": "yosys", "version": "Yosys 9.99"}}
        assert gated_view(louder) == view

    def test_the_study_doc_quotes_the_committed_record(self, record):
        assert STUDY_MD.read_text(encoding="utf-8") == render_study(record)

    def test_the_study_keeps_the_shell_caveat_and_names_no_headline_number(
            self, record):
        study = render_study(record)
        assert "UNKNOWN UNTIL P7b" in study
        assert "report_utilization" in study
        assert "There is no single headline number" in study
        assert LIMITS_JSON.name in study
        assert "yosys is not Vivado" in study
