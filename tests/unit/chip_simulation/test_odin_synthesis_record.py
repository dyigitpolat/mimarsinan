"""[ODIN5.5a] The synthesis record: the cell census, the loud BRAM finding, the artifact.

These are the FAST half of plan §7 row 19: they need no yosys. They pin how a
yosys `stat -json` census becomes the per-core resource table, that a memory
which does NOT reach block RAM is reported as a finding rather than absorbed,
and that the two committed artifacts under `hw/fpga/` still describe the RTL
tree that is actually in the repo. Running the synthesizer itself is the [slow]
gate in `scripts/hw_tests/`.
"""

from __future__ import annotations

import json

import pytest

from mimarsinan.chip_simulation.odin_rtl.synth_artifacts import (
    RESOURCES_JSON,
    REPORT_MD,
    SCHEMA,
    gated_view,
    load_committed,
    render_report,
    source_names,
)
from mimarsinan.chip_simulation.odin_rtl.synth_census import (
    NEURON_MEMORY_MODULE,
    SYNAPSE_MEMORY_MODULE,
    SynthesisError,
    memory_inferences,
    primitive_cells,
    resource_table,
)
from mimarsinan.chip_simulation.odin_rtl.synthesis import (
    TARGET_FAMILY,
    TOP_MODULE,
    script_lines,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    OVERLAY_MEM,
    VENDOR_SRC,
    design_sources,
)


def _stat(design_cells, modules=None):
    """A minimal yosys `stat -top ... -json` document."""
    return {
        "creator": "Yosys (test double)",
        "modules": dict(modules or {}),
        "design": {
            "num_cells": sum(design_cells.values()),
            "num_cells_by_type": dict(design_cells),
        },
    }


def _memory_module(cells):
    return {"num_cells": sum(cells.values()), "num_cells_by_type": dict(cells)}


def _both_memories(cells):
    return {
        f"\\{NEURON_MEMORY_MODULE}": _memory_module(cells),
        f"\\{SYNAPSE_MEMORY_MODULE}": _memory_module(cells),
    }


class TestTheCellCensusBecomesTheResourceTable:
    def test_every_primitive_family_lands_in_its_own_column(self):
        table = resource_table({
            "LUT1": 2, "LUT2": 985, "LUT6": 1341,
            "INV": 450,
            "FDRE": 3313, "FDCE": 919, "FDPE": 59, "FDRE_1": 1,
            "CARRY4": 381, "MUXF7": 32, "MUXF8": 13,
            "RAMB36E2": 10, "IBUF": 23, "OBUF": 11, "BUFG": 2,
        })
        assert table.luts == 2 + 985 + 1341
        assert table.inverters == 450
        assert table.flip_flops == 3313 + 919 + 59 + 1
        assert table.carry == 381
        assert table.muxf == 32 + 13
        assert table.bram36 == 10
        assert table.bram18 == 0
        assert table.uram == 0
        assert table.io_buffers == 23 + 11 + 2
        assert table.lutram == 0
        assert table.unclassified == {}

    def test_a_half_width_block_ram_counts_as_half_a_tile(self):
        table = resource_table({"RAMB36E2": 3, "RAMB18E2": 5})
        assert (table.bram36, table.bram18) == (3, 5)
        assert table.bram_tiles == pytest.approx(5.5)

    def test_ultra_ram_is_counted_apart_from_block_ram(self):
        table = resource_table({"URAM288": 4})
        assert (table.uram, table.bram_tiles) == (4, 0.0)

    def test_distributed_ram_has_its_own_column_and_is_not_unclassified(self):
        table = resource_table({"LUT6": 4, "RAM64M8": 164, "SRL16E": 3})
        assert table.lutram_cells == {"RAM64M8": 164, "SRL16E": 3}
        assert table.lutram == 167
        assert table.unclassified == {}
        assert table.as_record()["lutram_cells"] == {"RAM64M8": 164, "SRL16E": 3}

    def test_block_ram_is_never_counted_as_distributed_ram(self):
        table = resource_table({"RAMB36E2": 10, "RAMB18E2": 1, "URAM288": 2})
        assert table.lutram_cells == {}
        assert (table.bram36, table.bram18, table.uram) == (10, 1, 2)

    def test_a_cell_no_column_claims_is_named_not_absorbed(self):
        table = resource_table({"LUT6": 1, "DSP48E2": 7})
        assert table.unclassified == {"DSP48E2": 7}
        assert table.total_cells == 8
        assert "DSP48E2" in table.as_record()["unclassified"]

    def test_the_columns_and_the_leftovers_account_for_every_cell(self):
        cells = {"LUT4": 9, "FDRE": 4, "CARRY8": 2, "RAMB18E2": 1, "RAM64M8": 3,
                 "PS8": 1}
        table = resource_table(cells)
        counted = (table.luts + table.inverters + table.flip_flops + table.carry
                   + table.muxf + table.bram36 + table.bram18 + table.uram
                   + table.lutram + table.io_buffers
                   + sum(table.unclassified.values()))
        assert counted == table.total_cells == sum(cells.values())

    def test_hierarchy_instances_are_not_primitives(self):
        stat = _stat(
            {"LUT6": 4, "$paramod$abc\\scheduler": 1, "SRAM_256x128_wrapper": 1},
            modules={
                "$paramod$abc\\scheduler": _memory_module({}),
                "\\SRAM_256x128_wrapper": _memory_module({}),
            },
        )
        assert primitive_cells(stat) == {"LUT6": 4}

    def test_a_census_yosys_did_not_produce_is_refused(self):
        with pytest.raises(SynthesisError, match="stat"):
            primitive_cells({"modules": {}})


class TestABlockRamThatDidNotHappenIsAFinding:
    def test_block_ram_only_is_the_passing_outcome(self):
        inferences = memory_inferences(
            _stat({}, _both_memories({"RAMB36E2": 2, "LUT2": 1})))
        assert [m.module for m in inferences] == [
            NEURON_MEMORY_MODULE, SYNAPSE_MEMORY_MODULE]
        assert all(m.inferred_as_bram for m in inferences)
        assert all(m.finding() is None for m in inferences)

    def test_distributed_ram_is_reported_loudly_and_names_the_cells(self):
        inference = memory_inferences(
            _stat({}, _both_memories({"RAMB36E2": 1, "RAM128X1D": 64})))[0]
        assert not inference.inferred_as_bram
        finding = inference.finding()
        assert finding is not None
        assert "RAM128X1D" in finding and NEURON_MEMORY_MODULE in finding
        assert "distributed" in finding.lower()

    def test_a_shift_register_lookup_table_counts_as_distributed(self):
        inference = memory_inferences(_stat({}, _both_memories({"SRL16E": 8})))[0]
        assert not inference.inferred_as_bram
        assert "SRL16E" in (inference.finding() or "")

    def test_a_memory_flattened_to_flip_flops_is_reported_loudly(self):
        inference = memory_inferences(_stat({}, _both_memories({"FDRE": 8192})))[0]
        assert not inference.inferred_as_bram
        finding = inference.finding()
        assert finding is not None
        assert "no RAMB" in finding and "8192" in finding

    def test_a_memory_module_yosys_dissolved_is_reported_loudly(self):
        inference = memory_inferences(_stat({}, {}))[0]
        assert not (inference.present or inference.inferred_as_bram)
        assert "no `SRAM_256x128_wrapper` module" in (inference.finding() or "")

    def test_the_declared_geometry_travels_with_the_finding(self):
        inferences = memory_inferences(
            _stat({}, _both_memories({"RAMB36E2": 2})))
        assert [(m.words, m.width) for m in inferences] == [(256, 128), (8192, 32)]
        assert inferences[1].as_record()["geometry"] == "8192x32"


class TestTheYosysScriptIsTheOneTheReportQuotes:
    def test_the_overlay_precedes_the_vendor_tree(self):
        sources = design_sources(overlay=True)
        overlay = [p for p in sources if OVERLAY_MEM in p.parents]
        vendor = [p for p in sources if VENDOR_SRC in p.parents]
        assert len(overlay) == 2 and len(vendor) > 5
        assert sources[:len(overlay)] == overlay

    def test_the_first_declaration_wins_flag_is_what_shadows_the_vendor_copy(self):
        lines = script_lines(sources=design_sources(overlay=True), stat_path=None)
        read, synth, stat = lines
        assert read.startswith("read_verilog -nooverwrite ")
        assert read.index("hw/fpga/mem/") < read.index("hw/vendor/odin/src/")
        assert synth == f"synth_xilinx -family {TARGET_FAMILY} -top {TOP_MODULE}"
        assert stat == f"stat -top {TOP_MODULE} -json"

    def test_a_parameter_override_becomes_a_chparam_line_and_nothing_else(self):
        plain = script_lines(sources=design_sources(overlay=True), stat_path=None)
        parameterised = script_lines(
            sources=design_sources(overlay=True), stat_path=None,
            top="odin_fpga_kernel_top", parameters={"CAP_WORDS": 1024})
        assert len(plain) == 3 and len(parameterised) == 4
        assert parameterised[1] == "chparam -set CAP_WORDS 1024 odin_fpga_kernel_top"
        assert parameterised[2].endswith("-top odin_fpga_kernel_top")

    def test_the_script_paths_are_repo_relative_so_the_record_travels(self):
        read = script_lines(sources=design_sources(overlay=True), stat_path=None)[0]
        assert "/home/" not in read


@pytest.fixture(scope="module")
def record():
    return load_committed()


class TestTheCommittedArtifactsDescribeThisTree:
    def test_the_json_is_the_schema_the_gate_reads(self, record):
        assert record["schema"] == SCHEMA
        assert (record["top"], record["family"]) == (TOP_MODULE, TARGET_FAMILY)

    def test_the_committed_numbers_are_the_committed_census(self, record):
        assert (resource_table(record["cells_by_type"]).as_record()
                == record["per_core"])

    def test_the_committed_source_list_is_the_tree_on_disk(self, record):
        assert record["sources"] == source_names(design_sources(overlay=True))

    def test_the_committed_script_is_the_committed_source_list(self, record):
        assert record["script"] == list(
            script_lines(sources=design_sources(overlay=True), stat_path=None))

    def test_both_memories_are_recorded_as_block_ram(self, record):
        outcomes = {m["module"]: m["inferred_as_bram"] for m in record["memories"]}
        assert outcomes == {NEURON_MEMORY_MODULE: True, SYNAPSE_MEMORY_MODULE: True}

    def test_the_vendored_memories_alone_are_recorded_as_not_reaching_block_ram(
            self, record):
        assert record["vendor_only_control"]["block_ram_reached"] is False

    def test_the_gate_compares_the_numbers_and_ignores_the_tool_banner(self, record):
        view = gated_view(record)
        assert "tool" not in view
        assert view["per_core"] == record["per_core"]
        louder = json.loads(json.dumps(record))
        louder["tool"]["version"] = "Yosys 9.99"
        assert gated_view(louder) == view
        louder["per_core"]["luts"] += 1
        assert gated_view(louder) != view

    def test_the_report_quotes_the_committed_record(self, record):
        report = REPORT_MD.read_text(encoding="utf-8")
        assert report == render_report(record)
        for number in (record["per_core"]["luts"], record["per_core"]["flip_flops"],
                       record["per_core"]["bram36"]):
            assert f"{number:,}" in report or str(number) in report
        assert record["tool"]["version"] in report
        assert RESOURCES_JSON.name in report

    def test_the_report_keeps_the_necessary_not_sufficient_caveat(self, record):
        report = render_report(record)
        assert "necessary" in report and "not sufficient" in report
        assert "Vivado" in report and "P7" in report
