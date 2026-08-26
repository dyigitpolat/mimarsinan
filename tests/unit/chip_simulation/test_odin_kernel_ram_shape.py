"""The kernel's block RAMs keep the shape a tile can be: one read, one write.

This is a SILICON gate expressed in source. `prog_ram` was read in five places
under two sequencer states — the opcode plus three arguments plus the TAG alias,
all at `pc` — which yosys merged into one port and Vivado 2022.2 did not: the
routed U55C build put the whole 262,144x32 array into distributed RAM (163,840
LUTs as RAM, 24 BRAM tiles total, none of them the program). Nothing local
re-runs Vivado, so the invariant the fix rests on is pinned HERE instead: every
array declared `ram_style = "block"` is indexed in exactly one read place and
one write place.
"""

from __future__ import annotations

from mimarsinan.chip_simulation.odin_fpga.kernel_sim import (
    KERNEL_ROOT,
    block_ram_ports,
    kernel_sources,
)


class TestEveryBlockRamIsSinglePorted:
    def test_the_kernel_tree_declares_the_two_block_rams(self):
        declared = {
            name
            for source in kernel_sources()
            for name in block_ram_ports(source)
        }
        assert declared == {"prog_ram", "cap_ram"}

    def test_each_block_ram_has_exactly_one_read_and_one_write_point(self):
        for source in kernel_sources():
            for array, (reads, writes) in block_ram_ports(source).items():
                assert (reads, writes) == (1, 1), (
                    f"{source.name}: `{array}` is indexed in {reads} read and "
                    f"{writes} write places; a tile is one of each, and a "
                    "synthesizer that infers more falls back to LUTRAM")

    def test_the_program_ram_is_read_through_the_named_register(self):
        text = (KERNEL_ROOT / "odin_fpga_kernel.v").read_text()
        assert "prog_rdata <= prog_ram[pc[PROG_AW-1:0]];" in text, (
            "the sequencer's one read point is not the clocked prog_rdata "
            "register at the single address source `pc`")


class TestTheReferenceScannerSeesWhatItClaims:
    def test_a_second_read_point_is_counted(self, tmp_path):
        source = tmp_path / "two_reads.v"
        source.write_text(
            '(* ram_style = "block" *) reg [31:0] ram [0:7];\n'
            "always @(posedge clk) ram[waddr[2:0]] <= wdata;\n"
            "always @(posedge clk) a <= ram[raddr[2:0]];\n"
            "always @(posedge clk) b <= ram[raddr[2:0]];\n")
        assert block_ram_ports(source) == {"ram": (2, 1)}

    def test_a_declaration_is_not_itself_a_reference(self, tmp_path):
        source = tmp_path / "one_each.v"
        source.write_text(
            '(* ram_style = "block" *) reg [31:0] ram [0:7];\n'
            "always @(posedge clk) begin\n"
            "    q <= ram[raddr[2:0]];\n"
            "    if (we) ram[waddr[2:0]] <= wdata;\n"
            "end\n")
        assert block_ram_ports(source) == {"ram": (1, 1)}

    def test_an_array_without_the_attribute_is_not_reported(self, tmp_path):
        source = tmp_path / "plain.v"
        source.write_text(
            "reg [31:0] ram [0:7];\n"
            "always @(posedge clk) q <= ram[raddr[2:0]];\n")
        assert block_ram_ports(source) == {}
