"""The kernel's block RAMs keep the shape a tile can be: one read, one write.

This is a SILICON gate expressed in source. The program RAM this kernel used to
carry was read in five places under two sequencer states — the opcode plus three
arguments plus the TAG alias, all at `pc` — which yosys merged into one port and
Vivado 2022.2 did not: the routed U55C build put the whole 262,144x32 array into
distributed RAM (163,840 LUTs as RAM, 24 BRAM tiles total, none of them the
program). That array is gone entirely — the op stream is streamed from the host
through `fifo_ram` and never stored — but the invariant it taught is what the
two arrays that remain are held to here, because nothing local re-runs Vivado:
every array declared `ram_style = "block"` is indexed in exactly one read place
and one write place.
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
        assert declared == {"fifo_ram", "cap_ram"}

    def test_each_block_ram_has_exactly_one_read_and_one_write_point(self):
        for source in kernel_sources():
            for array, (reads, writes) in block_ram_ports(source).items():
                assert (reads, writes) == (1, 1), (
                    f"{source.name}: `{array}` is indexed in {reads} read and "
                    f"{writes} write places; a tile is one of each, and a "
                    "synthesizer that infers more falls back to LUTRAM")

    def test_the_stream_fifo_is_read_at_its_one_pointer(self):
        text = (KERNEL_ROOT / "odin_fpga_kernel.v").read_text()
        assert "if (lift) ram_q <= fifo_ram[rd_ptr];" in text, (
            "the FIFO's one read point is not the clocked head register at the "
            "single address source `rd_ptr`")

    def test_no_program_ram_came_back(self):
        """The fabric is the chip; it does not hold a copy of the host's program."""
        text = (KERNEL_ROOT / "odin_fpga_kernel.v").read_text()
        assert "prog_ram" not in text
        assert "PROG_WORDS" not in text


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

    def test_the_two_sides_may_live_in_separate_blocks(self, tmp_path):
        """The capture RAM's read is DMA time and its write is core time, so
        they are two always blocks on one clock — still one port each."""
        source = tmp_path / "split_blocks.v"
        source.write_text(
            '(* ram_style = "block" *) reg [31:0] ram [0:7];\n'
            "always @(posedge clk) q <= ram[raddr[2:0]];\n"
            "always @(posedge clk) if (en && we) ram[waddr[2:0]] <= wdata;\n")
        assert block_ram_ports(source) == {"ram": (1, 1)}

    def test_an_array_without_the_attribute_is_not_reported(self, tmp_path):
        source = tmp_path / "plain.v"
        source.write_text(
            "reg [31:0] ram [0:7];\n"
            "always @(posedge clk) q <= ram[raddr[2:0]];\n")
        assert block_ram_ports(source) == {}
