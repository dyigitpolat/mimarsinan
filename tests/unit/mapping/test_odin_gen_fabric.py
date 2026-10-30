"""The variant FABRIC template: the kernel a chip configuration is built around."""

from __future__ import annotations

import pytest

from mimarsinan.mapping.export.odin_gen.fabric import (
    KERNEL_MODULE,
    KERNEL_TEMPLATE,
    FabricError,
    GeneratedFabric,
    generate_fabric,
    kernel_filename,
    kernel_substitution_table,
    render_variant_kernel,
    write_generated_fabric,
)
from mimarsinan.mapping.export.odin_gen.passthrough import stock_core_spec
from mimarsinan.mapping.export.odin_gen.render import CORE_MODULE, spec_flags_word
from mimarsinan.mapping.export.odin_gen.templates import (
    placeholders_in,
    template_path,
)
from mimarsinan.mapping.export.odin_gen.variants import (
    PROVEN_VARIANTS,
    WIDE_CHIP_VARIANT,
    variant_named,
)

WIDE = variant_named(WIDE_CHIP_VARIANT)


class TestTheTableAndTheTemplateAgreeBothWays:
    def test_every_placeholder_the_template_has_is_in_the_table(self):
        text = template_path(KERNEL_TEMPLATE).read_text(encoding="utf-8")
        assert set(placeholders_in(text)) == set(
            kernel_substitution_table(WIDE.spec))

    @pytest.mark.parametrize(
        "name", [v.name for v in PROVEN_VARIANTS])
    def test_every_proven_variant_renders_without_a_survivor(self, name):
        spec = variant_named(name).spec
        text = render_variant_kernel(spec)
        assert placeholders_in(text) == ()
        assert f"module {KERNEL_MODULE} #(" in text

    def test_the_table_is_the_spec_and_nothing_else(self):
        spec = WIDE.spec
        assert kernel_substitution_table(spec) == {
            "AXONS": 1024, "NEURONS": 256, "AW": 10, "NW": 8,
            "MBITS": 16, "WBITS": 8, "FLAGS": spec_flags_word(spec),
        }


class TestTheEmittedKernelIsTheOneTheWrapperBinds:
    def test_the_module_name_is_the_stock_kernel_s(self):
        """A chip configuration selects a FILE, not a parameter: the wrapper
        instantiates `odin_fpga_kernel` and must never learn which body it got."""
        assert KERNEL_MODULE == "odin_fpga_kernel"
        assert kernel_filename() == "odin_fpga_kernel.v"

    def test_the_geometry_reaches_the_localparams(self):
        text = render_variant_kernel(WIDE.spec)
        for line in ("localparam AXONS    = 1024;",
                     "localparam NEURONS  = 256;",
                     "localparam AW       = 10;",
                     "localparam NW       = 8;",
                     "localparam MBITS    = 16;",
                     "localparam WBITS    = 8;"):
            assert line in text, line

    def test_it_carries_the_licence_statement_of_changes(self):
        text = render_variant_kernel(WIDE.spec)
        assert "GENERATED FILE -- do not edit" in text
        assert "MIT licence" in text

    def test_it_instantiates_the_generated_core_and_no_spi_master(self):
        text = render_variant_kernel(WIDE.spec)
        assert f"{CORE_MODULE} dut (" in text
        assert "odin_spi_master" not in text
        assert "ODIN #(" not in text

    def test_it_implements_the_variant_programming_opcode(self):
        text = render_variant_kernel(WIDE.spec)
        assert "OP_PROG   = 32'd7;" in text
        assert "OP_PROG:  begin argc <= 3'd4; state <= S_ARG; end" in text


class TestTheFabricIsCoreAndKernelTogether:
    def test_it_emits_both_files_from_one_spec(self):
        fabric = generate_fabric(WIDE.spec)
        assert isinstance(fabric, GeneratedFabric)
        assert [name for name, _ in fabric.files] == [
            "odin_gen_core.v", kernel_filename()]
        assert fabric.core.spec == WIDE.spec

    def test_the_core_and_the_kernel_cannot_disagree_about_geometry(self):
        """Both files come out of ONE spec, so the fabric's address widths are
        the core's by construction; the run-time SPEC guard is the belt."""
        fabric = generate_fabric(WIDE.spec)
        kernel = fabric.source_text(kernel_filename())
        core = fabric.source_text("odin_gen_core.v")
        assert f"parameter AXONS         = {WIDE.spec.max_axons}," in core
        assert f"localparam AXONS    = {WIDE.spec.max_axons};" in kernel
        assert "spec_bad_w[c] = (spec_axons_w   != AXONS)" in kernel

    def test_the_stock_spec_has_no_generated_fabric(self):
        with pytest.raises(FabricError, match="vendored passthrough"):
            generate_fabric(stock_core_spec())

    def test_writing_a_fabric_lands_both_files_and_the_descriptor(self, tmp_path):
        fabric = generate_fabric(WIDE.spec)
        written = write_generated_fabric(fabric, tmp_path)
        assert sorted(path.name for path in written) == [
            "core_descriptor.json", "odin_fpga_kernel.v", "odin_gen_core.v"]
        assert (tmp_path / kernel_filename()).read_bytes() == dict(
            fabric.files)[kernel_filename()]

    def test_an_unknown_file_is_named_rather_than_returned_empty(self):
        fabric = generate_fabric(WIDE.spec)
        with pytest.raises(KeyError, match="not one of this fabric's files"):
            fabric.source_text("odin_nothing.v")
