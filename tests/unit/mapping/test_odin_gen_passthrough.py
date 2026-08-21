"""[ODIN6, plan §7 row 20] The stock spec is a vendored PASSTHROUGH.

Two claims, and the second is the one that matters. First, generating the stock
`CoreSpec` yields the byte-identical `hw/vendor/odin` file set — asserted inside
`generate_core`, so a template that started rewriting the vendored core could
not be published at all. Second, the OUT-OF-THE-BOX path never depends on
generation: the exporter and the P5 cosimulation harness read the vendored tree
directly, and this module proves that by construction rather than by promise.

The stock spec itself is PROJECTED from the registered `odin_stock_core`
platform, so "the stock core" means one thing in the platform table, in the
exporter and in the generator.
"""

import numpy as np
import pytest

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.export.odin_gen import (
    CoreSpecError,
    generate_core,
    is_stock_spec,
    stock_core_spec,
)
from mimarsinan.mapping.export.odin_gen.generate import (
    DESCRIPTOR_FILENAME,
    write_generated_core,
)
from mimarsinan.mapping.export.odin_gen.passthrough import (
    StockPassthroughError,
    assert_stock_passthrough,
    vendored_file_set,
)
from mimarsinan.mapping.export.odin_gen.render import render_core_rtl
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec
from mimarsinan.mapping.export.odin_gen.templates import (
    HW_VENDOR_ROOT,
    assert_no_placeholders,
)
from mimarsinan.mapping.platform.imc_platforms import get_imc_platform


class TestTheStockSpecIsTheRegisteredPlatform:
    def test_it_projects_the_declared_core_type_verbatim(self):
        platform = get_imc_platform("odin_stock_core")
        spec = stock_core_spec()
        assert spec.core_type() == dict(platform.cores[0])
        assert spec.weight_bits == platform.weight_bits
        assert spec.weight_sign_granularity == "per_axon"
        assert spec.membrane_bits == 8

    def test_it_carries_the_stock_soma_law(self):
        law = stock_core_spec().soma_law
        assert law.is_per_event is True
        assert law.membrane_arithmetic == "saturating_unsigned"
        assert law.firing_mode == "Novena"
        assert law.point_tag() == "per_event-sat8"

    def test_the_stock_spec_recognises_itself_and_nothing_else(self):
        assert is_stock_spec(stock_core_spec()) is True
        variant = CoreSpec.project(
            {"max_axons": 128, "max_neurons": 256, "count": 1,
             "has_bias": False},
            soma_law=stock_core_spec().soma_law, weight_bits=4,
            weight_sign_granularity="per_synapse")
        assert is_stock_spec(variant) is False


class TestGeneratingTheStockSpecReproducesTheVendoredTree:
    def test_it_emits_the_vendored_file_set_byte_for_byte(self):
        generated = generate_core(stock_core_spec())
        assert generated.vendored is True
        assert dict(generated.files) == dict(vendored_file_set())

    def test_every_file_matches_the_tree_on_disk(self):
        generated = generate_core(stock_core_spec())
        for name, payload in generated.files:
            assert payload == (HW_VENDOR_ROOT / "src" / name).read_bytes(), name

    def test_the_vendored_set_is_the_whole_rtl_tree(self):
        names = {name for name, _ in vendored_file_set()}
        on_disk = {
            str(path.relative_to(HW_VENDOR_ROOT / "src"))
            for path in (HW_VENDOR_ROOT / "src").rglob("*.v")
        }
        assert names == on_disk
        assert len(names) == 18

    def test_the_descriptor_says_it_is_vendored_and_names_no_module(self):
        descriptor = generate_core(stock_core_spec()).descriptor
        assert descriptor["vendored"] is True
        assert descriptor["module"] is None
        assert "ODIN.v" in descriptor["files"]

    def test_a_rewritten_file_would_be_refused(self):
        """Teeth: the assertion is what makes the passthrough a claim."""
        files = list(vendored_file_set())
        name, payload = files[0]
        files[0] = (name, payload + b"\n// drift\n")
        with pytest.raises(StockPassthroughError, match=name):
            assert_stock_passthrough(tuple(files))

    def test_a_missing_file_would_be_refused(self):
        with pytest.raises(StockPassthroughError, match="missing"):
            assert_stock_passthrough(tuple(vendored_file_set()[1:]))


class TestTheStockSpecIsNeverRunThroughTheVariantGenerator:
    def test_the_stock_layout_is_not_generatable_as_a_variant(self):
        """The stock crossbar signs a whole row through SPI_SYN_SIGN, which the
        generated cell cannot represent — so the passthrough is not a shortcut
        past a capability, it is the only correct answer."""
        with pytest.raises(CoreSpecError, match="per_axon"):
            render_core_rtl(stock_core_spec())


class TestWritingAGeneratedCore:
    def test_it_writes_the_sources_and_the_descriptor(self, tmp_path):
        law = SomaLaw.resolve({
            "spiking_family": "lif", "spiking_variant": "streamed",
            "firing_mode": "Novena", "thresholding_mode": "<=",
            "firing_granularity": "per_event", "membrane_bits": 8})
        spec = CoreSpec.project(
            {"max_axons": 16, "max_neurons": 16, "count": 1, "has_bias": False},
            soma_law=law, weight_bits=4, weight_sign_granularity="per_synapse")
        written = write_generated_core(generate_core(spec), tmp_path)
        assert {path.name for path in written} == {
            "odin_gen_core.v", DESCRIPTOR_FILENAME}
        assert_no_placeholders((tmp_path / "odin_gen_core.v").read_text())

    def test_the_vendored_tree_can_be_written_out_unchanged(self, tmp_path):
        written = write_generated_core(generate_core(stock_core_spec()), tmp_path)
        for path in written:
            if path.name == DESCRIPTOR_FILENAME:
                continue
            relative = path.relative_to(tmp_path)
            assert path.read_bytes() == (
                HW_VENDOR_ROOT / "src" / relative).read_bytes()
        assert np.all([path.exists() for path in written])
