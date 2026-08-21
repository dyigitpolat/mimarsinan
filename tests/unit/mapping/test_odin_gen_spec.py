"""[ODIN6] ``CoreSpec`` is a PROJECTION, and it refuses what it cannot emit.

The spec exists so the generated RTL, the packer tables and the descriptor
cannot drift from the platform a deployment was mapped against: it is READ from
one declared core type (the ``CORE_FIELDS`` names) and the resolved ``SomaLaw``,
and everything downstream is derived from it. What it cannot represent it
REFUSES by name — a generator that quietly rounds a geometry emits a chip the
mapping was never checked against.
"""

import pytest

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.config_schema.registry.parse import CORE_FIELDS
from mimarsinan.mapping.export.odin_gen.descriptor import build_descriptor
from mimarsinan.mapping.export.odin_gen.spec import (
    CORE_TYPE_FIELDS,
    CoreSpec,
    CoreSpecError,
    require_generatable,
)

_LIF = {"spiking_family": "lif", "spiking_variant": "streamed",
        "firing_mode": "Novena", "thresholding_mode": "<="}
_PER_EVENT = SomaLaw.resolve({**_LIF, "firing_granularity": "per_event",
                              "membrane_bits": 8})
_SYNC_FIRE = SomaLaw.resolve({**_LIF, "membrane_bits": 16,
                              "membrane_signed": True})
_UNBOUNDED = SomaLaw.resolve(dict(_LIF))

_CORE_TYPE = {"max_axons": 128, "max_neurons": 256, "count": 4,
              "has_bias": False}


def _spec(core_type=None, law=_PER_EVENT, **kwargs):
    merged = {**_CORE_TYPE, **(core_type or {})}
    return CoreSpec.project(
        merged, soma_law=law,
        weight_bits=kwargs.pop("weight_bits", 4),
        weight_sign_granularity=kwargs.pop(
            "weight_sign_granularity", "per_synapse"),
    )


class TestTheSpecIsReadFromTheDeclaration:
    def test_it_uses_the_declared_core_field_names(self):
        assert set(CORE_TYPE_FIELDS) == set(CORE_FIELDS)

    def test_every_declared_field_survives_the_projection(self):
        spec = _spec()
        assert spec.core_type() == _CORE_TYPE

    def test_a_core_type_missing_its_geometry_is_refused_not_defaulted(self):
        with pytest.raises(CoreSpecError, match="max_neurons"):
            CoreSpec.project(
                {"max_axons": 128}, soma_law=_PER_EVENT, weight_bits=4,
                weight_sign_granularity="per_synapse")

    def test_the_law_is_carried_whole_and_not_re_derived(self):
        spec = _spec(law=_SYNC_FIRE)
        assert spec.soma_law is _SYNC_FIRE
        assert spec.membrane_bits == 16
        assert spec.membrane_signed is True
        assert spec.per_event is False
        assert spec.asserts_no_saturation is True

    def test_two_specs_of_the_same_declaration_are_equal(self):
        assert _spec() == _spec()
        assert _spec() != _spec(law=_SYNC_FIRE)

    def test_the_spec_key_separates_the_geometry_and_the_register(self):
        assert _spec().spec_key() == "a128n256w4m8u_pe"
        assert _spec(law=_SYNC_FIRE).spec_key() == "a128n256w4m16s_pc"


class TestDerivedWidths:
    def test_the_address_widths_follow_the_geometry(self):
        spec = _spec({"max_axons": 512, "max_neurons": 128})
        assert spec.axon_address_bits == 9
        assert spec.neuron_address_bits == 7

    def test_the_synapse_image_tiles_the_programming_word(self):
        spec = _spec({"max_axons": 512, "max_neurons": 256})
        assert spec.cells_per_word == 8
        assert spec.synapse_words_per_row == 32
        assert spec.synapse_depth == 512 * 32
        assert spec.synapse_address_bits == 14

    def test_the_register_interval_comes_from_the_law(self):
        assert (_spec().membrane_low, _spec().membrane_high) == (0, 255)
        signed = _spec(law=_SYNC_FIRE)
        assert (signed.membrane_low, signed.membrane_high) == (-32768, 32767)

    def test_the_generated_cell_signs_itself_so_a_slot_costs_one_row(self):
        assert _spec().physical_row_factor == 1


class TestItRefusesWhatItCannotEmit:
    def test_a_non_power_of_two_geometry_is_refused_by_name(self):
        with pytest.raises(CoreSpecError, match="max_axons=100"):
            require_generatable(_spec({"max_axons": 100}))

    def test_a_geometry_over_the_v1_ceiling_is_refused(self):
        with pytest.raises(CoreSpecError, match="max_neurons=1024"):
            require_generatable(_spec({"max_neurons": 1024}))

    def test_an_undeclared_membrane_width_is_refused(self):
        with pytest.raises(CoreSpecError, match="membrane_bits"):
            require_generatable(_spec(law=_UNBOUNDED))

    def test_a_width_the_generator_does_not_emit_is_refused(self):
        law = SomaLaw.resolve({**_LIF, "firing_granularity": "per_event",
                               "membrane_bits": 12})
        with pytest.raises(CoreSpecError, match="membrane_bits=12"):
            require_generatable(_spec(law=law))

    def test_an_on_chip_bias_lane_is_refused_with_its_remedy(self):
        with pytest.raises(CoreSpecError, match="has_bias=false"):
            require_generatable(_spec({"has_bias": True}))

    def test_the_stock_row_pair_layout_is_refused_by_the_generator(self):
        with pytest.raises(CoreSpecError, match="per_axon"):
            require_generatable(_spec(weight_sign_granularity="per_axon"))

    def test_an_unknown_sign_granularity_is_refused(self):
        with pytest.raises(ValueError, match="weight_sign_granularity"):
            require_generatable(_spec(weight_sign_granularity="per_core"))

    def test_a_signed_register_under_the_event_serial_law_is_refused(self):
        law = SomaLaw(
            firing_mode="Novena", thresholding_mode="<=",
            firing_granularity="per_event",
            membrane_arithmetic="saturating_signed", membrane_bits=16)
        with pytest.raises(CoreSpecError, match="floors at zero"):
            require_generatable(_spec(law=law))

    def test_a_subtractive_reset_under_the_event_serial_law_is_refused(self):
        law = SomaLaw(
            firing_mode="Default", thresholding_mode="<=",
            firing_granularity="per_event",
            membrane_arithmetic="saturating_unsigned", membrane_bits=8)
        with pytest.raises(CoreSpecError, match="Novena"):
            require_generatable(_spec(law=law))

    def test_a_weight_wider_than_its_membrane_is_refused(self):
        law = SomaLaw.resolve({**_LIF, "firing_granularity": "per_event",
                               "membrane_bits": 8})
        with pytest.raises(CoreSpecError, match="wider than"):
            require_generatable(_spec(law=law, weight_bits=16))

    def test_a_geometry_that_does_not_tile_the_word_is_refused(self):
        with pytest.raises(CoreSpecError, match="tile"):
            require_generatable(_spec({"max_neurons": 4}, weight_bits=2))

    @pytest.mark.parametrize("axons,neurons", [(128, 128), (512, 256), (256, 256)])
    def test_the_gates_shipped_geometries_are_admitted(self, axons, neurons):
        for law in (_PER_EVENT, _SYNC_FIRE):
            require_generatable(_spec({"max_axons": axons, "max_neurons": neurons},
                                      law=law))


class TestTheDescriptorEchoesOnlyEstablishedValues:
    def test_it_carries_the_declaration_the_law_and_the_conventions(self):
        spec = _spec(law=_SYNC_FIRE)
        descriptor = build_descriptor(
            spec, files=["odin_gen_core.v"], vendored=False)
        assert descriptor["spec_key"] == spec.spec_key()
        assert descriptor["geometry"]["core_type"] == _CORE_TYPE
        assert descriptor["soma_law"]["membrane_interval"] == [-32768, 32767]
        assert descriptor["soma_law"]["point_tag"] == "ssat16"
        assert descriptor["soma_law"]["asserts_no_saturation"] is True
        assert descriptor["programming"]["selectors"]["membrane"] == 2
        assert descriptor["module"] == "odin_gen_core"
        assert descriptor["vendored"] is False

    def test_a_vendored_descriptor_names_no_generated_module(self):
        descriptor = build_descriptor(
            _spec(), files=["ODIN.v"], vendored=True)
        assert descriptor["vendored"] is True
        assert descriptor["module"] is None
