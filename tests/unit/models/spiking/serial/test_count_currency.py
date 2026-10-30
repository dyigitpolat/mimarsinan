"""``count_ceiling``: the count currency is CHIP-CLAIMED, never universal.

Phase C2 proved the 127 was not fabric — the generated core carries no count
field at all and multiplicity is k adjacent AER transactions. What it did not
prove is that 127 is the right REPRESENTATION for every chip, and this is the
SSOT that says which chip gets which ceiling. Every number below is derived by
hand from the declared width, not read back from the implementation.
"""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.models.spiking.serial.refusals import (
    COUNT_CURRENCY_FLOOR_BITS,
    COUNT_CURRENCY_LIMIT,
    COUNT_CURRENCY_WORD_BITS,
    EMISSION_COUNT_CEILING,
    count_ceiling,
)


def _law(bits: int) -> SomaLaw:
    return SomaLaw(
        firing_mode="Novena", thresholding_mode="<=",
        firing_granularity="per_event",
        membrane_arithmetic="saturating_unsigned", membrane_bits=bits,
    )


class TestTheCurrencyIsTheChipsDeclaredWidth:
    def test_an_eight_bit_chip_carries_the_byte_the_stock_fabric_always_did(self):
        assert count_ceiling(_law(8)) == 127
        assert count_ceiling({"membrane_bits": 8}) == 127

    def test_a_sixteen_bit_chip_carries_a_signed_sixteen_bit_word(self):
        assert count_ceiling(_law(16)) == 32767
        assert count_ceiling(_law(16)) == (1 << 15) - 1

    def test_the_default_point_is_the_byte_and_it_is_the_module_constant(self):
        assert EMISSION_COUNT_CEILING == 127
        assert count_ceiling(None) == EMISSION_COUNT_CEILING
        assert count_ceiling(_law(0)) == EMISSION_COUNT_CEILING


class TestTheCurrencyIsNotAGeometryLimit:
    """C2's finding, kept as a property: only the REGISTER moves the currency."""

    def test_a_wider_crossbar_does_not_move_it(self):
        from mimarsinan.mapping.export.odin_gen.variants import (
            per_event_law,
            spec_for,
        )
        narrow = spec_for(per_event_law(8), axons=128, neurons=128)
        wide = spec_for(per_event_law(8), axons=1024, neurons=256,
                        weight_bits=8)
        assert wide.max_axons > narrow.max_axons
        assert wide.weight_bits > narrow.weight_bits
        assert count_ceiling(wide) == count_ceiling(narrow) == 127

    def test_only_the_declared_register_width_moves_it(self):
        from mimarsinan.mapping.export.odin_gen.variants import (
            per_event_law,
            spec_for,
        )
        same_geometry_wider_register = spec_for(
            per_event_law(16), axons=128, neurons=128)
        assert count_ceiling(same_geometry_wider_register) == 32767


class TestTheCurrencyNeverOutrunsAnImplementation:
    def test_no_chip_gets_more_than_the_currency_word_holds(self):
        assert COUNT_CURRENCY_LIMIT == (1 << (COUNT_CURRENCY_WORD_BITS - 1)) - 1
        for bits in (17, 24, 30, 64):
            assert count_ceiling(_law(bits)) == COUNT_CURRENCY_LIMIT

    def test_a_sub_byte_register_still_gets_the_bytes_currency(self):
        assert COUNT_CURRENCY_FLOOR_BITS == 8
        for bits in (1, 3, 7):
            assert count_ceiling(_law(bits)) == 127


class TestTheClaimsSurfaceIsTotal:
    """A SomaLaw, a CoreSpec, a ChipConfig and a sealed bundle's block, one call."""

    def test_every_claims_shape_answers_the_same_ceiling(self):
        from mimarsinan.chip_simulation.odin_fpga.chip_configs import (
            STOCK_CHIP,
            WIDE_CHIP,
            chip_config_named,
        )
        stock = chip_config_named(STOCK_CHIP)
        wide = chip_config_named(WIDE_CHIP)
        assert count_ceiling(stock) == count_ceiling(stock.core_spec) == 127
        assert count_ceiling(wide) == count_ceiling(wide.core_spec) == 32767
        assert count_ceiling(wide.core_spec.soma_law) == 32767
        # The sealed bundle's own block, read the way a bundle spells it.
        assert count_ceiling({"soma_law": {"membrane_bits": 16}}) == 32767
        assert count_ceiling(wide.bundle_claims()) == 32767

    @pytest.mark.parametrize(
        "claims", [None, {}, object(), {"membrane_bits": None},
                   {"membrane_bits": "wide"}, {"membrane_bits": True},
                   {"membrane_bits": -4}, {"soma_law": "unset"}])
    def test_an_undeclared_or_malformed_width_reads_as_undeclared(self, claims):
        """Totality, exactly like ``soma_axes.resolved_membrane_bits``: the
        registry's own bounds error is the single truth about a bad claim."""
        assert count_ceiling(claims) == EMISSION_COUNT_CEILING


class TestTheStockFabricsBoundIsPinned:
    def test_the_stock_chip_keeps_the_ceiling_it_has_always_had(self):
        from mimarsinan.chip_simulation.odin_fpga.chip_configs import (
            STOCK_CHIP,
            chip_config_named,
        )
        from mimarsinan.mapping.export.odin.feasibility import EMISSION_CEILING

        stock = chip_config_named(STOCK_CHIP)
        assert stock.core_spec.membrane_bits == 8
        assert count_ceiling(stock.core_spec) == EMISSION_CEILING == 127
