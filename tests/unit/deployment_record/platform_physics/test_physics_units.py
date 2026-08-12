"""Unit conversion is dimensional: a wrong-dimension declaration must be refused."""

import pytest

from mimarsinan.deployment_record.platform_physics.units import (
    AREA,
    DATA,
    DIMENSIONLESS,
    DIMENSIONS,
    ENERGY,
    POWER,
    TIME,
    canonical_symbol,
    to_canonical,
    unit_for,
    units_in,
)


def test_every_dimension_has_a_canonical_unit_of_scale_one():
    for dimension in DIMENSIONS:
        canonical = unit_for(canonical_symbol(dimension))
        assert canonical.dimension == dimension
        assert canonical.si_scale == 1.0


def test_canonical_units_are_si():
    assert canonical_symbol(ENERGY) == "J"
    assert canonical_symbol(AREA) == "m^2"
    assert canonical_symbol(TIME) == "s"
    assert canonical_symbol(POWER) == "W"
    assert canonical_symbol(DATA) == "B"
    assert canonical_symbol(DIMENSIONLESS) == "1"


@pytest.mark.parametrize(
    "value,symbol,expected",
    [
        (1.0, "pJ", 1e-12),
        (23.6, "pJ", 23.6e-12),
        (1.0, "nJ", 1e-9),
        (1.0, "J", 1.0),
        (1.0, "um^2", 1e-12),
        (0.41, "mm^2", 0.41e-6),
        (5.0, "ns", 5e-9),
        (1.0, "s", 1.0),
        (65.0, "mW", 65e-3),
        (8.0, "B", 8.0),
        (1.0, "KiB", 1024.0),
        (24.0, "bit", 24.0),
        (256.0, "levels", 256.0),
    ],
)
def test_to_canonical_scales_into_si(value, symbol, expected):
    assert to_canonical(value, symbol) == pytest.approx(expected, rel=1e-12)


def test_a_vendor_may_declare_any_unit_of_the_right_dimension():
    """23600 fJ and 23.6 pJ are the same constant — string equality would refuse one."""
    assert to_canonical(23600.0, "fJ") == pytest.approx(to_canonical(23.6, "pJ"))


def test_unknown_unit_raises_naming_the_known_ones():
    with pytest.raises(ValueError, match=r"unknown unit 'joules'"):
        unit_for("joules")


def test_units_in_lists_only_that_dimension():
    energy = {unit.symbol for unit in units_in(ENERGY)}
    assert {"J", "mJ", "pJ"} <= energy
    assert "s" not in energy


def test_dimensionless_units_are_labels_and_never_rescale():
    """'bit', 'levels' and '1' are widths and counts, not payloads: scale is exactly 1."""
    for symbol in ("1", "bit", "levels", "columns"):
        unit = unit_for(symbol)
        assert unit.dimension == DIMENSIONLESS
        assert unit.si_scale == 1.0


def test_bytes_are_data_and_bits_are_not():
    """A payload byte count converts; a bit WIDTH is a plain number — separate dimensions."""
    assert unit_for("B").dimension == DATA
    assert unit_for("bit").dimension == DIMENSIONLESS
