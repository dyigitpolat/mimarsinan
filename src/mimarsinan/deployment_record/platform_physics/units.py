"""Dimensional units for physics constants — the one conversion to canonical SI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

ENERGY = "energy"
AREA = "area"
TIME = "time"
POWER = "power"
DATA = "data"
DIMENSIONLESS = "dimensionless"

DIMENSIONS: Tuple[str, ...] = (ENERGY, AREA, TIME, POWER, DATA, DIMENSIONLESS)


@dataclass(frozen=True)
class Unit:
    """A declarable unit: its dimension, and the factor onto that dimension's SI base."""

    symbol: str
    dimension: str
    si_scale: float


#: A vendor may declare a constant in any unit of the right DIMENSION — 23600 fJ and
#: 23.6 pJ are the same number, and only the dimension is a real constraint.
_UNITS: Tuple[Unit, ...] = (
    Unit("J", ENERGY, 1.0),
    Unit("mJ", ENERGY, 1e-3),
    Unit("uJ", ENERGY, 1e-6),
    Unit("nJ", ENERGY, 1e-9),
    Unit("pJ", ENERGY, 1e-12),
    Unit("fJ", ENERGY, 1e-15),
    Unit("m^2", AREA, 1.0),
    Unit("cm^2", AREA, 1e-4),
    Unit("mm^2", AREA, 1e-6),
    Unit("um^2", AREA, 1e-12),
    Unit("s", TIME, 1.0),
    Unit("ms", TIME, 1e-3),
    Unit("us", TIME, 1e-6),
    Unit("ns", TIME, 1e-9),
    Unit("ps", TIME, 1e-12),
    Unit("W", POWER, 1.0),
    Unit("mW", POWER, 1e-3),
    Unit("uW", POWER, 1e-6),
    Unit("nW", POWER, 1e-9),
    Unit("B", DATA, 1.0),
    Unit("KiB", DATA, 1024.0),
    Unit("MiB", DATA, 1024.0 * 1024.0),
    # Dimensionless symbols are display labels for pure numbers — widths, level counts,
    # sharing factors and ratios. They never rescale, which is what keeps a bit WIDTH
    # (a number) from ever being mistaken for a DATA payload (which converts).
    Unit("1", DIMENSIONLESS, 1.0),
    Unit("bit", DIMENSIONLESS, 1.0),
    Unit("levels", DIMENSIONLESS, 1.0),
    Unit("columns", DIMENSIONLESS, 1.0),
    Unit("fraction", DIMENSIONLESS, 1.0),
)

_BY_SYMBOL: Dict[str, Unit] = {unit.symbol: unit for unit in _UNITS}
_CANONICAL: Dict[str, str] = {
    ENERGY: "J",
    AREA: "m^2",
    TIME: "s",
    POWER: "W",
    DATA: "B",
    DIMENSIONLESS: "1",
}


def unit_for(symbol: str) -> Unit:
    """The declared unit, or a loud error naming every unit this layer accepts."""
    try:
        return _BY_SYMBOL[symbol]
    except KeyError:
        raise ValueError(
            f"unknown unit {symbol!r}; physics constants accept "
            f"{sorted(_BY_SYMBOL)}"
        ) from None


def canonical_symbol(dimension: str) -> str:
    """The SI base symbol every constant of ``dimension`` is stored in."""
    try:
        return _CANONICAL[dimension]
    except KeyError:
        raise ValueError(
            f"unknown dimension {dimension!r}; expected one of {sorted(_CANONICAL)}"
        ) from None


def units_in(dimension: str) -> Tuple[Unit, ...]:
    """Every unit a vendor may use for a constant of this dimension."""
    canonical_symbol(dimension)
    return tuple(unit for unit in _UNITS if unit.dimension == dimension)


def to_canonical(value: float, symbol: str) -> float:
    """``value`` expressed in its dimension's SI base unit."""
    return float(value) * unit_for(symbol).si_scale
