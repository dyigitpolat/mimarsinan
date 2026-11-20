"""The quantity catalog: every priceable multiplicand, declared once with its meaning.

A quantity is a number a deployment produces (a count, a byte total, a wall time) that
a physics constant multiplies. The catalog is closed so a pricing formula can never
reference a number nobody produces, and ABSENCE is meaningful throughout: a quantity a
view cannot answer is missing, never zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple

from mimarsinan.deployment_record.quantities.catalog.rows import ROWS as _ROWS
from mimarsinan.deployment_record.units import DIMENSIONS, unit_for

#: measured = read off an executed deployment; static = a fact of the declared shape
#: or packing; modeled = rests on a stated assumption (and says so downstream).
PROVENANCE_KINDS = frozenset({"measured", "static", "modeled"})


@dataclass(frozen=True)
class QuantitySpec:
    """One catalog row: what the number is, in what unit, and what it means."""

    key: str
    dimension: str
    unit: str
    doc: str

    def __post_init__(self) -> None:
        if self.dimension not in DIMENSIONS:
            raise ValueError(f"{self.key}: unknown dimension {self.dimension!r}")
        if unit_for(self.unit).dimension != self.dimension:
            raise ValueError(
                f"{self.key}: unit {self.unit!r} is not a {self.dimension} unit"
            )


def _build() -> Dict[str, QuantitySpec]:
    specs: Dict[str, QuantitySpec] = {}
    for key, (dimension, unit), doc in _ROWS:
        if key in specs:
            raise ValueError(f"quantity {key!r} is declared twice")
        specs[key] = QuantitySpec(key=key, dimension=dimension, unit=unit, doc=doc)
    return specs


QUANTITY_SPECS: Mapping[str, QuantitySpec] = _build()


def quantity_spec(key: str) -> QuantitySpec:
    """The declaration for ``key``, or a loud error naming the catalog."""
    try:
        return QUANTITY_SPECS[key]
    except KeyError:
        raise KeyError(
            f"{key!r} is not a catalog quantity; the catalog is {sorted(QUANTITY_SPECS)}"
        ) from None


@dataclass(frozen=True)
class QuantityValue:
    """One produced number and how it was obtained."""

    value: float
    provenance: str

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE_KINDS:
            raise ValueError(
                f"QuantityValue.provenance must be one of {sorted(PROVENANCE_KINDS)}, "
                f"got {self.provenance!r}"
            )


def _absent(given: Optional[Mapping[str, str]], values: Mapping[str, QuantityValue],
            taken: Iterable[str] = ()) -> Dict[str, str]:
    """One absent state's reasons in catalog order: never a produced key, never a key an earlier state holds."""
    given, taken = dict(given or {}), set(taken)
    for key in given:
        quantity_spec(key)
    return {key: str(given[key]) for key in QUANTITY_SPECS if key in given and key not in values and key not in taken}


class Quantities:
    """A view's produced quantities. Absence is meaningful — there is no default.

    A WITHHELD quantity is absent with a reason: the view knows work may exist but cannot size it
    (a value carry without the declared value grid), so every term it multiplies is named unpriced
    with that reason rather than silently dropped. [U-ENERGY class 2] Two more absent states, each with its
    reason: NOT PRODUCED (``unproduced``: the plane never counts it, so a declared constant over it is named beside
    the estimate) and ZERO BY DECLARATION (``known_zero``: the platform declares no component that does the work).
    """

    def __init__(self, values: Mapping[str, QuantityValue],
                 withheld: Optional[Mapping[str, str]] = None, unproduced: Optional[Mapping[str, str]] = None,
                 known_zero: Optional[Mapping[str, str]] = None) -> None:
        for key in values:
            quantity_spec(key)
        # Catalog order, so reports and comparisons are stable.
        self._values: Dict[str, QuantityValue] = {
            key: values[key] for key in QUANTITY_SPECS if key in values
        }
        withheld = dict(withheld or {})
        for key in withheld:
            quantity_spec(key)
            if key in self._values:
                raise ValueError(f"{key!r} is both produced and withheld")
        self._withheld: Dict[str, str] = {
            key: str(withheld[key]) for key in QUANTITY_SPECS if key in withheld
        }
        self._known_zero = _absent(known_zero, self._values, self._withheld)
        self._unproduced = _absent(unproduced, self._values, set(self._withheld) | set(self._known_zero))

    def has(self, key: str) -> bool:
        quantity_spec(key)
        return key in self._values

    def get(self, key: str) -> QuantityValue:
        quantity_spec(key)
        if key not in self._values:
            raise KeyError(
                f"this view produces no {key!r}; the terms that need it are "
                f"unavailable, and no default may stand in for a census"
            )
        return self._values[key]

    def missing(self, keys: Iterable[str]) -> Tuple[str, ...]:
        return tuple(key for key in keys if not self.has(key))

    def keys(self) -> Iterator[str]:
        return iter(self._values)

    def withheld_reason(self, key: str) -> Optional[str]:
        """Why ``key`` is absent although its work may exist, or None."""
        quantity_spec(key)
        return self._withheld.get(key)

    @property
    def withheld(self) -> Dict[str, str]:
        return dict(self._withheld)

    def unproduced_reason(self, key: str) -> Optional[str]:
        """[U-ENERGY class 2] Why this plane never counts ``key``, or None."""
        quantity_spec(key)
        return self._unproduced.get(key)

    @property
    def unproduced(self) -> Dict[str, str]:
        return dict(self._unproduced)

    def known_zero_reason(self, key: str) -> Optional[str]:
        """[U-ENERGY class 2] Why ``key``'s work is zero by the platform's declaration, or None."""
        quantity_spec(key)
        return self._known_zero.get(key)

    @property
    def known_zero(self) -> Dict[str, str]:
        return dict(self._known_zero)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Quantities) and self._values == other._values
                and self._withheld == other._withheld and self._unproduced == other._unproduced
                and self._known_zero == other._known_zero)

    def __repr__(self) -> str:
        return f"Quantities({sorted(self._values)})"
