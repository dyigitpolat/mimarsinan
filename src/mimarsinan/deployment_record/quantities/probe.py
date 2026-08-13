"""The all-keys quantity probe: 'could this axis ever answer', never a real census."""

from __future__ import annotations

from mimarsinan.deployment_record.quantities.spec import (
    QUANTITY_SPECS,
    Quantities,
    QuantityValue,
)

#: Strictly positive so no availability predicate can be tripped by a zero, and
#: obviously synthetic so a probe value can never be mistaken for a census.
_PROBE_VALUE = 1.0


def probe_quantities() -> Quantities:
    """Every catalog key at a positive placeholder — capability questions only."""
    return Quantities({
        key: QuantityValue(value=_PROBE_VALUE, provenance="static")
        for key in QUANTITY_SPECS
    })
