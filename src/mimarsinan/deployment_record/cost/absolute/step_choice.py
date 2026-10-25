"""E85's priced plan: the residency step whose period census prices the least write and carry energy per inference.

The planner states the step function and prices nothing (``mapping/support/schedule/residency_units.py``); the
pricer chooses among the steps' period censuses at the declared batch K, as ``x9_0_v3.price`` takes its argmin
over the plan's steps. Ties go to the first step (the smaller r): two steps whose exact prices are equal price
within a few units of the last place in floating point (each term is rounded once), so prices within ``TIE`` of
each other, relatively, count as equal. The compute term is the same for every step of one program, so it never
enters the choice. A profile whose external component is undeclared (an SRAM-CIM macro without its backing store)
prices no step and refuses by name, as ``price_absolute`` and the write law do (``context.external_refusals``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute.context import PricingContext
from mimarsinan.deployment_record.cost.absolute.formulas import ENERGY_COMPONENTS, add_bands
from mimarsinan.deployment_record.cost.absolute.context import external_refusals
from mimarsinan.deployment_record.cost.absolute.writes import price_write_terms
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.schema.provenance import Band

#: The carry components of the energy table, one per direction (``carry_inject``, ``carry_readout``).
CARRY_COMPONENTS = tuple(row for row in ENERGY_COMPONENTS if row[2] in ("carry_inject", "carry_readout"))
_J_TO_MJ = 1e3
CORNERS = ("low", "nominal", "high")
#: Relative price difference below which two steps tie (a few ulps of a double).
TIE = 2.0 ** -48


class StepChoiceError(ToolchainRefusal, ValueError):
    """A step price the target cannot state (an undeclared external component, no commit energy, unpriced carried
    bytes) or no step to choose."""


@dataclass(frozen=True)
class StepPrice:
    """One step's period census priced: write energy and carry energy per inference, in mJ."""

    write: Band
    carry: Band

    def at(self, corner: str) -> float:
        return float(getattr(self.write, corner)) + float(getattr(self.carry, corner))

    @property
    def nominal(self) -> float:
        return self.at("nominal")


def carry_energy(census: Any, physics: PlatformPhysics) -> Band:
    """The carry components priced on the census's carried bytes per direction; unpriced bytes refuse by name."""
    quantities = Quantities({"carry_in_bytes": QuantityValue(float(census.carry_in_bytes), "static"),
                             "carry_out_bytes": QuantityValue(float(census.carry_out_bytes), "static")})
    bands, _evidence, unpriced = PricingContext.create(quantities, physics).component_bands(CARRY_COMPONENTS, _J_TO_MJ)
    if unpriced:
        raise StepChoiceError(f"the step carries bytes the target does not price: {', '.join(unpriced)}")
    return add_bands(bands, " + ".join(b.basis for b in bands)) if bands else Band(0.0, 0.0, 0.0, "no carry")


def price_step(census: Any, physics: PlatformPhysics, policy: Any) -> StepPrice:
    """``price_write_terms``'s write energy plus the carry components of one step's period census."""
    if refusals := external_refusals(physics):
        raise StepChoiceError("; ".join(f"{r.name}: {r.reason}" for r in refusals))
    writes = price_write_terms(census, physics, policy)
    if writes.energy is None:
        reason = next((r.reason for r in writes.refusals if r.name == "energy_write_mj"), "write energy refused")
        raise StepChoiceError(f"the step's write energy is not priced: {reason}")
    return StepPrice(write=writes.energy, carry=carry_energy(census, physics))


def cheapest_step(censuses: Sequence[Any], physics: PlatformPhysics, policy: Any, *, corner: str = "nominal") -> int:
    """Index of the step whose census prices the least writes plus carry at ``corner`` (ties to the first)."""
    if corner not in CORNERS:
        raise StepChoiceError(f"corner {corner!r} is not one of {CORNERS}")
    best_index, best_cost = None, None
    for index, census in enumerate(censuses):
        cost = price_step(census, physics, policy).at(corner)
        if best_cost is None or cost < best_cost - TIE * abs(best_cost):
            best_index, best_cost = index, cost
    if best_index is None:
        raise StepChoiceError("no residency step to choose from")
    return best_index
