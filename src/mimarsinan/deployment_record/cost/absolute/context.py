"""The pricing context: what every absolute metric shares, passed explicitly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute.formulas import (
    div_band,
    quantity_product,
    scale_band,
)
from mimarsinan.deployment_record.cost.terms import CostTerm
from mimarsinan.deployment_record.platform_physics.constants import (
    resolve_supersessions,
)
from mimarsinan.deployment_record.platform_physics.conversion import (
    conversion_model_for,
)
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.spec import Quantities
from mimarsinan.deployment_record.schema.provenance import Band

ComponentTable = Sequence[Tuple[str, Tuple[str, ...], str]]


@dataclass(frozen=True)
class PricingRefusal:
    """A headline that cannot be answered, and exactly why."""

    name: str
    reason: str


@dataclass(frozen=True)
class AbsolutePricing:
    """What the physics could price, and what it refused by name."""

    terms: Tuple[CostTerm, ...]
    refusals: Tuple[PricingRefusal, ...]


@dataclass(frozen=True)
class HostTime:
    """The host wall on the DEPLOYMENT host, or why it cannot be priced.

    Three distinguishable states, because they mean different things: a priced
    ``band``; ``blocked`` (the host share itself is unknown); ``missing`` constants
    (host work exists and the target never priced it). All-empty means there is
    genuinely no host work.
    """

    band: Optional[Band] = None
    blocked: Optional[str] = None
    missing: Tuple[str, ...] = ()

    @property
    def unpriceable(self) -> bool:
        return self.blocked is not None or bool(self.missing)


@dataclass
class PricingContext:
    """One target's physics against one view's quantities, accumulating terms."""

    quantities: Quantities
    physics: PlatformPhysics
    priceable: frozenset
    terms: List[CostTerm] = field(default_factory=list)
    refusals: List[PricingRefusal] = field(default_factory=list)

    @classmethod
    def create(
        cls, quantities: Quantities, physics: PlatformPhysics
    ) -> "PricingContext":
        """Supersession is resolved ONCE here: an aggregate absorbs its components.

        The target's declared DATAFLOW also runs once here, deriving the conversion
        quantities no record counts. They join the view's own quantities rather than
        replacing any: a model may only ADD what the record could not know.
        """
        model = conversion_model_for(physics.conversion_model)
        derived = model.derive(quantities)
        if derived:
            merged = {key: quantities.get(key) for key in quantities.keys()}
            for key, value in derived.items():
                merged.setdefault(key, value)
            quantities = Quantities(merged)
        return cls(
            quantities=quantities,
            physics=physics,
            priceable=frozenset(resolve_supersessions(physics.constants)),
        )

    def term(self, name: str, unit: str, band: Band, source: str) -> CostTerm:
        term = CostTerm(name=name, unit=unit, value=band.nominal, band=band,
                        kind="modeled", source=source)
        self.terms.append(term)
        return term

    def refuse(self, name: str, reason: str) -> None:
        self.refusals.append(PricingRefusal(name=name, reason=reason))

    def refusal_reason(self, name: str) -> Optional[str]:
        """Why ``name`` was refused, so a dependent term can name the ROOT cause."""
        for refusal in self.refusals:
            if refusal.name == name:
                return refusal.reason
        return None

    def priced(self, constant: str) -> bool:
        """Declared AND not absorbed by an aggregate."""
        return constant in self.priceable

    def component_bands(
        self, table: ComponentTable, scale: float
    ) -> Tuple[List[Band], List[str], List[str]]:
        """(priced bands, evidence notes, unpriced-work notes) over a component table."""
        bands: List[Band] = []
        evidence: List[str] = []
        unpriced: List[str] = []
        for constant, factors, label in table:
            product = quantity_product(self.quantities, factors)
            if self.priced(constant):
                if product is not None:
                    value = self.physics.band(constant)
                    bands.append(scale_band(
                        value, product * scale,
                        f"{constant} x {' x '.join(factors)}"))
                    evidence.append(f"{label}: {constant} [{value.basis}]")
            elif not self.physics.has(constant) and product:
                # Work exists (a positive census) but the target never priced it.
                unpriced.append(f"{constant}({label})")
        return bands, evidence, unpriced

    def host_time(self) -> HostTime:
        """The host wall, priced from a measured wall or the declared host rate."""
        if not self.quantities.has("host_macs"):
            return HostTime(blocked=(
                "the host share is unknown (no partition census), so a "
                "host-inclusive number cannot be certified"))
        if self.quantities.get("host_macs").value == 0:
            return HostTime()
        if self.quantities.has("host_ops_s"):
            if not self.physics.has("host_compute_rate"):
                return HostTime(missing=("host_compute_rate",))
            wall = self.quantities.get("host_ops_s").value
            return HostTime(band=div_band(
                Band(wall, wall, wall, "measured host wall"),
                self.physics.band("host_compute_rate"),
                "host_ops_s / host_compute_rate",
            ))
        if self.physics.has("host_macs_per_s"):
            macs = self.quantities.get("host_macs").value
            return HostTime(band=div_band(
                Band(macs, macs, macs, "host MAC census"),
                self.physics.band("host_macs_per_s"),
                "host_macs / host_macs_per_s (declared host rate)",
            ))
        return HostTime(missing=("host_ops_s (measured)", "host_macs_per_s"))

    def host_refusal(self, host: HostTime) -> str:
        """Why a host-inclusive headline cannot be certified."""
        if host.blocked is not None:
            return host.blocked
        return (
            "host work exists (host_macs > 0) and pricing it needs "
            f"{', '.join(host.missing)} — without a host price, moving work "
            "host-side would look free (the subsume rig)"
        )
