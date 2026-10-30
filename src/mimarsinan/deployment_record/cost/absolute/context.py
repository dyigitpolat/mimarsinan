"""The pricing context: what every absolute metric shares, passed explicitly."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.absolute.formulas import (
    Factor,
    div_band,
    factor_withheld,
    quantity_product,
    add_bands,
    scale_band,
)
from mimarsinan.deployment_record.cost.host_terms import host_memory_time
from mimarsinan.deployment_record.cost.terms import CostTerm
from mimarsinan.deployment_record.platform_physics.constants import (
    resolve_supersessions,
)
from mimarsinan.deployment_record.platform_physics.conversion import (
    conversion_model_for,
)
from mimarsinan.deployment_record.platform_physics.hosts.description import describes_host
from mimarsinan.deployment_record.platform_physics.profile import ExternalComponent, PlatformPhysics
from mimarsinan.deployment_record.quantities.spec import Quantities
from mimarsinan.deployment_record.schema.provenance import Band

ComponentTable = Sequence[Tuple[str, Tuple[Factor, ...], str]]


@dataclass(frozen=True)
class PricingRefusal:
    """A headline that cannot be answered, and exactly why."""

    name: str
    reason: str


#: The refusal of every absolute term at once: a target that declares no physics, or a profile whose external
#: component (``PlatformPhysics.external_components``) the run has not declared.
ABSOLUTE_PRICING = "absolute_pricing"


def _external_refusal(physics: PlatformPhysics, component: ExternalComponent) -> PricingRefusal:
    missing = ", ".join(physics.missing(component.constants))
    note = f" ({component.note})" if component.note else ""
    return PricingRefusal(name=ABSOLUTE_PRICING, reason=(
        f"{physics.name} is read only with its {component.name} declared: {missing} undeclared{note}; declare "
        "them in a named override block (merge_override_blocks)"))


def external_refusals(physics: PlatformPhysics) -> Tuple[PricingRefusal, ...]:
    """The ``absolute_pricing`` refusal of every external component the run has not declared (empty when none):
    the one gate of every pricer that reads the profile (``price_absolute``, the write law ``price_write_terms`` on
    every plane [fix_c], the step choice)."""
    return tuple(_external_refusal(physics, component) for component in physics.undeclared_external())


@dataclass(frozen=True)
class AdmissibilityVerdict:
    """Whether a priced program is admitted, by name; it never suppresses a priced term."""

    name: str
    #: ``admitted``, ``inadmissible`` or ``straddles`` (the band crosses the floor).
    verdict: str
    reason: str
    evidence: str


@dataclass(frozen=True)
class AbsolutePricing:
    """What the physics could price, what it refused by name, and what it judged inadmissible."""

    terms: Tuple[CostTerm, ...]
    refusals: Tuple[PricingRefusal, ...]
    verdicts: Tuple[AdmissibilityVerdict, ...] = ()


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
    #: Host work the target never priced but that does not block the headline (named in it).
    unpriced: Tuple[str, ...] = ()

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
    #: [E190] The priced chip's core types (``validity.CoreGeometry``), declared or proved by the
    #: census, or ``validity.CensusCells`` where the census proves only its cell count; the aggregate
    #: constants are judged against the profile's validity geometry.
    geometry: Optional[Any] = None
    #: [E-5] Constants the run's declarations refuse with their reason (a converter constant its table does not state
    #: at the run's adc_bits): a component whose work exists blocks its headline by that reason (``blocked``).
    refused_constants: Mapping[str, str] = field(default_factory=dict)
    blocked: Dict[str, str] = field(default_factory=dict)
    #: [U-J, U-J3] The write law's bytes per inference toward the chip (``host_terms.WriteCrossing``: the store's and
    #: the program's own), from ``host_terms.write_link_bytes``.
    write_link_bytes: Optional[Any] = None
    #: [U-J2] The latency's parts the host does not execute (the chip, the link, exposed writes), summed directly:
    #: the wait a host's idle power is drawn over (``metrics.price_latency``; never e2e less host time, corner by corner).
    host_wait: Optional[Band] = None
    #: [U-J2 targeted M1] The headline energy's parts but the host's (the chip and its writes), summed where the
    #: headline prices: the whole-system line adds the whole host to it (``host_terms.price_system_energy``).
    energy_chip_side: Optional[Band] = None

    @classmethod
    def create(
        cls, quantities: Quantities, physics: PlatformPhysics,
        geometry: Optional[Any] = None, refused_constants: Optional[Mapping[str, str]] = None,
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
            withheld = {k: r for k, r in quantities.withheld.items() if k not in merged}
            quantities = Quantities(merged, withheld=withheld)
        return cls(
            quantities=quantities,
            physics=physics,
            priceable=frozenset(resolve_supersessions(physics.constants)),
            geometry=geometry,
            refused_constants=dict(refused_constants or {}),
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
            # A factor the view withholds: its work may exist, so the component is named, never dropped.
            withheld = next((reason for reason in (factor_withheld(self.quantities, f) for f in factors)
                             if reason is not None), None)
            if constant in self.refused_constants:
                if product or withheld is not None:  # work exists that the declared constant cannot price
                    self.blocked[constant] = self.refused_constants[constant]
                continue
            if self.priced(constant):
                if product is not None:
                    value = self.physics.band(constant)
                    bands.append(scale_band(
                        value, product * scale,
                        f"{constant} x {' x '.join(map(str, factors))}"))
                    evidence.append(f"{label}: {constant} [{value.basis}]")
                elif withheld is not None:
                    unpriced.append(f"{constant}({label}; {withheld})")
            elif not self.physics.has(constant) and (product or withheld is not None):
                # Work exists (a positive census) but the target never priced it.
                unpriced.append(f"{constant}({label})" if withheld is None else f"{constant}({label}; {withheld})")
        return bands, evidence, unpriced

    def host_time(self) -> HostTime:
        """The host wall, priced from a measured wall or the declared host rate; [U-J] plus the MAC-free host
        operators' bytes at the host memory time (``host_terms.host_memory_time``), where declared."""
        base = self._compute_host_time()
        memory, unpriced = host_memory_time(self)
        if base.unpriceable or (memory is None and not unpriced):
            return base
        band = base.band if memory is None else memory if base.band is None else add_bands(
            [base.band, memory], f"{base.band.basis} + {memory.basis}")
        return replace(base, band=band, unpriced=base.unpriced + tuple(unpriced))

    def _wall_is_this_host(self) -> bool:
        """[U-J] A measured host wall prices the host unless the physics describes a host of its own (a host
        description's whole model, ``describes_host``) without relating the machine that measured the wall to it
        (``host_compute_rate``): that host prices its own time, never another machine's wall. [targeted M1] A partial
        host block (a server's measured calibration) keeps the wall's rule, which needs ``host_compute_rate``."""
        return self.physics.has("host_compute_rate") or not describes_host(self.physics)

    def _compute_host_time(self) -> HostTime:
        if not self.quantities.has("host_macs"):
            return HostTime(blocked=(
                "the host share is unknown (no partition census), so a "
                "host-inclusive number cannot be certified"))
        if self.quantities.get("host_macs").value == 0:
            return self._dispatch_only_host_time()
        if self.quantities.has("host_ops_s") and self._wall_is_this_host():
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
            work = div_band(
                Band(macs, macs, macs, "host MAC census"),
                self.physics.band("host_macs_per_s"),
                "host_macs / host_macs_per_s (declared host rate)",
            )
            # [R2] The measured host wall is dispatch-bound at small op sizes
            # (~2000x past any rate model on the MLP study): every ComputeOp
            # invocation pays the deployment's own per-op overhead too.
            if (self.physics.has("t_host_op_overhead")
                    and self.quantities.has("compute_op_count")):
                ops = self.quantities.get("compute_op_count").value
                overhead = scale_band(
                    self.physics.band("t_host_op_overhead"), ops,
                    "t_host_op_overhead x compute_op_count",
                )
                return HostTime(band=add_bands(
                    [work, overhead],
                    "host_macs / host_macs_per_s + t_host_op_overhead x "
                    "compute_op_count",
                ))
            if self.quantities.has("compute_op_count"):
                return HostTime(band=Band(
                    work.low, work.nominal, work.high,
                    work.basis + "; unpriced: t_host_op_overhead("
                    "per-invocation dispatch)",
                ))
            return HostTime(band=work)
        return HostTime(missing=("host_ops_s (measured)", "host_macs_per_s"))

    def _dispatch_only_host_time(self) -> HostTime:
        """No host MACs: the host still dispatches every ComputeOp (ReLU, add, pool, softmax).

        A measured wall keeps today's empty host time (that path is the measured host repair's);
        otherwise dispatch is priced through the declared per-op overhead, or named unpriced.
        """
        if (self.quantities.has("host_ops_s") and self._wall_is_this_host()) or not self.quantities.has(
                "compute_op_count"):
            return HostTime()
        ops = self.quantities.get("compute_op_count").value
        if ops == 0:
            return HostTime()
        if self.physics.has("t_host_op_overhead"):
            return HostTime(band=scale_band(
                self.physics.band("t_host_op_overhead"), ops,
                "t_host_op_overhead x compute_op_count (host_macs = 0: dispatch only)",
            ))
        return HostTime(unpriced=("t_host_op_overhead(per-invocation dispatch)",))

    def host_refusal(self, host: HostTime) -> str:
        """Why a host-inclusive headline cannot be certified."""
        if host.blocked is not None:
            return host.blocked
        return (
            "host work exists (host_macs > 0) and pricing it needs "
            f"{', '.join(host.missing)} — without a host price, moving work "
            "host-side would look free (the subsume rig)"
        )
