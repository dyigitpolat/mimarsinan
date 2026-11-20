"""The pricing context: what every absolute metric shares, passed explicitly."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from mimarsinan.deployment_record.cost.absolute.formulas import (
    div_band,
    add_bands,
    scale_band,
)
from mimarsinan.deployment_record.cost.absolute.tables.components import ComponentTable, component_bands, with_derived
from mimarsinan.deployment_record.cost.absolute.walls.static_power import NO_STATIC, StaticSum
from mimarsinan.deployment_record.cost.host_terms import host_memory_time
from mimarsinan.deployment_record.cost.terms import CostTerm, PricingFlag
from mimarsinan.deployment_record.platform_physics.constants import (
    resolve_supersessions,
)
from mimarsinan.deployment_record.platform_physics.conversion import (
    conversion_model_for,
)
from mimarsinan.deployment_record.platform_physics.hosts.host_work import (
    HostWorkState, host_work_refusal, host_work_state,
)
from mimarsinan.deployment_record.platform_physics.profile import ExternalComponent, PlatformPhysics
from mimarsinan.deployment_record.quantities.spec import Quantities
from mimarsinan.deployment_record.schema.provenance import Band


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
    #: [U-AREA] Departures the priced terms name (``cost.terms.PricingFlag``), computed at read, never sealed.
    flags: Tuple[PricingFlag, ...] = ()
    #: [U-ENERGY class 1] The chip's one static sum this pricing resolved (``walls.static_power.StaticSum``); None
    #: where nothing was priced.
    static_power: Optional[StaticSum] = None


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
    #: [U-ENERGY class 1] The chip's one static sum on the resolved physics, set once by ``price_absolute``; every
    #: static leg of the pricing reads it.
    static: StaticSum = NO_STATIC

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
        quantities = with_derived(quantities, conversion_model_for(physics.conversion_model))  # [U-ENERGY class 2]
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
        """(priced bands, evidence notes, unpriced-work notes) over a component table, every factor in one of four
        states (``tables.components``)."""
        return component_bands(self, table, scale)

    def host_time(self) -> HostTime:
        """The host time, priced from the declared host's rate and dispatch; [U-J] plus the MAC-free host
        operators' bytes at the host memory time (``host_terms.host_memory_time``), where declared. [U-HOST] Never a
        measured wall: a host row is admitted only from an identified host description (``hosts.admission``), and
        the development machine's wall is a diagnostic (``stage_timing.host_wall``)."""
        base = self._compute_host_time()
        memory, unpriced = host_memory_time(self)
        if base.unpriceable or (memory is None and not unpriced):
            return base
        band = base.band if memory is None else memory if base.band is None else add_bands(
            [base.band, memory], f"{base.band.basis} + {memory.basis}")
        return replace(base, band=band, unpriced=base.unpriced + tuple(unpriced))

    def _compute_host_time(self) -> HostTime:
        state = host_work_state(self.quantities, self.physics.has)  # [U-HOST2] one judgment, both cost planes
        if state.blocked is not None or state.missing:
            return HostTime(blocked=state.blocked, missing=state.missing)
        if self.quantities.get("host_macs").value == 0:
            return self._dispatch_only_host_time()
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
        return HostTime(missing=("host_macs_per_s",))

    def _dispatch_only_host_time(self) -> HostTime:
        """No host MACs: the host still dispatches every ComputeOp (ReLU, add, pool, softmax): priced through the
        declared per-op overhead, or named unpriced ([U-HOST] a measured wall prices nothing here either)."""
        if not self.quantities.has("compute_op_count"):
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
        """Why a host-inclusive headline cannot be certified ([U-HOST2] the measured plane prints the same words)."""
        return host_work_refusal(HostWorkState(blocked=host.blocked, missing=host.missing))
