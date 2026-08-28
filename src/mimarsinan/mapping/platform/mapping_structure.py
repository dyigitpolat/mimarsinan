"""Shared, stateless structural decision helpers for mapping backends (tiling mode, bias-axon counting)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping

from mimarsinan.mapping.platform.platform_constraints import (
    resolve_platform_mapping_params,
)


def compute_core_input_count(
    n_sources: int,
    has_bias: bool,
    hardware_bias: bool,
    bias_rows: int = 1,
) -> int:
    """Return the effective axon count for a core; hardware_bias keeps the bias in a register (no axon), otherwise the bias spends ``bias_rows`` always-on axon rows (1 = the unsplit legacy row)."""
    if has_bias and not hardware_bias:
        return n_sources + int(bias_rows)
    return n_sources


TilingMode = Literal["single", "coalescing", "output_tiled"]


class WideFanInUnsupportedError(ValueError):
    """A layer's fan-in exceeds one core and the chip cannot coalesce; raised instead of emitting an unrunnable mapping."""


def compute_fc_tiling_mode(
    in_features: int,
    out_features: int,
    max_axons: int | None,
    max_neurons: int | None,
    has_bias: bool,
    hardware_bias: bool,
    allow_coalescing: bool,
    bias_rows: int = 1,
) -> TilingMode:
    """Decide the FC mapping path (single/coalescing/output_tiled); wide fan-in maps via coalescing, or raises WideFanInUnsupportedError when allow_coalescing is False."""
    effective_in = compute_core_input_count(
        in_features, has_bias, hardware_bias, bias_rows
    )
    is_wide = max_axons is not None and effective_in > max_axons
    if is_wide:
        if not allow_coalescing:
            raise WideFanInUnsupportedError(
                f"Layer fan-in {effective_in} exceeds max_axons {max_axons} but "
                f"allow_coalescing=False: this chip cannot map a wide fan-in (no "
                f"inter-core membrane partial-sum transfer). Enable coalescing, raise "
                f"max_axons, or reduce the layer's fan-in."
            )
        return "coalescing"
    if max_neurons is not None and out_features > max_neurons:
        return "output_tiled"
    return "single"


@dataclass(frozen=True)
class ChipCapabilities:
    """Declared chip permissions and core grid — the capability layer MappingStrategy derives per-layer decisions from.

    allow_per_layer_s is a RESERVED gate (default False ⇒ byte-identical; no mapping decision consults it yet).
    """

    max_axons: int | None = None
    max_neurons: int | None = None
    hardware_bias: bool = False
    allow_coalescing: bool = False
    allow_neuron_splitting: bool = False
    allow_scheduling: bool = False
    allow_per_layer_s: bool = False
    # The scheduled-build pass budget the planner's residency floor is sized
    # against; dropping it pins the answer to the default and disarms streaming.
    max_schedule_passes: int = 8

    @classmethod
    def from_platform_constraints(
        cls, constraints: Mapping[str, Any]
    ) -> "ChipCapabilities":
        """Read the WHOLE declaration from a platform-constraints / wizard body dict.

        The permission/scheduling flags come off the dict directly; the geometry
        (``max_axons``/``max_neurons``/``hardware_bias``) is resolved from the
        declared core grid — ``cores`` (a resolved platform) or ``core_types`` (a
        wizard body) — by :func:`resolve_platform_mapping_params`, the SAME SSOT
        the layout mapper is configured from. They are therefore the EFFECTIVE
        per-core limits the tiling decision consumes: the grid maximum, minus the
        bias axon when the cores carry no hardware bias. A dict declaring no core
        grid leaves them at their "not declared" defaults (``None``/``False``)
        rather than inventing a chip.
        """
        cores = constraints.get("cores") or constraints.get("core_types") or ()
        allow_coalescing = bool(constraints.get("allow_coalescing", False))
        geometry = (
            resolve_platform_mapping_params(
                list(cores), allow_coalescing=allow_coalescing
            )
            if cores
            else None
        )
        return cls(
            max_axons=None if geometry is None else geometry.effective_max_axons,
            max_neurons=None if geometry is None else geometry.effective_max_neurons,
            hardware_bias=False if geometry is None else geometry.hardware_bias,
            allow_coalescing=allow_coalescing,
            allow_neuron_splitting=bool(constraints.get("allow_neuron_splitting", False)),
            allow_scheduling=bool(constraints.get("allow_scheduling", False)),
            allow_per_layer_s=bool(constraints.get("allow_per_layer_s", False)),
            max_schedule_passes=int(constraints.get("max_schedule_passes", 8) or 8),
        )

    def capability_bits(self) -> dict[str, Any]:
        """The COMPLETE declaration, one entry per declared field.

        Derived from the dataclass rather than listed, so a capability added
        here reaches every consumer that serves this dict instead of quietly
        going unforwarded (which is how the retired ``schedule_policy`` enum
        once stayed invisible to search). Every bit must therefore be
        POPULATED by whoever builds the object — a served ``None`` means
        "this platform declares no core grid", never "we did not look".
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def permission_kwargs(self) -> dict[str, bool]:
        """The three permission bits the leaf packing helpers take.

        Kept for helpers whose signature is exactly those bits; the layout
        answer takes :meth:`layout_kwargs`, which also carries the scheduler.
        """
        return {
            "allow_neuron_splitting": self.allow_neuron_splitting,
            "allow_coalescing": self.allow_coalescing,
            "allow_scheduling": self.allow_scheduling,
        }

    def layout_kwargs(self) -> dict[str, Any]:
        """Everything the layout/verify entry points need to answer as deployment would.

        The permission bits PLUS the pass budget — the pass structure is not a
        property of the permission bits alone, and a layout answer computed
        without the budget describes a program the chip will never run.
        """
        return {
            **self.permission_kwargs(),
            "max_schedule_passes": self.max_schedule_passes,
        }


@dataclass(frozen=True)
class MappingStrategy:
    """Derives the per-layer mapping decision (FC tiling mode) from a ChipCapabilities — the single place turning permissions into a concrete decision."""

    capabilities: ChipCapabilities

    @classmethod
    def resolve(cls, capabilities: ChipCapabilities) -> "MappingStrategy":
        """Resolve a strategy for the given declared capabilities."""
        return cls(capabilities=capabilities)

    @classmethod
    def from_permissions(
        cls,
        *,
        allow_coalescing: bool = False,
        allow_neuron_splitting: bool = False,
        allow_scheduling: bool = False,
    ) -> "MappingStrategy":
        """Resolve a strategy from the three raw permission bits."""
        return cls.resolve(
            ChipCapabilities(
                allow_coalescing=bool(allow_coalescing),
                allow_neuron_splitting=bool(allow_neuron_splitting),
                allow_scheduling=bool(allow_scheduling),
            )
        )

    @property
    def allow_coalescing(self) -> bool:
        return self.capabilities.allow_coalescing

    @property
    def allow_neuron_splitting(self) -> bool:
        return self.capabilities.allow_neuron_splitting

    @property
    def allow_scheduling(self) -> bool:
        return self.capabilities.allow_scheduling

    @property
    def allow_per_layer_s(self) -> bool:
        """The EW1 RESERVED per-layer-S gate (no mapping decision consults it yet)."""
        return self.capabilities.allow_per_layer_s

    @property
    def max_schedule_passes(self) -> int:
        """The declared scheduled-build pass budget."""
        return self.capabilities.max_schedule_passes

    def permission_kwargs(self) -> dict[str, bool]:
        """The resolved mapping permission bits as layout/verify kwargs (allow_per_layer_s is a temporal gate, intentionally not spread here)."""
        return self.capabilities.permission_kwargs()

    def layout_kwargs(self) -> dict[str, Any]:
        """The permission bits PLUS the scheduling declaration (see ChipCapabilities)."""
        return self.capabilities.layout_kwargs()

    def tiling_mode(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool,
        bias_rows: int = 1,
    ) -> TilingMode:
        """Derive the FC tiling mode for one layer from shape × capabilities; raises WideFanInUnsupportedError when a wide fan-in cannot be mapped."""
        caps = self.capabilities
        return compute_fc_tiling_mode(
            in_features,
            out_features,
            caps.max_axons,
            caps.max_neurons,
            has_bias,
            caps.hardware_bias,
            caps.allow_coalescing,
            bias_rows,
        )
