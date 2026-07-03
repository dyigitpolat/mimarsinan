"""Shared, stateless structural decision helpers for mapping backends (tiling mode, bias-axon counting)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


def compute_core_input_count(
    n_sources: int,
    has_bias: bool,
    hardware_bias: bool,
) -> int:
    """Return the effective axon count for a core; hardware_bias keeps the bias in a register (no axon), legacy mode adds one always-on axon row."""
    if has_bias and not hardware_bias:
        return n_sources + 1
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
) -> TilingMode:
    """Decide the FC mapping path (single/coalescing/output_tiled); wide fan-in maps via coalescing, or raises WideFanInUnsupportedError when allow_coalescing is False."""
    effective_in = compute_core_input_count(in_features, has_bias, hardware_bias)
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

    allow_per_layer_s and allow_weight_reuse are RESERVED gates (default False ⇒ byte-identical; no mapping decision consults them yet).
    """

    max_axons: int | None = None
    max_neurons: int | None = None
    hardware_bias: bool = False
    allow_coalescing: bool = False
    allow_neuron_splitting: bool = False
    allow_scheduling: bool = False
    allow_per_layer_s: bool = False
    allow_weight_reuse: bool = False

    @classmethod
    def from_platform_constraints(
        cls, constraints: Mapping[str, Any]
    ) -> "ChipCapabilities":
        """Read the permission bits from a platform-constraints / wizard body dict (single source for these flags)."""
        return cls(
            allow_coalescing=bool(constraints.get("allow_coalescing", False)),
            allow_neuron_splitting=bool(constraints.get("allow_neuron_splitting", False)),
            allow_scheduling=bool(constraints.get("allow_scheduling", False)),
            allow_per_layer_s=bool(constraints.get("allow_per_layer_s", False)),
            allow_weight_reuse=bool(constraints.get("allow_weight_reuse", False)),
        )

    def permission_kwargs(self) -> dict[str, bool]:
        """The permission bits as the kwargs the layout/verify leaf helpers expect."""
        return {
            "allow_neuron_splitting": self.allow_neuron_splitting,
            "allow_coalescing": self.allow_coalescing,
            "allow_scheduling": self.allow_scheduling,
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
    def allow_weight_reuse(self) -> bool:
        """The RESERVED weight-reuse gate (no mapping/build decision consults it yet)."""
        return self.capabilities.allow_weight_reuse

    def permission_kwargs(self) -> dict[str, bool]:
        """The resolved mapping permission bits as layout/verify kwargs (allow_per_layer_s is a temporal gate, intentionally not spread here)."""
        return self.capabilities.permission_kwargs()

    def tiling_mode(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool,
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
        )
