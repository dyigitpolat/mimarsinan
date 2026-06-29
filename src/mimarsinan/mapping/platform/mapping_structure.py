"""Shared structural decision helpers for mapping backends.

Both ``LayoutIRMapping`` (shape-only) and ``IRMapping`` (full IR) call these
pure functions so that tiling mode and bias-axon counting are computed
identically.  No mapping state is accessed or mutated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


# Bias axon counting

def compute_core_input_count(
    n_sources: int,
    has_bias: bool,
    hardware_bias: bool,
) -> int:
    """Return the effective axon count for a core.

    When ``hardware_bias`` is True the bias lives in a dedicated register and
    does **not** consume an axon slot.  In legacy mode (``hardware_bias=False``)
    the bias occupies one extra always-on axon row.
    """
    if has_bias and not hardware_bias:
        return n_sources + 1
    return n_sources


# FC tiling mode

TilingMode = Literal["single", "coalescing", "output_tiled"]


class WideFanInUnsupportedError(ValueError):
    """A layer's fan-in exceeds one core and the chip cannot coalesce.

    A wide fan-in can only be mapped by *coalescing* — fusing N hard cores into
    one wider crossbar, which the chip realises by transferring partial-sum
    membrane potentials between cores. When ``allow_coalescing`` is False the
    chip lacks that capability, so the layer is unmappable (the lossy spike-domain
    partial-sum fallback was removed). Raise rather than silently emit a mapping
    the hardware cannot run.
    """


def compute_fc_tiling_mode(
    in_features: int,
    out_features: int,
    max_axons: int | None,
    max_neurons: int | None,
    has_bias: bool,
    hardware_bias: bool,
    allow_coalescing: bool,
) -> TilingMode:
    """Decide which FC mapping path to use.

    The wide-layer test uses the effective axon count (including legacy bias
    axon when ``hardware_bias`` is False) so that both backends agree on when
    a layer exceeds ``max_axons``. A wide fan-in maps via ``coalescing``: it emits
    one full-width core that the packer fuses from N hard cores of the same type
    (one wider crossbar), so the weighted sum is computed once and the deployment
    is bit-exact. Coalescing is a chip capability (inter-core membrane transfer);
    when ``allow_coalescing`` is False a wide fan-in is unmappable and raises
    :class:`WideFanInUnsupportedError`.
    """
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


# Capabilities (declared) + strategy (derived)


@dataclass(frozen=True)
class ChipCapabilities:
    """Declared chip permissions and core grid — the *capability* layer.

    Capabilities are what the chip allows: the core grid (``max_axons`` /
    ``max_neurons`` / ``hardware_bias``) plus the independent permission bits
    (``allow_coalescing`` = inter-core membrane partial-sum transfer,
    ``allow_neuron_splitting`` = a wide channel may span cores,
    ``allow_scheduling`` = passes may run on fresh core pools,
    ``allow_per_layer_s`` = each cascade depth / latency group may run at its own
    temporal resolution ``S_d`` rather than one global ``simulation_steps``). They
    are *declared* once; the per-layer mapping *strategy* is derived from them
    (see :class:`MappingStrategy`).

    ``allow_per_layer_s`` is the EW1 RESERVED gate: it is declared here alongside
    ``allow_coalescing`` but no mapping decision consults it yet (the per-depth S map
    derivation is deferred to research). Default False ⇒ the
    uniform global S is the only path ⇒ byte-identical.

    ``allow_weight_reuse`` is the RESERVED time-domain weight-reuse gate: when set, a
    maximal run of scheduled passes over one resident weight bank is a CHEAP reuse
    phase (positions time-multiplexed through fixed-mapping cores) instead of N
    reprograms. Declared here alongside ``allow_per_layer_s``; no mapping/sim/build
    decision consults it yet (round-1 only the cost model classifies reuse vs
    reprogram phases). Default False ⇒ every pass is a reprogram ⇒ byte-identical.
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
        """Read the permission bits from a platform-constraints / wizard body dict.

        Single source for ``bool(d.get("allow_coalescing", False))`` & friends, so
        the entry points that assemble these flags from config all read them
        identically (the grid is carried separately as ``cores`` / ``core_types``).
        """
        return cls(
            allow_coalescing=bool(constraints.get("allow_coalescing", False)),
            allow_neuron_splitting=bool(constraints.get("allow_neuron_splitting", False)),
            allow_scheduling=bool(constraints.get("allow_scheduling", False)),
            allow_per_layer_s=bool(constraints.get("allow_per_layer_s", False)),
            allow_weight_reuse=bool(constraints.get("allow_weight_reuse", False)),
        )

    def permission_kwargs(self) -> dict[str, bool]:
        """The three permission bits as the kwargs the layout/verify helpers expect.

        Lets an entry point spread one resolved capability object into the leaf
        helper signatures (``verify_hardware_config`` / ``compute_mapping_stats`` /
        ``build_layout_plan`` keep their bool params) instead of re-reading config.
        """
        return {
            "allow_neuron_splitting": self.allow_neuron_splitting,
            "allow_coalescing": self.allow_coalescing,
            "allow_scheduling": self.allow_scheduling,
        }


@dataclass(frozen=True)
class MappingStrategy:
    """Resolver that *derives* the per-layer mapping decision from capabilities.

    Given a layer's shape and the chip's :class:`ChipCapabilities`, this
    resolver decides the FC tiling mode (coalesce / split / sync-point) — the
    single place that turns capability permissions into a concrete mapping
    decision. The builder consults the resolved strategy's permission
    accessors (``allow_*``) and :meth:`tiling_mode`, never the raw flags.
    """

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
        """Resolve a strategy from the three raw permission bits.

        SSOT for the (now removed) ``build_hybrid_hard_core_mapping`` back-compat
        kwargs: callers that only have the loose ``allow_*`` bools wrap them in
        one capability object + strategy instead of threading three flags.
        """
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
        """The resolved MAPPING permission bits as layout/verify helper kwargs.

        ``allow_per_layer_s`` is intentionally NOT spread here: it is a temporal
        (per-depth S) capability, not a layout/verify input, so the leaf helper
        signatures stay unchanged (byte-identical).
        """
        return self.capabilities.permission_kwargs()

    def tiling_mode(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool,
    ) -> TilingMode:
        """Derive the FC tiling mode for one layer from shape × capabilities.

        Raises :class:`WideFanInUnsupportedError` when a wide fan-in cannot be
        mapped because coalescing is not permitted.
        """
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
