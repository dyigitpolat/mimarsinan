"""Shared structural decision helpers for mapping backends.

Both ``LayoutIRMapping`` (shape-only) and ``IRMapping`` (full IR) call these
pure functions so that tiling mode and bias-axon counting are computed
identically.  No mapping state is accessed or mutated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
    ``max_neurons`` / ``hardware_bias``) plus the three independent permission
    bits (``allow_coalescing`` = inter-core membrane partial-sum transfer,
    ``allow_neuron_splitting`` = a wide channel may span cores,
    ``allow_scheduling`` = passes may run on fresh core pools). They are
    *declared* once; the per-layer mapping *strategy* is derived from them
    (see :class:`MappingStrategy`).
    """

    max_axons: int | None = None
    max_neurons: int | None = None
    hardware_bias: bool = False
    allow_coalescing: bool = False
    allow_neuron_splitting: bool = False
    allow_scheduling: bool = False


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

    @property
    def allow_coalescing(self) -> bool:
        return self.capabilities.allow_coalescing

    @property
    def allow_neuron_splitting(self) -> bool:
        return self.capabilities.allow_neuron_splitting

    @property
    def allow_scheduling(self) -> bool:
        return self.capabilities.allow_scheduling

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
