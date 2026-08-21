"""``SomaLaw``: the resolved per-neuron firing law every consumer takes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.chip_simulation.soma_axes import (
    FIRING_GRANULARITY_KEY,
    MEMBRANE_ARITHMETIC_KEY,
    MEMBRANE_BITS_KEY,
    PER_CYCLE_FIRING,
    PER_EVENT_FIRING,
    SATURATING_UNSIGNED_MEMBRANE,
    UNBOUNDED_MEMBRANE,
    resolved_firing_granularity,
    resolved_membrane_arithmetic,
    resolved_membrane_bits,
)
from mimarsinan.chip_simulation.spiking_semantics import (
    DEFAULT_FIRING_MODE,
    DEFAULT_THRESHOLDING_MODE,
)

_FIRING_MODE_KEY = "firing_mode"
_THRESHOLDING_MODE_KEY = "thresholding_mode"

# Where the always-on (parameter-encoded) bias row sits in the canonical event
# order. v1 fixes it at the tail; multi-row / head placements are new machinery.
BIAS_SLOT_TAIL = "tail"

_LAW_KEYS = (
    _FIRING_MODE_KEY, _THRESHOLDING_MODE_KEY, FIRING_GRANULARITY_KEY,
    MEMBRANE_ARITHMETIC_KEY, MEMBRANE_BITS_KEY,
)


def _mapping_view(source: Any) -> Mapping[str, Any]:
    """One reading surface for both shapes the law resolves from: a resolved
    config mapping and a resolved contract carrying the same names."""
    if isinstance(source, Mapping):
        return source
    view: Dict[str, Any] = {key: getattr(source, key, None) for key in _LAW_KEYS}
    return view


@dataclass(frozen=True)
class SomaLaw:
    """One resolved point of the soma axes — reset, compare, granularity,
    membrane arithmetic and width, and where the bias row sits.

    The capability query, the cell identity, the kernels, and (later) the
    exporter manifest all consume THIS; nobody reads the raw keys.
    """

    firing_mode: str
    thresholding_mode: str
    firing_granularity: str
    membrane_arithmetic: str
    membrane_bits: int
    bias_slot: str = BIAS_SLOT_TAIL

    @classmethod
    def resolve(cls, source: Any) -> "SomaLaw":
        """THE constructor: the resolved point of a config mapping or contract."""
        view = _mapping_view(source)
        firing_mode = view.get(_FIRING_MODE_KEY)
        thresholding_mode = view.get(_THRESHOLDING_MODE_KEY)
        return cls(
            firing_mode=(
                DEFAULT_FIRING_MODE if firing_mode is None else str(firing_mode)
            ),
            thresholding_mode=(
                DEFAULT_THRESHOLDING_MODE if thresholding_mode is None
                else str(thresholding_mode)
            ),
            firing_granularity=resolved_firing_granularity(view),
            membrane_arithmetic=resolved_membrane_arithmetic(view),
            membrane_bits=resolved_membrane_bits(view),
        )

    @property
    def is_per_event(self) -> bool:
        """The threshold is evaluated after every arriving event occurrence."""
        return self.firing_granularity == PER_EVENT_FIRING

    @property
    def saturates(self) -> bool:
        """The membrane clamps to ``[0, 2**membrane_bits - 1]`` on every update."""
        return self.membrane_arithmetic == SATURATING_UNSIGNED_MEMBRANE

    @property
    def membrane_bounds(self) -> Optional[Tuple[float, float]]:
        """The representable membrane interval, or ``None`` when unbounded.

        An unsigned fixed-width register holds ``[0, 2**bits - 1]`` and clamps
        on EVERY update; the default accumulator declares no interval, so a
        consumer that reads ``None`` keeps today's arithmetic exactly.
        """
        if not self.saturates:
            return None
        return (0.0, float(2 ** int(self.membrane_bits) - 1))

    @property
    def membrane_lattice_quantum(self) -> Optional[float]:
        """The exact arithmetic quantum of the membrane, or ``None``.

        A fixed-width unsigned register counts in whole LSBs, so the quantum
        is 1 in the chip's own units — the value a per-event snap projects
        onto. An unbounded accumulator declares no lattice here: the default
        points keep their own ``membrane_integer_lattice`` machinery
        untouched (plan §2.4).
        """
        return 1.0 if self.saturates else None

    @property
    def is_default_point(self) -> bool:
        """The law every pre-axes configuration resolves to (byte-identical)."""
        return (
            self.firing_granularity == PER_CYCLE_FIRING
            and self.membrane_arithmetic == UNBOUNDED_MEMBRANE
            and self.membrane_bits == 0
        )

    def point_tag(self) -> Optional[str]:
        """The identity suffix a NON-default point carries into cell keys;
        ``None`` at the default point, so every historical key is unchanged."""
        parts = []
        if self.is_per_event:
            parts.append(PER_EVENT_FIRING)
        if self.saturates:
            parts.append(f"sat{self.membrane_bits}")
        elif self.membrane_bits:
            parts.append(f"bits{self.membrane_bits}")
        return "-".join(parts) or None


DEFAULT_SOMA_LAW = SomaLaw.resolve({})
