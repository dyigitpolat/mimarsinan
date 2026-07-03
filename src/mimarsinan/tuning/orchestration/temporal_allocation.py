"""Per-layer-S temporal-allocation axis — declare the intent, reserve the map."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

S_ALLOCATION_UNIFORM = "uniform"
S_ALLOCATION_EXPLICIT = "explicit"
S_ALLOCATION_BUDGET = "budget"

S_ALLOCATION_MODES: Tuple[str, ...] = (
    S_ALLOCATION_UNIFORM,
    S_ALLOCATION_EXPLICIT,
    S_ALLOCATION_BUDGET,
)

S_ALLOCATION_SUPPORTED_MODES: Tuple[str, ...] = (S_ALLOCATION_UNIFORM,)

BUDGET_DERIVATION_DEFERRED = (
    "derivation deferred to research (per-depth S allocator not on this branch)"
)

_VALID_BUDGET_KEYS = frozenset(
    {"max_energy_proxy", "max_latency_steps", "target"}
)

__all__ = [
    "S_ALLOCATION_UNIFORM",
    "S_ALLOCATION_EXPLICIT",
    "S_ALLOCATION_BUDGET",
    "S_ALLOCATION_MODES",
    "S_ALLOCATION_SUPPORTED_MODES",
    "BUDGET_DERIVATION_DEFERRED",
    "resolve_s_allocation_mode",
    "unsupported_s_allocation_error",
    "TemporalAllocation",
    "TemporalAllocationResolver",
]


def unsupported_s_allocation_error(mode: str) -> str:
    """Canonical loud-reject message for a reserved (unwired) ``s_allocation`` mode."""
    return (
        f"s_allocation={mode!r} is reserved/not implemented; "
        "only 'uniform' is supported"
    )


def resolve_s_allocation_mode(config: Mapping[str, Any]) -> str:
    """The ``uniform | explicit | budget`` allocation axis (default ``uniform``);
    an unknown value raises rather than silently falling back."""
    raw = config.get("s_allocation")
    if raw is None:
        return S_ALLOCATION_UNIFORM
    value = str(raw).lower()
    if value in S_ALLOCATION_MODES:
        return value
    raise ValueError(
        f"s_allocation must be one of {S_ALLOCATION_MODES}, got {raw!r}"
    )


@dataclass(frozen=True)
class TemporalAllocation:
    """The resolved per-depth S map (reserved axis) for one model.

    ``per_depth_steps`` is ``S_d`` for each cascade depth; the default repeats the
    global ``simulation_steps``. ``derivation_deferred`` flags a reserved mode that
    returned uniform because its derivation is not landed yet."""

    mode: str
    per_depth_steps: Tuple[int, ...]
    global_steps: int
    derivation_deferred: Optional[str] = None
    budget: Mapping[str, Any] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return len(self.per_depth_steps)

    @property
    def is_uniform(self) -> bool:
        """True when every depth gets the global S (the byte-identical default)."""
        return all(s == self.global_steps for s in self.per_depth_steps)

    def steps_at(self, depth_index: int) -> int:
        """``S_d`` for depth ``depth_index`` (0-based)."""
        return self.per_depth_steps[depth_index]


@dataclass(frozen=True)
class TemporalAllocationResolver:
    """Resolve the per-depth S map from a config + a per-model depth.

    ``uniform`` repeats the global ``simulation_steps``; ``explicit`` validates a
    declared list; ``budget`` is a no-op returning uniform + the
    ``derivation_deferred`` marker. Nothing threads a non-uniform map yet.
    """

    mode: str
    global_steps: int
    explicit: Optional[Tuple[int, ...]] = None
    budget: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TemporalAllocationResolver":
        """Resolve the resolver from a flat config dict. The per-depth list / budget
        body are parsed + validated here (so a malformed declaration fails loud) even
        though no map is threaded yet."""
        mode = resolve_s_allocation_mode(config)
        global_steps = _resolve_global_steps(config)
        explicit = _parse_explicit(config.get("s_allocation_explicit"))
        budget = _parse_budget(config.get("s_allocation_budget"))
        return cls(
            mode=mode,
            global_steps=global_steps,
            explicit=explicit,
            budget=budget,
        )

    def resolve(self, *, depth: int) -> TemporalAllocation:
        """Return the per-depth S map for a model of ``depth`` cascade depths (the
        caller owns ``depth``; this layer does not introspect the model)."""
        if depth <= 0:
            raise ValueError(f"depth must be a positive int, got {depth!r}")

        if self.mode == S_ALLOCATION_EXPLICIT:
            return self._resolve_explicit(depth)
        if self.mode == S_ALLOCATION_BUDGET:
            return self._resolve_budget(depth)
        return self._uniform(depth)

    def _uniform(self, depth: int, *, derivation_deferred=None) -> TemporalAllocation:
        return TemporalAllocation(
            mode=self.mode,
            per_depth_steps=tuple([self.global_steps] * depth),
            global_steps=self.global_steps,
            derivation_deferred=derivation_deferred,
            budget=dict(self.budget),
        )

    def _resolve_explicit(self, depth: int) -> TemporalAllocation:
        if self.explicit is None:
            raise ValueError(
                "s_allocation='explicit' requires s_allocation_explicit "
                "(a per-depth list of positive ints)"
            )
        if len(self.explicit) != depth:
            raise ValueError(
                f"s_allocation_explicit has {len(self.explicit)} entries but the "
                f"model has depth {depth}; one S per cascade depth is required"
            )
        return TemporalAllocation(
            mode=self.mode,
            per_depth_steps=self.explicit,
            global_steps=self.global_steps,
            derivation_deferred=None,
            budget=dict(self.budget),
        )

    def _resolve_budget(self, depth: int) -> TemporalAllocation:
        return self._uniform(depth, derivation_deferred=BUDGET_DERIVATION_DEFERRED)


def _resolve_global_steps(config: Mapping[str, Any]) -> int:
    raw = config.get("simulation_steps")
    if raw is None:
        raise ValueError(
            "TemporalAllocation needs the global 'simulation_steps' (the uniform S)"
        )
    steps = int(raw)
    if steps <= 0:
        raise ValueError(f"simulation_steps must be a positive int, got {raw!r}")
    return steps


def _parse_explicit(raw: Any) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    if isinstance(raw, Mapping) or isinstance(raw, (str, bytes)):
        raise ValueError(
            f"s_allocation_explicit must be a list of positive ints, got {raw!r}"
        )
    try:
        values = tuple(int(v) for v in raw)
    except TypeError as exc:
        raise ValueError(
            f"s_allocation_explicit must be a list of positive ints, got {raw!r}"
        ) from exc
    if len(values) == 0:
        raise ValueError("s_allocation_explicit must be non-empty")
    if any(v <= 0 for v in values):
        raise ValueError(
            f"s_allocation_explicit entries must be positive ints, got {raw!r}"
        )
    return values


def _parse_budget(raw: Any) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"s_allocation_budget must be a dict, got {type(raw).__name__}"
        )
    unknown = set(raw) - _VALID_BUDGET_KEYS
    if unknown:
        raise ValueError(
            f"s_allocation_budget has unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(_VALID_BUDGET_KEYS)}"
        )
    return dict(raw)
