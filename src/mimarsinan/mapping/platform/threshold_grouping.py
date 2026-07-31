"""Threshold grouping: which softcores a hardware core may host together, per declared target."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Hashable, Mapping

CONSTRAINT_KEY = "single_threshold_per_core"

# Conservative: a target keeps the constraint unless it declares support for
# independent per-column thresholds, so none silently loses a limit it relies on.
DEFAULT_SINGLE_THRESHOLD_PER_CORE = True

_UNCONSTRAINED_KEY = 0


class ThresholdGroupingPolicy(Enum):
    """What the target's hardware imposes on thresholds within one core."""

    UNCONSTRAINED = "unconstrained"
    SINGLE_THRESHOLD_PER_CORE = "single_threshold_per_core"


def resolve_threshold_grouping_policy(
    platform_constraints: Mapping[str, Any] | None,
) -> ThresholdGroupingPolicy:
    """THE SSOT mapping declared platform constraints onto a grouping policy."""
    declared = DEFAULT_SINGLE_THRESHOLD_PER_CORE
    if platform_constraints is not None:
        declared = bool(platform_constraints.get(CONSTRAINT_KEY, declared))
    return (
        ThresholdGroupingPolicy.SINGLE_THRESHOLD_PER_CORE
        if declared
        else ThresholdGroupingPolicy.UNCONSTRAINED
    )


def threshold_group_key(
    core: Any, *, policy: ThresholdGroupingPolicy
) -> Hashable:
    """The equivalence class of ``core`` under ``policy``; equal keys may share a hardware core.

    Under the constraint the key IS the threshold value, which is the hardware predicate itself,
    so two cores are merged only when the target genuinely permits it. Keying on provenance
    instead both forbids legal packings and, when provenance is absent, degenerates to one group
    per core. There is deliberately no such fallback here: an unreadable threshold under a
    constraining policy is a mapping error, not a reason to fragment.
    """
    if policy is ThresholdGroupingPolicy.UNCONSTRAINED:
        return _UNCONSTRAINED_KEY

    threshold = getattr(core, "threshold", None)
    if threshold is None:
        raise ValueError(
            f"{type(core).__name__} carries no threshold, but the target declares "
            f"{CONSTRAINT_KEY}=True, so its threshold group cannot be determined"
        )
    value = float(threshold)
    if not math.isfinite(value):
        raise ValueError(
            f"{type(core).__name__} has a non-finite threshold ({value!r}); a hardware "
            f"threshold register cannot hold it"
        )
    return value


def ungrouped_fallback_id(seed: int) -> int:
    """The id an UNGROUPED core gets: a unique negative, never colliding with a real group.

    THE one definition. It reads "unknown, so share with nothing", which is the maximally
    conservative answer and the reason a vehicle without provenance fragments completely; four
    copies of it made that behaviour impossible to change in one place.
    """
    return -(int(seed) + 1)


def provenance_group_id(perceptron_index: int | None, *, fallback: int) -> int:
    """Today's id: the source perceptron, else a caller-supplied id.

    THE one place this rule is written. It is a proxy -- the hardware constrains per-core VALUES,
    not provenance -- and it is replaced by ``residency_key`` in the next step; routing every
    assignment site through here first makes that swap a single edit instead of three, and makes
    it impossible for the layout planner and the runtime packer to key differently.
    """
    return int(perceptron_index) if perceptron_index is not None else int(fallback)


class ThresholdGroupInterner:
    """Stable small integer ids for grouping keys within one mapping.

    Ids are only ever compared inside a single mapping, so interning per mapping is enough and
    avoids a global registry. Assignment follows first-seen order, so a given mapping is
    reproducible.
    """

    def __init__(self, policy: ThresholdGroupingPolicy | None = None) -> None:
        self._policy = policy if policy is not None else resolve_threshold_grouping_policy(None)
        self._ids: dict[Hashable, int] = {}

    @property
    def policy(self) -> ThresholdGroupingPolicy:
        return self._policy

    def id_for(self, core: Any) -> int:
        key = threshold_group_key(core, policy=self._policy)
        if key not in self._ids:
            self._ids[key] = len(self._ids)
        return self._ids[key]
