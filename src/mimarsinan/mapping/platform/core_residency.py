"""Core residency: the per-core values a hardware core holds exactly one of."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable

import numpy as np


def _scalar_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return False
    return fa == fb


def _tensor_equal(a: Any, b: Any) -> bool:
    """Value equality for the scale tensors, tolerant of scalar/array/tensor spellings."""
    if a is None or b is None:
        return a is b
    try:
        av = np.asarray(a.detach().cpu() if hasattr(a, "detach") else a, dtype=np.float64)
        bv = np.asarray(b.detach().cpu() if hasattr(b, "detach") else b, dtype=np.float64)
    except (TypeError, ValueError):
        return a is b
    return av.shape == bv.shape and bool(np.array_equal(av, bv))


def _object_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if a is b:
        return True
    verdict = a == b
    # An array-valued __eq__ answers elementwise; every element must agree.
    return bool(verdict) if isinstance(verdict, bool) else bool(np.all(verdict))


@dataclass(frozen=True)
class SingletonProperty:
    """A per-core value a hardware core stores once, shared by every softcore placed in it."""

    name: str
    compare: Callable[[Any, Any], bool]


# THE SSOT. A hardware core holds ONE of each of these, so softcores may only share a core when
# they agree on all of them. Adding a per-core value later means adding it here -- and nowhere
# else -- or it silently becomes another way for a merge to corrupt.
#
# `latency` is deliberately absent: schedule_split aggregates it as a max over the bundle and
# segment_id partitions separately, so it is not an equality constraint.
# `hardware_bias` is absent because add_softcore merges it per neuron range rather than adopting.
CORE_SINGLETON_PROPERTIES: tuple[SingletonProperty, ...] = (
    SingletonProperty("threshold", _scalar_equal),
    SingletonProperty("activation_scale", _tensor_equal),
    SingletonProperty("parameter_scale", _tensor_equal),
    SingletonProperty("input_activation_scale", _tensor_equal),
    SingletonProperty("boundary_grid", _object_equal),
)

ALL_SINGLETON_NAMES: frozenset[str] = frozenset(p.name for p in CORE_SINGLETON_PROPERTIES)


class CoreResidencyViolation(AssertionError):
    """A softcore was placed in a hardware core that cannot represent its per-core values."""


def adopt_or_check(
    hard_core: Any,
    softcore: Any,
    *,
    constrained: Iterable[str] = ALL_SINGLETON_NAMES,
) -> None:
    """First softcore into a hardware core sets its singleton values; later ones must agree.

    Adopting silently, as this once did, discards the later core's value and leaves it computing
    against the first core's -- a wrong result with no signal. Raising here is what turns a
    residency key that is too coarse into a loud, located failure.

    ``constrained`` is the target's declaration. A property it does NOT constrain is one the
    hardware stores per neuron rather than per core, so it is recorded per neuron range instead
    of being checked -- relaxing a constraint means storing the values, never skipping the check.
    """
    names = frozenset(constrained)
    for prop in CORE_SINGLETON_PROPERTIES:
        incoming = getattr(softcore, prop.name, None)
        if prop.name not in names:
            _record_per_neuron(hard_core, softcore, prop.name, incoming)
            continue
        current = getattr(hard_core, prop.name, None)
        if current is None:
            setattr(hard_core, prop.name, incoming)
            continue
        if not prop.compare(current, incoming):
            raise CoreResidencyViolation(
                f"softcore {getattr(softcore, 'id', '?')} has {prop.name}={incoming!r} but the "
                f"hardware core already holds {prop.name}={current!r}; a core stores one value, "
                f"so these may not share it"
            )


PER_NEURON_ATTR = "per_neuron_values"


def _record_per_neuron(hard_core: Any, softcore: Any, name: str, value: Any) -> None:
    """Write an unconstrained property across the neuron range this softcore occupies.

    Mirrors how ``hardware_bias`` is already merged. Without this a target that declares, say,
    per-neuron thresholds would pack cores together and then represent only the first one's
    value -- claiming a capability it does not store.
    """
    total = int(getattr(hard_core, "neurons_per_core", 0))
    width = int(softcore.get_output_count())
    offset = total - int(getattr(hard_core, "available_neurons", 0)) - width
    if total <= 0 or width <= 0 or offset < 0:
        return
    store = getattr(hard_core, PER_NEURON_ATTR, None)
    if store is None:
        store = {}
        setattr(hard_core, PER_NEURON_ATTR, store)
    column = store.get(name)
    if column is None:
        column = [None] * total
        store[name] = column
    for i in range(offset, offset + width):
        column[i] = value


def residency_key(core: Any, *, constrained: Iterable[str] = ALL_SINGLETON_NAMES) -> Hashable:
    """The equivalence class of ``core``: equal keys may share a hardware core.

    The key IS the constrained values, so two cores are merged only where the target genuinely
    permits it -- it can never authorize a merge the hardware cannot represent.
    """
    names = frozenset(constrained)
    parts: list[Hashable] = []
    for prop in CORE_SINGLETON_PROPERTIES:
        if prop.name not in names:
            continue
        parts.append(_hashable(getattr(core, prop.name, None)))
    return tuple(parts)


def _hashable(value: Any) -> Hashable:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return repr(value)
    if arr.ndim == 0:
        return float(arr)
    return (arr.shape, arr.tobytes())
