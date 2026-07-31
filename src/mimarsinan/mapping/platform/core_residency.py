"""Core residency: the per-core values a hardware core holds exactly one of."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
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


class Granularity(Enum):
    """At what grain a target stores a core-level quantity."""

    ABSENT = "absent"          # the target has no such quantity; any value is vacuous
    PER_CORE = "per_core"      # one register per hardware core -> constrains residency
    PER_NEURON = "per_neuron"  # one per neuron column -> stored per range, constrains nothing


ResidencyPolicy = dict


def default_residency_policy(*, value_domain: bool) -> ResidencyPolicy:
    """Per-quantity granularity for a target, defaulting per deployment domain.

    ``threshold`` and ``parameter_scale`` are not independent: on a spiking target
    ``export/chip_quantize`` folds the weight scale INTO the firing threshold and resets
    ``parameter_scale`` to 1.0, because the chip carries no scale register. A value-domain
    target is the mirror image -- it has no firing threshold at all, but does carry a
    quantization scale. Declaring a granularity PER QUANTITY expresses both without either
    encoding being hard-coded, and lets a target that carries both, or neither, say so.

    Conservative in the same sense as before: a quantity a target does carry defaults to
    PER_CORE, so nothing silently loses a constraint.
    """
    policy: ResidencyPolicy = {
        "activation_scale": Granularity.PER_CORE,
        "input_activation_scale": Granularity.PER_CORE,
        "boundary_grid": Granularity.PER_CORE,
    }
    if value_domain:
        policy["threshold"] = Granularity.ABSENT
        policy["parameter_scale"] = Granularity.PER_CORE
    else:
        policy["threshold"] = Granularity.PER_CORE
        policy["parameter_scale"] = Granularity.ABSENT
    return policy


def resolve_residency_policy(
    platform_constraints: Any = None, *, value_domain: bool = False
) -> ResidencyPolicy:
    """THE SSOT turning a target's declaration into a granularity per quantity.

    A target overrides any entry by name, e.g. ``{"threshold": "per_neuron"}`` for hardware with
    a threshold register per neuron column.
    """
    policy = default_residency_policy(value_domain=value_domain)
    declared = {}
    if platform_constraints is not None:
        declared = platform_constraints.get(RESIDENCY_KEY, {}) or {}
    for name, value in declared.items():
        if name not in ALL_SINGLETON_NAMES:
            raise ValueError(
                f"{RESIDENCY_KEY}: {name!r} is not a core-level quantity; known: "
                f"{sorted(ALL_SINGLETON_NAMES)}"
            )
        policy[name] = value if isinstance(value, Granularity) else Granularity(value)
    return policy


RESIDENCY_KEY = "core_value_granularity"


def constrained_names(policy: ResidencyPolicy) -> frozenset[str]:
    """The quantities that constrain residency: exactly those stored once per hardware core."""
    return frozenset(n for n, g in policy.items() if g is Granularity.PER_CORE)


def ungrouped_fallback_id(seed: int) -> int:
    """The id an UNGROUPED core gets: a unique negative, never colliding with a real class.

    THE one definition. It reads "unknown, so share with nothing", which is the maximally
    conservative answer and why a vehicle without provenance fragments completely.
    """
    return -(int(seed) + 1)


def provenance_group_id(perceptron_index: int | None, *, fallback: int) -> int:
    """The legacy proxy: the source perceptron, else a caller-supplied id.

    Kept only for the degraded reconstruction path, which has no layout record to read a class
    from. Every other producer uses ``residency_key``; see ``residency_basis`` on the spec for
    which one produced a given id.
    """
    return int(perceptron_index) if perceptron_index is not None else int(fallback)


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
    per_neuron = getattr(hard_core, "per_neuron_names", None)
    for prop in CORE_SINGLETON_PROPERTIES:
        incoming = getattr(softcore, prop.name, None)
        if prop.name not in names:
            # PER_NEURON is stored per range; ABSENT is vacuous on this target, so the first
            # value stands and nothing can be corrupted by it.
            if per_neuron is None or prop.name in per_neuron:
                _record_per_neuron(hard_core, softcore, prop.name, incoming)
            elif getattr(hard_core, prop.name, None) is None:
                setattr(hard_core, prop.name, incoming)
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


def residency_key(
    core: Any,
    *,
    constrained: Iterable[str] = ALL_SINGLETON_NAMES,
    weight_banks: Any = None,
) -> Hashable:
    """The equivalence class of ``core``: equal keys may share a hardware core.

    The key IS the constrained values, so two cores are merged only where the target genuinely
    permits it -- it can never authorize a merge the hardware cannot represent.

    ``weight_banks`` resolves bank-backed indirection. A bank-backed core's effective
    ``parameter_scale`` lives on its BANK, and `export/chip_quantize` reads it from there before
    re-expressing it as the core's threshold; reading the node's own field would report 1.0 for
    every sharer and merge cores that diverge the moment they are exported.
    """
    names = frozenset(constrained)
    parts: list[Hashable] = []
    for prop in CORE_SINGLETON_PROPERTIES:
        if prop.name not in names:
            continue
        parts.append(_hashable(_effective(core, prop.name, weight_banks)))
    return tuple(parts)


def _effective(core: Any, name: str, weight_banks: Any) -> Any:
    """The value that will actually be programmed, following bank-backed indirection."""
    own = getattr(core, name, None)
    if name != "parameter_scale" or weight_banks is None:
        return own
    bank_id = getattr(core, "weight_bank_id", None)
    if bank_id is None:
        return own
    bank = weight_banks.get(bank_id) if hasattr(weight_banks, "get") else None
    if bank is None:
        return own
    return getattr(bank, name, own)


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
