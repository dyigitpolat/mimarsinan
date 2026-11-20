"""[U-LIF2 W1] The firing discipline of a synchronized LIF hop: the joint legal table of ``lif_window_firing``, one
function of the resolved point, and the resolution of the run's value (the vocabulary and the declared set and default,
the registry entry's data, live in ``window_firing_axis``). The axis applies at the (lif, synchronized) point of the
spiking core semantics only; it is independent of the soma axis ``firing_granularity`` (WHEN a threshold is compared
inside one cycle).

Which values the toolchain offers is the declared set, data and never code: a document may hold only a declared value
that the joint table admits at its point, and a silent document holds the declared default. [U-LIF2 r3] While a reader
derives a stored run's saved document (``stored_point.derived_at_stored_point``), the entry's data read as the window
firing that run resolved (``stored_window_firing``), so a saved document is never resolved again under today's data."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Optional, Tuple

from mimarsinan.chip_simulation.activation_axes import LIF_FAMILY, SYNCHRONIZED_VARIANT
from mimarsinan.chip_simulation.activation_semantics import resolve_activation_semantics
from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics, resolve_core_semantics
from mimarsinan.chip_simulation.soma_axes import soma_keys_refusing
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.chip_simulation.spiking_semantics import is_default_firing_mode
from mimarsinan.chip_simulation import window_firing_axis as _axis
from mimarsinan.chip_simulation.window_firing_axis import (
    DEFAULT_WINDOW_FIRING as DEFAULT_WINDOW_FIRING,
    FIRE_WHILE_INTEGRATING as FIRE_WHILE_INTEGRATING,
    INTEGRATE_THEN_EMIT as INTEGRATE_THEN_EMIT,
    WINDOW_FIRING_KEY,
    WINDOW_FIRINGS,
)
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.common.refusal.legal_value import illegal_value_message


class WindowFiringRefusal(ToolchainRefusal, ValueError):
    """A window firing the declared set or the joint table does not admit at the resolved point, refused by name."""


#: [U-LIF2 r3] The window firing a stored run resolved, while a reader derives that run's saved document; unset (None)
#: everywhere else, where the entry's data are today's.
_STORED_WINDOW_FIRING: ContextVar[Optional[str]] = ContextVar("stored_window_firing", default=None)


@contextmanager
def stored_window_firing(value: Optional[str]) -> Iterator[None]:
    """[U-LIF2 r3] For the scope, the declared set is ``value`` alone and the default is ``value`` (``None``: today's
    data): a stored run's saved document derived at the window firing its run resolved. Entered only by the stored-run
    reader (``stored_point.derived_at_stored_point``, the ratchet t02c)."""
    token = _STORED_WINDOW_FIRING.set(value)
    try:
        yield
    finally:
        _STORED_WINDOW_FIRING.reset(token)


def declared_window_firings() -> Tuple[str, ...]:
    """The declared set, read at call time: the registry entry's legal set and the resolution read this one datum."""
    stored = _STORED_WINDOW_FIRING.get()
    return tuple(_axis.DECLARED_WINDOW_FIRINGS) if stored is None else (stored,)


def window_firing_declaration(config: Mapping[str, Any]) -> Any:
    """[U-LIF2 r4] What a document or config declares for the key (``None``: silent), the one read of the key: the
    resolution's, and the rule that a run takes the point its directory holds
    (``stored_point.run_at_directory_point``)."""
    return config.get(WINDOW_FIRING_KEY)


def window_firing_applies(config: Mapping[str, Any]) -> bool:
    """Whether the axis applies: the (lif, synchronized) point of the spiking core semantics. Total over raw documents
    and resolved configs; an unresolvable semantics axis is not a point here (its own keyed error is the single
    truth)."""
    if is_mvm_core_semantics(resolve_core_semantics(config)):
        return False
    try:
        semantics = resolve_activation_semantics(config)
    except ValueError:
        return False
    return semantics.family == LIF_FAMILY and semantics.variant == SYNCHRONIZED_VARIANT


def jointly_legal_window_firings(soma_law: SomaLaw) -> Tuple[str, ...]:
    """The joint legal table at a (lif, synchronized) point with this resolved soma law: every value of the axis, less
    ``integrate_then_emit`` under a reset other than the subtractive one (an input-free emission window of a zero-reset
    neuron emits at most one spike, not its window's count).

    Per-event comparison and an unsigned register never reach this table: their own keys refuse them at this point
    under both values (``SOMA_LEGAL_VALUES``, the soma contract). A signed register is admitted here; whether
    it holds a program's window sum is a property of the program, judged where the program is generated."""
    zero_reset = not is_default_firing_mode(soma_law.firing_mode)
    return tuple(value for value in WINDOW_FIRINGS if not (zero_reset and value == INTEGRATE_THEN_EMIT))


def legal_window_firings(config: Mapping[str, Any]) -> Optional[Tuple[str, ...]]:
    """The key's legal set at the config's point: ``None`` where the axis does not apply, else the declared values the
    joint table admits (an empty set: the point admits no declared value). [U-LIF2 r2, m-1; r3, n-1] A soma value its
    own key refuses at this point (any soma key: ``soma_keys_refusing``) never reaches the table: the declared set
    stands, so that key names the refusal at both doors and the table never states a legal set the point does not have."""
    if not window_firing_applies(config):
        return None
    if soma_keys_refusing(config):
        return declared_window_firings()
    joint = jointly_legal_window_firings(SomaLaw.resolve(config))
    return tuple(value for value in declared_window_firings() if value in joint)


def derived_window_firing(config: Mapping[str, Any]) -> Optional[str]:
    """What a silent document holds: the declared default where the axis applies, else nothing. The default does not
    yield to the point (a silent document whose point refuses it is refused, never moved to another value)."""
    if not window_firing_applies(config):
        return None
    stored = _STORED_WINDOW_FIRING.get()
    return _axis.DEFAULT_WINDOW_FIRING if stored is None else stored


def resolve_window_firing(config: Mapping[str, Any]) -> Optional[str]:
    """The run's window firing: the declaration, else the declared default, judged by the joint table within the
    declared set and refused by name outside it; ``None`` where the axis does not apply (a declaration there is
    refused by name)."""
    declared = window_firing_declaration(config)
    legal = legal_window_firings(config)
    if legal is None:
        if declared is not None:
            raise WindowFiringRefusal(
                f"{WINDOW_FIRING_KEY}={declared!r} is declared where the axis does not apply: it is the firing "
                f"discipline of a synchronized LIF hop (spiking_family='lif', spiking_variant='synchronized'). "
                f"Remove {WINDOW_FIRING_KEY}.")
        return None
    if declared is not None:
        if declared not in legal:
            raise WindowFiringRefusal(illegal_value_message(WINDOW_FIRING_KEY, declared, legal))
        return str(declared)
    default = derived_window_firing(config)
    if default not in legal:
        raise WindowFiringRefusal(
            f"{WINDOW_FIRING_KEY}: the declared default {default!r} is not legal at this point, which admits "
            f"{{{', '.join(repr(value) for value in legal)}}} of the declared values: "
            + (f"declare {WINDOW_FIRING_KEY} as one of them." if legal else
               "no declared value is legal here, so the point itself must change (a subtractive reset, "
               "firing_mode='Default', admits every value of the axis)."))
    return default


def derived_census_samples(config: Mapping[str, Any]) -> int:
    """[U-LIF2 W13, the design's m-6] The event census a run takes when it declares none: where no external backend
    executes the run's point, the record's census comes from the executor of record. A synchronized LIF run that
    integrates then emits (no external backend executes it, so none seals a census) reads SANA-FE's declared sample
    count on the Hard Core Mapping executor; every other run reads none. A point the table refuses reads none (its own
    refusal names it)."""
    try:
        firing = resolve_window_firing(config)
    except ValueError:
        return 0
    return int(config.get("sanafe_sample_count") or 1) if firing == INTEGRATE_THEN_EMIT else 0
