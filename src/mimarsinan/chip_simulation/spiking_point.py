"""[U-LIF2 W2] The run's resolved spiking point: one frozen record, resolved once, the one argument of every capability
judgment (the conversion recipe, the mode policy, the registry's assembly check, every runner and the configuration
surface).

``resolve_spiking_point`` is total over raw documents and resolved configs and idempotent: a raw document and the run
config derived from it give one record. The axes fold calls ``fold_spiking_point`` beside the family/variant and soma
folds, before the conversion recipe, so the recipe's backend enables read the window firing the run executes.

[U-LIF2 r2] No capability judgment and no policy build takes a point the run did not resolve: there is no point of a
mode alone. A holder that carries no run's point (a contract or behavior built from its fields or stored before the
point existed, an executor built from its own fields) states its own through ``stated_point``, with the window firing it
executes named explicitly, never the declared default.

[U-LIF2 r3] A point stored as data is read back as it was written (``point_record``, ``point_from_record``): a reader of a
stored run takes the point resolved when it was written, never a resolution under today's entry data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple

from mimarsinan.chip_simulation.activation_axes import LIF_FAMILY
from mimarsinan.chip_simulation.activation_semantics import resolve_activation_semantics
from mimarsinan.chip_simulation.backend_extensions import EXECUTION_EXTENSIONS, declared_extensions
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.chip_simulation.spiking_semantics import (
    is_synchronized_ttfs,
    is_ttfs_cycle_based,
    require_known_spiking_mode,
)
from mimarsinan.chip_simulation.window_firing import (
    FIRE_WHILE_INTEGRATING,
    INTEGRATE_THEN_EMIT,
    WINDOW_FIRING_KEY,
    WindowFiringRefusal,
    resolve_window_firing,
    window_firing_applies,
)
from mimarsinan.common.refusal import ToolchainRefusal

#: The retired internal key of the executor's evaluation form: a config that still carries it is refused by name, so
#: no value of it is read silently as another discipline.
_RETIRED_DISCIPLINE_KEY = "lif_execution_discipline"

#: [U-LIF2 r2] The window firing every run executed before the axis existed (every executor of the set export fires while
#: integrating): what a holder that predates the axis states, never the declared default.
HISTORICAL_WINDOW_FIRING = FIRE_WHILE_INTEGRATING


class UnresolvedPointRefusal(ToolchainRefusal, TypeError):
    """A capability question asked with something other than a resolved point (a mode alone), refused by name."""


#: The config keys the record resolves from: the basis every backend enable derives from (its ``derived_from``).
POINT_BASIS_KEYS: Tuple[str, ...] = (
    "spiking_mode", "ttfs_cycle_schedule", "spiking_family", "spiking_variant", "firing_mode", "firing_granularity",
    "membrane_arithmetic", "membrane_bits", "membrane_signed", WINDOW_FIRING_KEY, *EXECUTION_EXTENSIONS,
)


@dataclass(frozen=True)
class SpikingPoint:
    """One resolved point: the mode and its schedule (the internal twins of the family/variant axes), the variant,
    the soma law, the window firing (``None`` away from (lif, synchronized)) and the declared execution extensions."""

    spiking_mode: str
    ttfs_cycle_schedule: str
    spiking_variant: str
    soma_law: SomaLaw
    window_firing: Optional[str]
    execution_extensions: Tuple[str, ...]

    @property
    def is_synchronized_ttfs(self) -> bool:
        return is_synchronized_ttfs(self.spiking_mode, self.ttfs_cycle_schedule)

    @property
    def policy_schedule(self) -> Optional[str]:
        """The schedule a mode policy carries: the TTFS-cycle family's, ``None`` for every other mode."""
        return self.ttfs_cycle_schedule if is_ttfs_cycle_based(self.spiking_mode) else None

    @property
    def integrates_then_emits(self) -> bool:
        """The hop integrates its whole window, then emits the window's count (the two-window discipline)."""
        return self.window_firing == INTEGRATE_THEN_EMIT

    def point_tag(self) -> Optional[str]:
        """[U-LIF2 W7] The identity suffix a non-default point carries into cell keys: the soma law's tag, then a
        window firing other than fire_while_integrating (what every historical cell ran), so every historical key is
        unchanged and a new point never inherits another point's certificate or regression floor."""
        parts = [tag for tag in (self.soma_law.point_tag(),) if tag]
        if self.window_firing not in (None, FIRE_WHILE_INTEGRATING):
            parts.append(str(self.window_firing))
        return "-".join(parts) or None


def _point(config: Mapping[str, Any], soma_law: Optional[SomaLaw],
           window_firing: Callable[[SomaLaw], Optional[str]]) -> SpikingPoint:
    semantics = resolve_activation_semantics(config)
    mode = require_known_spiking_mode(semantics.legacy_spiking_mode)
    law = SomaLaw.resolve(config) if soma_law is None else soma_law
    return SpikingPoint(
        spiking_mode=mode,
        ttfs_cycle_schedule=semantics.legacy_ttfs_cycle_schedule,
        spiking_variant=semantics.variant,
        soma_law=law,
        window_firing=window_firing(law),
        execution_extensions=declared_extensions(config) if semantics.family == LIF_FAMILY else (),
    )


def resolve_spiking_point(config: Mapping[str, Any]) -> SpikingPoint:
    """THE resolution of a config's spiking point (raw document or resolved config): the family/variant axes through
    the legacy bridge, the soma law, the window firing judged by the joint table within the declared set (refused by
    name outside it) and the execution extensions the point declares. An unknown mode or axis raises its own keyed
    error."""
    if (retired := config.get(_RETIRED_DISCIPLINE_KEY)) is not None:
        raise WindowFiringRefusal(
            f"{_RETIRED_DISCIPLINE_KEY}={retired!r} is retired and read by no executor: the firing discipline of a "
            f"synchronized LIF hop is {WINDOW_FIRING_KEY}. Remove {_RETIRED_DISCIPLINE_KEY}.")
    return _point(config, None, lambda _law: resolve_window_firing(config))


def stated_point(record: Mapping[str, Any], *, window_firing: str, soma_law: Optional[SomaLaw] = None) -> SpikingPoint:
    """[U-LIF2 r2] The point a holder that carries no run's point states: an object built from its own fields or stored
    before the point existed (a contract, a behavior, an executor). Its own keys give the mode, the schedule, the
    variant, the soma law (or the holder's resolved law) and the execution extensions through the one resolution; the
    window firing is the one the holder executes, named by its caller (``HISTORICAL_WINDOW_FIRING`` for a holder that
    predates the axis) where the axis applies, never the declared set or its default: the holder states what it
    executes, and an executor that cannot run the statement refuses it by its own name (the count executor's zero-reset,
    per-event and saturating refusals), as the document door refuses a document. A record that states a window firing
    is a document: the run resolves it. Called only by the named holders (the ratchet t02c)."""
    for key in (WINDOW_FIRING_KEY, _RETIRED_DISCIPLINE_KEY):
        if record.get(key) is not None:
            raise WindowFiringRefusal(
                f"{key}={record[key]!r} is resolved by the run, never stated: a record that declares the window "
                f"firing is a document (resolve_spiking_point).")
    return _point(record, soma_law, lambda _law: window_firing if window_firing_applies(record) else None)


def point_record(point: SpikingPoint) -> Dict[str, Any]:
    """[U-LIF2 r3] The point as stored data, every field as it was resolved (``point_from_record`` reads it back)."""
    record = {field.name: getattr(point, field.name) for field in fields(SpikingPoint)}
    return {**record, "soma_law": asdict(point.soma_law), "execution_extensions": list(point.execution_extensions)}


def point_from_record(record: Mapping[str, Any]) -> SpikingPoint:
    """[U-LIF2 r3] The point a stored record holds, as it was resolved when written: neither a default nor the declared
    set is read. The soma law is read back by its one constructor from every field it holds, and a law that does not
    read back as written, or a record whose fields differ from the point's, is refused by name."""
    for held, kind in ((record, SpikingPoint), (record.get("soma_law") or {}, SomaLaw)):
        names = {field.name for field in fields(kind)}
        if set(held) != names:
            raise ValueError(f"a stored {kind.__name__} holds exactly {sorted(names)}; this one holds {sorted(held)}")
    law = SomaLaw.resolve(record["soma_law"])
    if asdict(law) != dict(record["soma_law"]):
        raise ValueError(f"a stored soma law reads back as {asdict(law)}, not as written: {dict(record['soma_law'])}")
    return SpikingPoint(**{**record, "soma_law": law, "execution_extensions": tuple(record["execution_extensions"])})


def recorded_window_firing(record: Mapping[str, Any]) -> Optional[str]:
    """[U-LIF2 f-1] The window firing a resolved config or a stored row states, for a coordinate that keys what a run
    executed (the coverage cell, the ledger row): at the (lif, synchronized) point its value as written, else
    ``HISTORICAL_WINDOW_FIRING`` (a row written before the axis), never the declared default; None away from the
    point. Nothing is resolved: a stored row reads back as written."""
    if not window_firing_applies(record):
        return None
    declared = record.get(WINDOW_FIRING_KEY)
    return HISTORICAL_WINDOW_FIRING if declared is None else str(declared)


def window_firing_apart(params: Mapping[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    """[U-LIF2 r6, M-2] The window firing a stored document's deployment parameters declare (as written, None where
    they are silent) and a copy of the parameters without the key: the one read of the key a stored run's statement
    takes (``stored_point``), held in the resolver beside ``recorded_window_firing``."""
    rest = dict(params)
    declared = rest.pop(WINDOW_FIRING_KEY, None)
    return (None if declared is None else str(declared)), rest


def recorded_point(record: Mapping[str, Any]) -> SpikingPoint:
    """[U-LIF2 r6, m-1] The point of a stored row's axes, resolved at the row's own window firing
    (``recorded_window_firing``: its value as written, else the historical one), never at the declared default."""
    firing = recorded_window_firing(record)
    return resolve_spiking_point(record if firing is None else {**record, WINDOW_FIRING_KEY: firing})


def fold_spiking_point(dp: MutableMapping[str, Any]) -> SpikingPoint:
    """Resolve the point and write its window firing into ``dp`` (idempotent), where the axes fold runs: before the
    conversion recipe derives a backend enable."""
    point = resolve_spiking_point(dp)
    if point.window_firing is not None:
        dp[WINDOW_FIRING_KEY] = point.window_firing
    return point
