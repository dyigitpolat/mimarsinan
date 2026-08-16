"""The objectives registry: registration, availability, and loud resolution.

Registration is the ONLY way an axis enters the program, and a duplicate key
raises — new objectives are additions, never surgeries on an existing one.
Availability is asked of a view, so "can this run/candidate be optimized for X"
is answered by the data, never by a hand-maintained per-mode list. Resolution
fails loud on an unknown or unavailable name (the search surface's silent drop
is gone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from mimarsinan.deployment_record.objectives.spec import ObjectiveSpecV2, RecordView
from mimarsinan.deployment_record.objectives.probes import (
    candidate_capability_probe,
    run_capability_probe,
)
from mimarsinan.deployment_record.platform_physics.probe import probe_physics
from mimarsinan.deployment_record.objectives.views import SEARCH_MODES


@dataclass
class ObjectiveRegistry:
    """An ordered set of objective axes, keyed by their (unique) key."""

    _specs: Dict[str, ObjectiveSpecV2] = field(default_factory=dict)

    def register(self, spec: ObjectiveSpecV2) -> ObjectiveSpecV2:
        """Add one axis; a duplicate key is a defect, never a redefinition."""
        if spec.key in self._specs:
            raise ValueError(
                f"objective {spec.key!r} is already registered; new objectives "
                f"are additions, never surgeries on a registered one"
            )
        self._specs[spec.key] = spec
        return spec

    def __contains__(self, key: str) -> bool:
        return key in self._specs

    def keys(self) -> Tuple[str, ...]:
        """Every registered key, in registration order."""
        return tuple(self._specs)

    def all(self) -> Tuple[ObjectiveSpecV2, ...]:
        """Every registered spec, in registration order."""
        return tuple(self._specs.values())

    def get(self, key: str) -> ObjectiveSpecV2:
        """The named spec, or a loud failure listing what is registered."""
        if key not in self._specs:
            raise ValueError(
                f"unknown objective {key!r}; registered: {sorted(self._specs)}"
            )
        return self._specs[key]

    def available_for(self, view: RecordView) -> Tuple[ObjectiveSpecV2, ...]:
        """The axes whose backing datum is populated in *view*, in catalog order."""
        return tuple(spec for spec in self.all() if spec.available(view))

    def for_search_mode(self, search_mode: str) -> Tuple[ObjectiveSpecV2, ...]:
        """The axes a search in this mode can optimize (asked of a candidate probe)."""
        return self.available_for(candidate_capability_probe(search_mode))

    def search_catalog(self) -> Tuple[ObjectiveSpecV2, ...]:
        """Every axis available in at least one search mode, in catalog order."""
        searchable = {
            spec.key for mode in SEARCH_MODES for spec in self.for_search_mode(mode)
        }
        return tuple(spec for spec in self.all() if spec.key in searchable)

    def modes_available(self, key: str) -> Tuple[str, ...]:
        """The search modes the named axis is available in (possibly none)."""
        self.get(key)
        return tuple(
            mode
            for mode in SEARCH_MODES
            if any(spec.key == key for spec in self.for_search_mode(mode))
        )

    def resolve_active(
        self, search_mode: str, names: Sequence[str],
        *, probe: Optional[RecordView] = None,
    ) -> Tuple[ObjectiveSpecV2, ...]:
        """Validate a selection against the mode's catalog, preserving caller order.

        ``probe`` narrows availability to what a specific RUN can back (its declared
        physics, say). Omitted, the question stays capability-level — what the MODE
        could carry — which is what an offer-the-whole-catalog caller like the wizard
        asks.
        """
        selection = tuple(names)
        if not selection:
            raise ValueError(
                f"no objective names given for search mode {search_mode!r}; the "
                f"caller resolves its own defaults — an empty active set is a defect"
            )
        available = {
            spec.key for spec in (
                self.available_for(probe) if probe is not None
                else self.for_search_mode(search_mode)
            )
        }
        resolved = []
        seen = set()
        for name in selection:
            spec = self.get(name)
            if name in seen:
                raise ValueError(
                    f"objective {name!r} is repeated in the active selection "
                    f"{list(selection)}; an objective vector carries each axis once"
                )
            if name not in available:
                raise ValueError(
                    f"objective {name!r} is not available in search mode "
                    f"{search_mode!r}: requires {spec.requires}. Available: "
                    f"{sorted(available)}"
                )
            seen.add(name)
            resolved.append(spec)
        return tuple(resolved)

    def requires_physics(self, key: str) -> bool:
        """Whether this axis needs a declared physics profile to answer.

        Asked of the probes, not of a list: an axis requires physics exactly when a
        run that declares none cannot back it while one that declares everything can.
        """
        spec = self.get(key)
        widest = max(SEARCH_MODES, key=lambda mode: len(self.for_search_mode(mode)))
        return (
            spec.available(run_capability_probe(widest, probe_physics()))
            and not spec.available(run_capability_probe(widest, None))
        )

    def requires_activity(self, key: str) -> bool:
        """Whether this axis needs a declared switching activity to answer.

        Same probe discipline as ``requires_physics``: an axis requires the
        declaration exactly when a run stating none cannot back it at
        candidate time while one stating it can.
        """
        spec = self.get(key)
        widest = max(SEARCH_MODES, key=lambda mode: len(self.for_search_mode(mode)))
        return (
            spec.available(
                run_capability_probe(widest, probe_physics(), activity_factor=1.0)
            )
            and not spec.available(
                run_capability_probe(widest, probe_physics(), activity_factor=0.0)
            )
        )

    def extract(self, view: RecordView) -> Dict[str, float]:
        """Every available axis' value on *view*, keyed by objective key."""
        return {spec.key: spec.value(view) for spec in self.available_for(view)}
