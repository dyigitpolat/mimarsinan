"""The introspection registry: registration, availability, and loud service.

Registration is the ONLY way a payload enters the program (a duplicate name
raises — new payloads are additions, never surgeries), and availability is asked
of a view, so "can this candidate/run answer X" is settled by the data. Serving
an unavailable payload raises with what it requires; the caller never receives a
silently empty answer that reads like a real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from mimarsinan.deployment_record.introspection.payloads import IntrospectionPayload

Builder = Callable[[Any], Optional[IntrospectionPayload]]


@dataclass(frozen=True)
class IntrospectionSpec:
    """One payload: what it answers, and how each view kind answers it."""

    name: str
    payload_type: type
    requires: str
    doc: str
    builders: Mapping[str, Builder]

    def __post_init__(self) -> None:
        for attribute in ("name", "requires", "doc"):
            if not getattr(self, attribute):
                raise ValueError(
                    f"IntrospectionSpec.{attribute} must be stated, got empty "
                    f"(payload {self.name!r})"
                )
        if not self.builders:
            raise ValueError(
                f"payload {self.name!r} declares no view it can be built from"
            )

    @property
    def version(self) -> int:
        return int(self.payload_type.VERSION)

    def build(self, view: Any) -> Optional[IntrospectionPayload]:
        """The payload for *view*, or ``None`` when its backing datum is absent."""
        builder = self.builders.get(view.view_kind)
        return None if builder is None else builder(view)

    def serve(self, view: Any) -> IntrospectionPayload:
        """The payload, or a loud refusal naming what it requires."""
        payload = self.build(view)
        if payload is None:
            raise ValueError(
                f"introspection payload {self.name!r} is not available on the "
                f"{view.view_kind} view: requires {self.requires}"
            )
        return payload


@dataclass
class IntrospectionRegistry:
    """An ordered set of payload specs, keyed by their (unique) name."""

    _specs: Dict[str, IntrospectionSpec] = field(default_factory=dict)

    def register(self, spec: IntrospectionSpec) -> IntrospectionSpec:
        if spec.name in self._specs:
            raise ValueError(
                f"introspection payload {spec.name!r} is already registered; new "
                f"payloads are additions, never surgeries on a registered one"
            )
        self._specs[spec.name] = spec
        return spec

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> Tuple[str, ...]:
        return tuple(self._specs)

    def all(self) -> Tuple[IntrospectionSpec, ...]:
        return tuple(self._specs.values())

    def get(self, name: str) -> IntrospectionSpec:
        if name not in self._specs:
            raise ValueError(
                f"unknown introspection payload {name!r}; registered: "
                f"{sorted(self._specs)}"
            )
        return self._specs[name]

    def serve(self, name: str, view: Any) -> IntrospectionPayload:
        """One named payload for *view* (raises when unavailable)."""
        return self.get(name).serve(view)

    def serve_all(self, view: Any) -> Dict[str, IntrospectionPayload]:
        """Every payload this view CAN answer, in registration order."""
        served: Dict[str, IntrospectionPayload] = {}
        for spec in self.all():
            payload = spec.build(view)
            if payload is not None:
                served[spec.name] = payload
        return served

    def serve_all_dicts(self, view: Any) -> Dict[str, Dict[str, Any]]:
        """:meth:`serve_all` as JSON-safe, self-describing envelopes."""
        return {
            name: payload.to_dict()
            for name, payload in self.serve_all(view).items()
        }

    def available_for(self, view: Any) -> Tuple[str, ...]:
        return tuple(self.serve_all(view))

    def catalog(self) -> Tuple[Dict[str, Any], ...]:
        """The declared surface: name, version, requirement, doc, view kinds."""
        return tuple(
            {
                "name": spec.name,
                "version": spec.version,
                "requires": spec.requires,
                "doc": spec.doc,
                "views": sorted(spec.builders),
            }
            for spec in self.all()
        )
