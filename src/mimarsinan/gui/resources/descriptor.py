"""Metadata + materialised source for a single GUI resource."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mimarsinan.gui.resources.sources import ResourceSource


@dataclass(frozen=True)
class ResourceDescriptor:
    """What a resource is, and what it is made of.

    ``kind``/``rid`` compose the URL path; ``source`` carries the materialised
    inputs and renders to bytes (binary) or JSON-safe values, tagged by
    ``media_type``. Every descriptor carries a source rather than a bare closure
    precisely so a run that renders nothing can still persist the resource.
    """

    kind: str
    rid: str
    source: ResourceSource
    media_type: str

    @property
    def producer(self) -> Callable[[], Any]:
        """The zero-arg render of this descriptor's source."""
        return self.source.render


__all__ = ["ResourceDescriptor"]
