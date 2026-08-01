"""Materialised source data for one GUI resource: render it now, or persist it and render later."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from mimarsinan.gui import heatmap_renderer

_SOURCE_TYPES: dict[str, type["ResourceSource"]] = {}


class ResourceSource(ABC):
    """The materialised inputs of ONE renderable resource.

    Construction is the moment the pipeline's buffer is handed over: a source
    owns host memory only, so nothing it holds can pin a device allocation past
    the step that produced it. ``to_state``/``from_state`` make it the unit a run
    writes to disk when it renders nothing itself.
    """

    SOURCE_TYPE: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only a class DECLARING its own type claims that name on disk; a
        # subclass that inherits one is a variant, not a second format.
        source_type = cls.__dict__.get("SOURCE_TYPE", "")
        if not source_type:
            return
        registered = _SOURCE_TYPES.setdefault(source_type, cls)
        if registered is not cls:
            raise ValueError(
                f"resource source type {source_type!r} is already registered to {registered}"
            )

    @abstractmethod
    def render(self) -> Any:
        """The resource payload: bytes for binary media, JSON-safe values otherwise."""

    @abstractmethod
    def to_state(self) -> dict[str, Any]:
        """State ``from_state`` reconstructs from; ndarray values are stored as arrays."""

    @classmethod
    @abstractmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ResourceSource":
        """Rebuild a source from what ``to_state`` produced."""


def resource_source_from_state(source_type: str, state: Mapping[str, Any]) -> ResourceSource:
    """Rebuild the registered source named by ``source_type``; raises on an unknown name."""
    source_class = _SOURCE_TYPES.get(source_type)
    if source_class is None:
        raise ValueError(f"unknown resource source type {source_type!r}")
    return source_class.from_state(state)


def as_host_array(value: Any, *, copy: bool = True) -> np.ndarray:
    """A plain numeric host array for ``value``; raises on anything unusable.

    A torch tensor is ALWAYS copied out: ``Tensor.numpy()`` shares storage, so a
    view would keep the tensor -- and, on an accelerator, its device allocation --
    alive for as long as the source lives.
    """
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        array = np.array(value.detach().cpu().numpy(), copy=True)
    else:
        array = np.asarray(value)
        if copy:
            array = array.copy()
    if array.dtype.kind not in "fiub":
        raise TypeError(
            f"a resource source needs a numeric array; got dtype {array.dtype!r}"
        )
    return array


def _as_bool_mask(mask: Sequence[Any] | None) -> list[bool] | None:
    return None if mask is None else [bool(x) for x in mask]


class HeatmapSource(ResourceSource):
    """A weight matrix plus optional pruned row/column masks, rendered as a PNG heatmap."""

    SOURCE_TYPE = "heatmap"

    def __init__(
        self,
        matrix: Any,
        *,
        pruned_row_mask: Sequence[Any] | None = None,
        pruned_col_mask: Sequence[Any] | None = None,
        copy: bool = True,
    ) -> None:
        self.matrix = as_host_array(matrix, copy=copy)
        self.pruned_row_mask = _as_bool_mask(pruned_row_mask)
        self.pruned_col_mask = _as_bool_mask(pruned_col_mask)

    def render(self) -> bytes:
        return heatmap_renderer.render_heatmap_png_bytes(
            self.matrix,
            pruned_row_mask=self.pruned_row_mask,
            pruned_col_mask=self.pruned_col_mask,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix,
            "pruned_row_mask": self.pruned_row_mask,
            "pruned_col_mask": self.pruned_col_mask,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "HeatmapSource":
        return cls(
            state["matrix"],
            pruned_row_mask=state.get("pruned_row_mask"),
            pruned_col_mask=state.get("pruned_col_mask"),
            copy=False,
        )


class JsonSource(ResourceSource):
    """An already-extracted JSON-safe payload: its source data and rendered form coincide."""

    SOURCE_TYPE = "json"

    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def render(self) -> Any:
        return self.payload

    def to_state(self) -> dict[str, Any]:
        return {"payload": self.payload}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "JsonSource":
        return cls(state["payload"])


def encode_resource_payload(payload: Any, media_type: str) -> bytes | None:
    """The exact bytes stored and served for ``payload`` under ``media_type``.

    ``None`` means "not storable": an unsupported media type, a binary resource
    whose producer returned something other than bytes, or an unserialisable JSON
    payload. The one encoder both the run and the monitor go through, so a
    deferred render reproduces the eager bytes exactly.
    """
    if media_type == "image/png":
        return bytes(payload) if isinstance(payload, (bytes, bytearray)) else None
    if media_type == "application/json":
        try:
            return json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError):
            return None
    return None


__all__ = [
    "HeatmapSource",
    "JsonSource",
    "ResourceSource",
    "as_host_array",
    "encode_resource_payload",
    "resource_source_from_state",
]
