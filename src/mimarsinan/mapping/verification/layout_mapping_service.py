"""Two-tier bounded-LRU cache for the wizard / NAS layout-mapping pipeline; returned softcore lists are shared across cache hits, so do not mutate them in place."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable

from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest
from mimarsinan.mapping.verification.verifier import (
    MappingVerificationResult,
    verify_soft_core_mapping,
)
from mimarsinan.mapping.verification.wizard_layout_verify import (
    model_repr_from_wizard_body,
)


class _BoundedLRU:
    """Thread-safe ``OrderedDict``-backed bounded LRU with explicit ``invalidate()``."""

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError(f"_BoundedLRU: maxsize must be positive, got {maxsize}")
        self._maxsize = int(maxsize)
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks: dict[Any, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

    def get_or_compute(self, key: Any, factory: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]

        with self._key_locks_guard:
            kl = self._key_locks.setdefault(key, threading.Lock())

        with kl:
            with self._lock:
                if key in self._data:
                    self._data.move_to_end(key)
                    return self._data[key]
            value = factory()
            with self._lock:
                self._data[key] = value
                self._data.move_to_end(key)
                while len(self._data) > self._maxsize:
                    self._data.popitem(last=False)
            return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
        with self._key_locks_guard:
            self._key_locks.clear()


class LayoutMappingService:
    """Cache-fronted service for the wizard / NAS / snapshot layout calls.

    Two cache levels keyed on ``model_identity_key`` (model_repr) and
    ``verification_key``, so tiling-only differences share their model_repr.
    """

    def __init__(
        self,
        *,
        model_repr_maxsize: int = 16,
        verification_maxsize: int = 16,
    ) -> None:
        self._model_repr_cache = _BoundedLRU(model_repr_maxsize)
        self._verification_cache = _BoundedLRU(verification_maxsize)

    def get_model_repr(self, request: LayoutMappingRequest) -> Any:
        """Return the cached mapper repr for ``request``'s model identity, building it on first access."""
        return self._model_repr_cache.get_or_compute(
            request.model_identity_key(),
            lambda: self._build_model_repr(request),
        )

    def get_verification(
        self, request: LayoutMappingRequest,
    ) -> MappingVerificationResult:
        """Return the cached ``MappingVerificationResult`` for ``request``."""
        return self._verification_cache.get_or_compute(
            request.verification_key(),
            lambda: self._build_verification(request),
        )

    def invalidate(self) -> None:
        self._model_repr_cache.clear()
        self._verification_cache.clear()

    def _build_model_repr(self, request: LayoutMappingRequest) -> Any:
        return model_repr_from_wizard_body(request.to_body())

    def _build_verification(
        self, request: LayoutMappingRequest,
    ) -> MappingVerificationResult:
        model_repr = self.get_model_repr(request)
        return verify_soft_core_mapping(
            model_repr,
            max_axons=request.max_axons,
            max_neurons=request.max_neurons,
            allow_coalescing=request.allow_coalescing,
            hardware_bias=request.hardware_bias,
        )


DEFAULT_LAYOUT_MAPPING_SERVICE = LayoutMappingService()
