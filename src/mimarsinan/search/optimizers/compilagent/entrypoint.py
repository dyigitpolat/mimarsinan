"""Entry-point anchor that registers ``MimarsinanLayoutBackend`` with compilagent."""

from __future__ import annotations

from compilagent import backend_registry

from .backend import MimarsinanLayoutBackend


BACKEND_ID = "mimarsinan_layout"


def register_backend() -> None:
    """Register ``MimarsinanLayoutBackend`` with compilagent's global registry; idempotent on duplicate ids."""

    if BACKEND_ID in backend_registry.ids():
        return
    backend_registry.register(BACKEND_ID, MimarsinanLayoutBackend)


__all__ = ["BACKEND_ID", "register_backend"]
