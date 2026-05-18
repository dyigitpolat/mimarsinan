"""Backend registration side effect for compilagent.

Compilagent's bootstrap calls ``load_entry_point_integrations()`` once per
session; integrations distributed via the ``compilagent.backends`` group
register themselves when their entry-point module is imported. This module
is the registration anchor — `pyproject.toml` advertises::

    [project.entry-points."compilagent.backends"]
    mimarsinan_layout = "mimarsinan.search.optimizers.compilagent.entrypoint"

so installing mimarsinan in a process that already has compilagent makes
the backend available to compilagent's CLI / observation UI without any
glue code on the user side. ``register_backend()`` is idempotent: it
checks the registry first, mirroring the in-tree integrations'
``if backend_id not in backend_registry.ids()`` guard.
"""

from __future__ import annotations

from compilagent import backend_registry

from .backend import MimarsinanLayoutBackend


BACKEND_ID = "mimarsinan_layout"


def register_backend() -> None:
    """Register ``MimarsinanLayoutBackend`` with compilagent's global registry.

    The registry currently raises ``ValueError`` on duplicate registration;
    we silently skip re-registration so re-importing the module (e.g. in
    tests, hot-reload, repeated `optimize()` calls) does not break.
    """

    if BACKEND_ID in backend_registry.ids():
        return
    backend_registry.register(BACKEND_ID, MimarsinanLayoutBackend)


__all__ = ["BACKEND_ID", "register_backend"]
