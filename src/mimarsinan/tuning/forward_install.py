"""Install/remove a cross-layer NF forward as a ``model.forward`` override."""

from __future__ import annotations


class LazyExecutorForward:
    """Picklable ``model.forward`` override running a cross-layer NF forward.

    Subclasses implement :meth:`_run`; the per-instance executor is built lazily via
    :meth:`_ensure_executor` and dropped on pickling so snapshots stay light.
    """

    def __init__(self, model, T: int):
        self.model = model
        self.T = int(T)
        self._executor = None

    def _unpatched_forward(self, x):
        """The model's class-level forward, bypassing this instance override."""
        return type(self.model).forward(self.model, x)

    def _ensure_executor(self, builder):
        """Return the cached executor, building it once via ``builder()``."""
        if self._executor is None:
            self._executor = builder()
        return self._executor

    def _run(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self._run(x)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_executor"] = None
        return state


class CascadeForwardInstall:
    """Symmetric, single-owner install/remove of an instance ``model.forward``.

    Guarantees no double-patch (an unremoved prior wrapper would silently shadow the
    new one) and an idempotent unpatch, so downstream stages see the class forward.
    """

    _patched_forward = False

    def _install_forward(self, forward_obj) -> None:
        assert "forward" not in self.model.__dict__, (
            f"{type(self).__name__}: model.forward is already patched; a double-"
            "install would shadow the prior wrapper. Remove it first."
        )
        self._patched_forward = True
        self.model.forward = forward_obj

    def _remove_forward(self) -> None:
        if getattr(self, "_patched_forward", False):
            try:
                del self.model.forward
            except AttributeError:
                pass
            self._patched_forward = False
