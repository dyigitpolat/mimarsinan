"""Install/remove a cross-layer NF forward as a ``model.forward`` override."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw


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


class ChipAlignedNFForward(LazyExecutorForward):
    """Picklable ``model.forward`` override running the chip-aligned segment
    walk — the ONE deployed composition (boundary rounds, clamps, host-op
    domains). ``synchronized`` selects the value-domain walk (one eval per
    hop: LIF hops and their theorem-equal staircase QAT stand-ins both run
    it); the raw walk is the per-cycle deployed twin."""

    def __init__(
        self, model, T: int, retime: bool = False, phase_dither: bool = False,
        synchronized: bool = False, soma_law: SomaLaw = DEFAULT_SOMA_LAW,
    ):
        super().__init__(model, T)
        self.retime = bool(retime)
        self.phase_dither = bool(phase_dither)
        self.synchronized = bool(synchronized)
        self.soma_law = soma_law

    def _run(self, x):
        from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

        return chip_aligned_segment_forward(
            self.model, x, self.T, retime=getattr(self, "retime", False),
            phase_dither=getattr(self, "phase_dither", False),
            synchronized=getattr(self, "synchronized", False),
            soma_law=getattr(self, "soma_law", DEFAULT_SOMA_LAW),
        )


class CascadeForwardInstall:
    """Symmetric, single-owner install/remove of an instance ``model.forward``.

    Guarantees no double-patch (an unremoved prior wrapper would silently shadow the
    new one) and an idempotent unpatch, so downstream stages see the class forward.
    """

    if TYPE_CHECKING:
        # Host contract: the owning tuner supplies the patched model.
        model: Any

    _patched_forward = False

    def _install_forward(self, forward_obj) -> None:
        # A patch this tuner did not install is a PRIOR STAGE's persisted
        # forward (e.g. the exact-QAT training walk); installing over it is
        # the defined stage handoff. A double-install within one owner still
        # fails loud — that is the shadowing bug this guard exists for.
        assert not (self._patched_forward and "forward" in self.model.__dict__), (
            f"{type(self).__name__}: model.forward is already patched by this "
            "tuner; a double-install would shadow the prior wrapper. Remove it "
            "first."
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
