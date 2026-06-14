"""CheckpointGuard — scoped, location-aware model snapshots for rollback.

The legacy path clones the entire ``state_dict`` on-device every cycle. This
service makes the two real scaling levers explicit (the LR sweep already clones
once, so per-probe cloning was never the issue):

- ``scope="tunable"`` clones only ``requires_grad`` params (+ all buffers),
  skipping a frozen backbone — the big win for partial fine-tuning.
- ``location="cpu_pinned"`` offloads the snapshot to CPU, freeing ~1× model of
  VRAM; an explicit stream sync precedes restore so the d2h copy has landed.

``scope="full"``/``location="device"`` (the default) delegates verbatim to
``clone_state_for_trainer``/``restore_state_for_trainer`` and is byte-identical
to the legacy path (golden-trace safe). Non-default scope/location apply to
single-model trainers; aux-model trainers fall back to the full/device clone.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch

from mimarsinan.tuning.learning_rate_explorer import (
    clone_state_for_trainer,
    restore_state_for_trainer,
)


@dataclass
class Handle:
    """Opaque snapshot handle; ``state`` is the cloned model state."""

    state: Any
    scope: str
    location: str


class CheckpointGuard:
    """Snapshot/restore a trainer's model state with a chosen scope + location."""

    def __init__(self, trainer, *, scope: str = "full", location: str = "device"):
        if scope not in ("full", "tunable"):
            raise ValueError(f"unknown checkpoint scope: {scope!r}")
        if location not in ("device", "cpu_pinned"):
            raise ValueError(f"unknown checkpoint location: {location!r}")
        self.trainer = trainer
        self.scope = scope
        self.location = location

    def _is_default(self) -> bool:
        return self.scope == "full" and self.location == "device"

    def _has_aux(self) -> bool:
        return hasattr(self.trainer, "aux_model")

    def snapshot(self) -> Handle:
        if self._is_default() or self._has_aux():
            return Handle(clone_state_for_trainer(self.trainer), "full", "device")

        model = self.trainer.model
        full = model.state_dict()
        if self.scope == "tunable":
            grad_names = {n for n, p in model.named_parameters() if p.requires_grad}
            buffer_names = {n for n, _ in model.named_buffers()}
            keep = grad_names | buffer_names
            items = {k: v for k, v in full.items() if k in keep}
        else:
            items = full

        to_cpu = self.location == "cpu_pinned"
        state = {
            k: (v.detach().to("cpu", copy=True) if to_cpu else v.detach().clone())
            for k, v in items.items()
        }
        return Handle(state, self.scope, self.location)

    def restore(self, handle: Handle) -> None:
        if handle.scope == "full" and handle.location == "device":
            restore_state_for_trainer(self.trainer, handle.state)
            return

        if handle.location == "cpu_pinned" and torch.cuda.is_available():
            torch.cuda.synchronize()
        device = getattr(self.trainer, "device", None)
        state = handle.state
        if device is not None:
            state = {k: v.to(device) for k, v in state.items()}
        # ``strict=False``: tunable scope intentionally omits frozen params.
        self.trainer.model.load_state_dict(state, strict=False)

    @contextlib.contextmanager
    def bracket(self):
        """Snapshot on enter; expose the handle. Caller restores on rollback."""
        handle = self.snapshot()
        yield handle
