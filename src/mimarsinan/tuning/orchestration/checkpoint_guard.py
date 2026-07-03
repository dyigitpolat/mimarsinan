"""CheckpointGuard — scoped, location-aware model snapshots for rollback."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch

from mimarsinan.tuning.learning_rate_explorer import (
    clone_state_for_trainer,
    restore_state_for_trainer,
)


def _to_pinned_host(src: "torch.Tensor") -> "torch.Tensor":
    """Copy ``src`` to a pinned host buffer with a non-blocking d2h (CUDA only).

    A fresh pinned buffer per call keeps nested snapshots from aliasing each other;
    the caller issues a single ``cuda.synchronize`` after launching all copies so
    they overlap. CPU tensors get a plain clone."""
    if src.is_cuda:
        dst = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
        dst.copy_(src, non_blocking=True)
        return dst
    return src.to("cpu", copy=True)


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

        if self.location == "cpu_pinned":
            state = {k: _to_pinned_host(v.detach()) for k, v in items.items()}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            state = {k: v.detach().clone() for k, v in items.items()}
        return Handle(state, self.scope, self.location)

    def restore(self, handle: Handle) -> None:
        if handle.scope == "full" and handle.location == "device":
            restore_state_for_trainer(self.trainer, handle.state)
            return

        device = getattr(self.trainer, "device", None)
        state = handle.state
        if device is not None:
            non_blocking = (
                handle.location == "cpu_pinned"
                and str(device) != "cpu"
                and torch.cuda.is_available()
            )
            state = {k: v.to(device, non_blocking=non_blocking) for k, v in state.items()}
            if non_blocking:
                torch.cuda.synchronize()
        # ``strict=False``: tunable scope intentionally omits frozen params.
        self.trainer.model.load_state_dict(state, strict=False)

    @contextlib.contextmanager
    def bracket(self):
        """Snapshot on enter; expose the handle. Caller restores on rollback."""
        handle = self.snapshot()
        yield handle
