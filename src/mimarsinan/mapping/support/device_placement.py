"""Device-placement discipline: one device per model, borrowed modules restored."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class MixedDevicePlacementError(RuntimeError):
    """A model whose state straddles devices reached a single-device consumer."""


def _state_devices(module: nn.Module) -> dict[str, torch.device]:
    return {
        name: tensor.device
        for name, tensor in (
            *module.named_parameters(), *module.named_buffers(),
        )
    }


def single_device_of(module: nn.Module, *, what: str = "model") -> torch.device:
    """The ONE device ``module``'s parameters and buffers live on.

    Consumers that materialize an input for a model (probe forwards, warmup
    passes) all need a device, and the standing ``next(model.parameters()).device``
    idiom is correct only while the model is coherent: on a half-migrated model
    it answers with whichever tensor happens to come first, and the mismatch then
    surfaces inside some kernel far from whatever produced the split. This names
    the split instead. A state-free module is CPU -- nothing can disagree.
    """
    devices = _state_devices(module)
    if not devices:
        return torch.device("cpu")

    grouped: dict[torch.device, list[str]] = {}
    for name, device in devices.items():
        grouped.setdefault(device, []).append(name)
    if len(grouped) == 1:
        return next(iter(grouped))

    raise MixedDevicePlacementError(
        f"{what} is half-migrated across {len(grouped)} devices -- "
        f"{_format_split(grouped)}. A model must live on one device; whatever "
        "wrote or moved this state left it inconsistent, and every consumer "
        "that materializes an input for the model would be guessing which "
        "device the model is on."
    )


def _format_split(grouped: dict[torch.device, list[str]]) -> str:
    def _names(names: list[str]) -> str:
        head = ", ".join(sorted(names)[:4])
        return head if len(names) <= 4 else f"{head} (+{len(names) - 4} more)"

    return "; ".join(
        f"{device}: {_names(names)}"
        for device, names in sorted(grouped.items(), key=lambda kv: str(kv[0]))
    )


@contextmanager
def preserved_module_placement(module) -> Iterator[None]:
    """Restore ``module``'s parameter/buffer placement on exit, exactly.

    For BORROWED modules -- an ``IRGraph`` ``ComputeOp`` holds a live reference
    to the model's host op, so an analysis probing it with a fabricated tensor
    must leave it where it was. Restoration puts the ORIGINAL tensors back
    rather than moving the module home, which is exact and works for device
    pairs ``.to()`` cannot round-trip (``meta`` has no data to copy out of).
    """
    if not isinstance(module, nn.Module):
        yield
        return

    saved_params = [
        (owner, name, param, param.data)
        for owner in module.modules()
        for name, param in owner._parameters.items()
        if param is not None
    ]
    saved_buffers = [
        (owner, name, buffer)
        for owner in module.modules()
        for name, buffer in owner._buffers.items()
        if buffer is not None
    ]
    try:
        yield
    finally:
        for owner, name, param, data in saved_params:
            owner._parameters[name] = param
            param.data = data
        for owner, name, buffer in saved_buffers:
            owner._buffers[name] = buffer
