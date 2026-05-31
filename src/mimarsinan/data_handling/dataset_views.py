"""View wrappers that compose a raw dataset with a per-call transform."""

from __future__ import annotations

from typing import Any, Callable

import torch


class ApplyTransform(torch.utils.data.Dataset):
    """Apply ``transform`` to the image half of each ``(image, label)`` from ``base``.

    The label passes through untouched. ``transform`` may be a torchvision
    ``Compose`` or any callable that takes one image and returns one image.
    """

    def __init__(self, base: torch.utils.data.Dataset, transform: Callable[[Any], Any]):
        self._base = base
        self._transform = transform

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        x, y = self._base[idx]
        return self._transform(x), y
