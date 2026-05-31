"""GPU postprocess ops + the torch-style shim around ``ffcv.Loader``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GPUResize:
    """uint8 → fp32 [0,1] on GPU, optional resize, optional grayscale reduction."""

    resize_to: Optional[int] = None
    interpolation: str = "bicubic"
    scale_255: bool = True
    to_grayscale: bool = False

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float()
        elif x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            x = x.float()
        if self.scale_255:
            x = x / 255.0

        if x.ndim == 4 and x.shape[1] != 3 and x.shape[-1] == 3:
            # H×W×C → C×H×W (FFCV ToTorchImage may leave it channels-last)
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.ndim == 3:
            x = x.unsqueeze(0)

        if self.resize_to is not None and (x.shape[-2] != self.resize_to or x.shape[-1] != self.resize_to):
            x = F.interpolate(
                x,
                size=(int(self.resize_to), int(self.resize_to)),
                mode=self.interpolation,
                align_corners=False,
                antialias=True if self.interpolation in ("bilinear", "bicubic") else False,
            )

        if self.to_grayscale:
            # Average the RGB-lifted channels back to 1 (e.g. for MNIST whose
            # beton is a 3× repeat of the original grayscale).
            x = x.mean(dim=1, keepdim=True)
        return x


@dataclass(frozen=True)
class GPUNormalize:
    """Apply per-channel ``(x - mean) / std`` on a fp32 GPU tensor."""

    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.229, 0.224, 0.225)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        mean_t = torch.tensor(self.mean[:channels], device=x.device, dtype=x.dtype).view(1, channels, 1, 1)
        std_t = torch.tensor(self.std[:channels], device=x.device, dtype=x.dtype).view(1, channels, 1, 1)
        return (x - mean_t) / std_t


@dataclass(frozen=True)
class GPUResizeNormalize:
    """Composite of :class:`GPUResize` then :class:`GPUNormalize` — one fewer kernel hop."""

    resize_to: Optional[int] = None
    interpolation: str = "bicubic"
    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.229, 0.224, 0.225)
    scale_255: bool = True
    to_grayscale: bool = False

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = GPUResize(
            resize_to=self.resize_to,
            interpolation=self.interpolation,
            scale_255=self.scale_255,
            to_grayscale=self.to_grayscale,
        ).apply(x)
        return GPUNormalize(mean=self.mean, std=self.std).apply(x)


class TorchLoaderShim:
    """Wrap an ``ffcv.Loader`` so it iterates like ``torch.utils.data.DataLoader``.

    Cloning x at the FFCV boundary owns the rotating-buffer contract on
    behalf of downstream consumers — FFCV yields tensor views into a small
    pre-allocated pool that gets overwritten as iteration advances, so any
    consumer holding a yielded tensor across iteration boundaries would
    silently see corrupted data without this clone.

    ``label_lookup`` (optional, set by :func:`build_loader` for FFCV loaders)
    replaces FFCV's per-batch label tensor with an indexed lookup into a
    pre-loaded label tensor — the FFCV ``IntDecoder`` is unreliable on
    ViT-class workloads under fused-multi-tensor-apply allocator state, so
    we bypass it entirely. Indices come from the ``IndexedLoader`` iterator
    via ``batch_index_queue``.
    """

    def __init__(self, ffcv_loader, postprocess=None, label_lookup=None):
        self._loader = ffcv_loader
        if postprocess is None:
            self._post_chain: list = []
        elif isinstance(postprocess, (list, tuple)):
            self._post_chain = list(postprocess)
        else:
            self._post_chain = [postprocess]
        # Mimic the attributes BasicTrainer reads off a torch DataLoader.
        self.batch_size = getattr(ffcv_loader, "batch_size", None)
        self.num_workers = getattr(ffcv_loader, "num_workers", 0)
        self._iterator = None
        self._label_lookup = label_lookup

    def close(self) -> None:
        """Explicitly close any held FFCV iterator + free its worker pool."""
        it = self._iterator
        if it is None:
            return
        self._iterator = None
        try:
            if hasattr(it, "close"):
                it.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __iter__(self) -> Iterator:
        it = iter(self._loader)
        self._iterator = it
        # Present on IndexedLoader iterators; None for stock ffcv.Loader.
        index_queue = getattr(it, "batch_index_queue", None)
        for batch in it:
            yield self._postprocess(batch, index_queue)

    def __len__(self) -> int:
        return len(self._loader)

    def _postprocess(self, batch, index_queue=None):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y_ffcv = batch
            x = x.clone()

            if self._label_lookup is not None and index_queue is not None:
                idx_np = index_queue.get_nowait()
                idx_t = torch.as_tensor(idx_np, dtype=torch.long, device=self._label_lookup.device)
                y = self._label_lookup.index_select(0, idx_t)
            else:
                y = y_ffcv.clone()

            for op in self._post_chain:
                x = op.apply(x)
            return x, y
        return batch
