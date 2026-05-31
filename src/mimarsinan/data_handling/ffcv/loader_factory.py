"""Build ``ffcv.Loader`` instances from a :class:`PipelineSpec` + provider."""

from __future__ import annotations

import os
from typing import Any, Callable

import torch

from mimarsinan.data_handling.ffcv.adapters import (
    GPUNormalize,
    GPUResize,
    GPUResizeNormalize,
    TorchLoaderShim,
)
from mimarsinan.data_handling.ffcv.cache import beton_path_for
from mimarsinan.data_handling.ffcv.pipeline_spec import (
    PipelineSpec,
    SplitSpec,
    normalize_split_name,
)
from mimarsinan.data_handling.ffcv.writer import ensure_beton


class FFCVNotAvailable(RuntimeError):
    """Raised when FFCV is requested but not importable."""


def _ffcv_enabled() -> bool:
    return os.environ.get("MIMARSINAN_PERF_FFCV", "").strip().lower() not in ("", "0", "false", "no", "off")


def available() -> bool:
    """Return ``True`` iff FFCV is importable and the global toggle is on."""
    if not _ffcv_enabled():
        return False
    try:
        import ffcv  # noqa: F401
    except Exception:
        return False
    return True


def _build_ops(transform_chain, device):
    """Materialize op classes from a spec's (class_name, kwargs) entries."""
    import ffcv.transforms as ffcv_t
    from ffcv.fields import decoders as ffcv_decoders

    ops = []
    for entry in transform_chain:
        cls_name, kwargs = entry
        kwargs = dict(kwargs or {})
        cls = getattr(ffcv_t, cls_name, None) or getattr(ffcv_decoders, cls_name, None)
        if cls is None:
            raise FFCVNotAvailable(f"unknown FFCV op: {cls_name}")
        if cls_name == "ToDevice" and "device" not in kwargs:
            kwargs["device"] = device
        if cls_name == "Convert" and isinstance(kwargs.get("target_dtype"), str):
            kwargs["target_dtype"] = getattr(torch, kwargs["target_dtype"])
        ops.append(cls(**kwargs))
    return ops


def _decoder_for(field_spec):
    if not field_spec.decode_type:
        return None
    from ffcv.fields import decoders as ffcv_decoders
    cls = getattr(ffcv_decoders, field_spec.decode_type, None)
    if cls is None:
        raise FFCVNotAvailable(f"unknown FFCV decoder: {field_spec.decode_type}")
    return cls()


def build_loader(
    spec: PipelineSpec,
    split: str,
    dataset_factory: Callable[[], Any],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    """Return a :class:`TorchLoaderShim` wrapping an ``IndexedLoader`` for ``split``.

    ``dataset_factory`` is a thunk returning the underlying torch dataset
    used to (a) write the beton on the first request and (b) preload labels
    into the on-device lookup that the shim uses to bypass FFCV's
    ``IntDecoder``.
    """
    if not available():
        raise FFCVNotAvailable("ffcv not available (need `pip install ffcv` and MIMARSINAN_PERF_FFCV=1)")

    from ffcv.loader import OrderOption
    from mimarsinan.data_handling.ffcv.label_passthrough import (
        IndexedLoader,
        preload_labels,
    )

    split = normalize_split_name(split)
    beton = ensure_beton(spec, split, dataset_factory)
    split_spec: SplitSpec = spec.splits[split]

    pipelines = {}
    for field in spec.fields:
        chain: list = []
        decoder = _decoder_for(field)
        if decoder is not None:
            chain.append(decoder)
        field_ops = [(cls, kw) for (fname, cls, kw) in split_spec.transforms if fname == field.name]
        chain.extend(_build_ops(field_ops, device))
        pipelines[field.name] = chain

    order = OrderOption.RANDOM if split_spec.shuffle else OrderOption.SEQUENTIAL

    ffcv_loader = IndexedLoader(
        str(beton),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        order=order,
        drop_last=bool(split_spec.drop_last),
        pipelines=pipelines,
    )

    # Preload labels on-device; the shim looks them up by batch_indices and
    # ignores FFCV's own per-batch label tensor (see label_passthrough.py).
    raw_ds = dataset_factory()
    label_lookup = preload_labels(raw_ds).to(device)

    chain = []
    for cls_name, kwargs in spec.gpu_postprocess:
        if cls_name == "GPUResizeNormalize":
            chain.append(GPUResizeNormalize(**(kwargs or {})))
        elif cls_name == "GPUResize":
            chain.append(GPUResize(**(kwargs or {})))
        elif cls_name == "GPUNormalize":
            chain.append(GPUNormalize(**(kwargs or {})))
        else:
            raise FFCVNotAvailable(f"unsupported gpu_postprocess op: {cls_name}")

    return TorchLoaderShim(
        ffcv_loader,
        postprocess=chain or None,
        label_lookup=label_lookup,
    )


class FFCVLoaderFactory:
    """High-level façade: provider + split → ready-to-iterate loader."""

    def __init__(self, data_provider_factory, *, num_workers: int = 4, device: str = "cuda"):
        self._data_provider_factory = data_provider_factory
        self._num_workers = num_workers
        self._device = torch.device(device)

    def create_data_provider(self):
        return self._data_provider_factory.create()

    def _provider_spec(self, provider) -> PipelineSpec:
        from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
        return infer_spec(provider)

    def _loader(self, provider, split: str, batch_size: int):
        from mimarsinan.data_handling.ffcv.spec_builder import raw_dataset_for

        spec = self._provider_spec(provider)
        split_norm = normalize_split_name(split)
        ds_factory = lambda: raw_dataset_for(provider, split_norm)
        return build_loader(
            spec, split_norm, ds_factory,
            batch_size=batch_size, num_workers=self._num_workers, device=self._device,
        )

    def create_training_loader(self, batch_size, data_provider):
        return self._loader(data_provider, "train", batch_size)

    def create_validation_loader(self, batch_size, data_provider):
        return self._loader(data_provider, "val", batch_size)

    def create_test_loader(self, batch_size, data_provider):
        return self._loader(data_provider, "test", batch_size)
