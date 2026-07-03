"""Build :class:`IndexedLoader` instances from a :class:`PipelineSpec` + provider."""

from __future__ import annotations

from typing import Any, Callable

import torch
from ffcv.loader import OrderOption  # pyright: ignore[reportImplicitRelativeImport, reportAttributeAccessIssue]  # third-party ffcv, name-shadowed by this package

from mimarsinan.data_handling.ffcv.loader import IndexedLoader, preload_labels
from mimarsinan.data_handling.ffcv.pipeline_spec import (
    PipelineSpec,
    SplitSpec,
    normalize_split_name,
)
from mimarsinan.data_handling.ffcv.spec_builder import infer_spec, raw_dataset_for
from mimarsinan.data_handling.ffcv.writer import ensure_beton


def _build_ops(transform_chain, device):
    """Materialize op classes from a spec's (class_name, kwargs) entries."""
    import ffcv.transforms as ffcv_t  # pyright: ignore[reportMissingImports]  # optional ffcv backend
    from ffcv.fields import decoders as ffcv_decoders  # pyright: ignore[reportMissingImports]  # optional ffcv backend

    ops = []
    for entry in transform_chain:
        cls_name, kwargs = entry
        kwargs = dict(kwargs or {})
        cls = getattr(ffcv_t, cls_name, None) or getattr(ffcv_decoders, cls_name, None)
        if cls is None:
            raise ValueError(f"unknown FFCV op: {cls_name}")
        if cls_name == "ToDevice" and "device" not in kwargs:
            kwargs["device"] = device
        if cls_name == "Convert" and isinstance(kwargs.get("target_dtype"), str):
            kwargs["target_dtype"] = getattr(torch, kwargs["target_dtype"])
        ops.append(cls(**kwargs))
    return ops


def build_loader(
    spec: PipelineSpec,
    split: str,
    dataset_factory: Callable[[], Any],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> IndexedLoader:
    """Return an :class:`IndexedLoader` for ``split``.

    ``dataset_factory`` is a thunk returning the underlying torch dataset
    used to (a) write the beton on first request and (b) preload labels
    onto ``device`` for the indexed-lookup label path.
    """
    split = normalize_split_name(split)
    beton = ensure_beton(spec, split, dataset_factory)
    split_spec: SplitSpec = spec.splits[split]

    pipelines = {}
    for field in spec.fields:
        field_ops = [(cls, kw) for (fname, cls, kw) in split_spec.transforms if fname == field.name]
        pipelines[field.name] = _build_ops(field_ops, device)

    order = OrderOption.RANDOM if split_spec.shuffle else OrderOption.SEQUENTIAL

    raw_ds = dataset_factory()
    label_lookup = preload_labels(raw_ds).to(device)

    return IndexedLoader(
        str(beton),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        order=order,
        drop_last=bool(split_spec.drop_last),
        pipelines=pipelines,
        label_lookup=label_lookup,
    )


class FFCVLoaderFactory:
    """High-level façade: provider + split → ready-to-iterate FFCV loader."""

    def __init__(self, data_provider_factory, *, num_workers: int = 4, device: str = "cuda"):
        self._data_provider_factory = data_provider_factory
        self._num_workers = num_workers
        self._device = torch.device(device)

    def create_data_provider(self):
        return self._data_provider_factory.create()

    def _provider_spec(self, provider) -> PipelineSpec:
        return infer_spec(provider)

    def _loader(self, provider, split: str, batch_size: int) -> IndexedLoader:
        spec = self._provider_spec(provider)
        split_norm = normalize_split_name(split)
        ds_factory = lambda: raw_dataset_for(provider, split_norm)
        return build_loader(
            spec, split_norm, ds_factory,
            batch_size=batch_size, num_workers=self._num_workers, device=self._device,
        )

    def create_training_loader(self, batch_size, data_provider) -> IndexedLoader:
        return self._loader(data_provider, "train", batch_size)

    def create_validation_loader(self, batch_size, data_provider) -> IndexedLoader:
        return self._loader(data_provider, "val", batch_size)

    def create_test_loader(self, batch_size, data_provider) -> IndexedLoader:
        return self._loader(data_provider, "test", batch_size)
