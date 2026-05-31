"""Generic ``PipelineSpec`` inference from a :class:`DataProvider`."""

from __future__ import annotations

import torch

from mimarsinan.data_handling.ffcv.pipeline_spec import (
    FieldSpec,
    PipelineSpec,
    SplitSpec,
)


def _provider_id(provider) -> str:
    return type(provider).__name__.lower().replace("_dataprovider", "").replace("_provider", "")


def _resolve_normalize(preproc) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if preproc is None or preproc.mean is None or preproc.std is None:
        return None
    return tuple(preproc.mean), tuple(preproc.std)


def _resolve_resize(preproc) -> int | None:
    if preproc is None:
        return None
    return int(preproc.resize_to) if preproc.resize_to else None


class _AsRGB(torch.utils.data.Dataset):
    """Lift a (grayscale PIL, int) dataset to 3-channel RGB.

    FFCV's stock ``RGBImageField`` requires 3 channels at write time; the
    grayscale model input is reconstituted via the GPU postprocess's
    ``to_grayscale=True``.
    """

    def __init__(self, base):
        self._base = base

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        img, label = self._base[idx]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        return img, int(label)


def raw_dataset_for(provider, split: str):
    """Return the FFCV-ready raw dataset for ``split`` (lifts to RGB if 1-channel)."""
    raw = provider.raw_datasets()[split]
    in_shape = provider.get_input_shape()
    channels = int(in_shape[0]) if len(in_shape) >= 3 else 3
    if channels == 1:
        return _AsRGB(raw)
    return raw


def infer_spec(provider) -> PipelineSpec:
    """Build a :class:`PipelineSpec` from a provider's data surface."""
    in_shape = provider.get_input_shape()
    channels = int(in_shape[0]) if len(in_shape) >= 3 else 3
    to_grayscale = (channels == 1)

    preproc = getattr(provider, "_preprocessing_spec", None)
    resize_to = _resolve_resize(preproc)
    mean_std = _resolve_normalize(preproc)
    if mean_std is None:
        # No normalization → identity.
        mean = (0.0,) if to_grayscale else (0.0, 0.0, 0.0)
        std = (1.0,) if to_grayscale else (1.0, 1.0, 1.0)
    else:
        mean, std = mean_std
    interp = preproc.interpolation if preproc is not None else "bicubic"

    image_field = FieldSpec(
        name="image", write_type="RGBImageField",
        decode_type="SimpleRGBImageDecoder",
    )
    label_field = FieldSpec(
        name="label", write_type="IntField", decode_type="IntDecoder",
    )

    # Provider declares per-split FFCV CPU op chains; we wrap each with the
    # standard ToTensor/ToDevice/ToTorchImage tail FFCV requires.
    provider_ffcv_tf = provider.ffcv_transforms()
    def _provider_image_ops(split: str) -> tuple:
        return tuple(("image", cls_name, dict(kwargs))
                     for cls_name, kwargs in provider_ffcv_tf.get(split, []))

    image_to_gpu = (
        ("image", "ToTensor", {}),
        ("image", "ToDevice", {"non_blocking": True}),
        ("image", "ToTorchImage", {}),
    )
    label_to_gpu = (
        ("label", "ToTensor", {}),
        ("label", "ToDevice", {}),
        ("label", "Squeeze", {}),
    )
    train_transforms = _provider_image_ops("train") + image_to_gpu + label_to_gpu
    val_transforms   = _provider_image_ops("val")   + image_to_gpu + label_to_gpu
    test_transforms  = _provider_image_ops("test")  + image_to_gpu + label_to_gpu

    gpu_postprocess = (
        ("GPUResizeNormalize", {
            "resize_to": resize_to,
            "interpolation": interp,
            "mean": mean,
            "std": std,
            "scale_255": True,
            "to_grayscale": to_grayscale,
        }),
    )

    return PipelineSpec(
        id=_provider_id(provider),
        fields=(image_field, label_field),
        splits={
            "train": SplitSpec(transforms=train_transforms, shuffle=True, drop_last=True),
            "val":   SplitSpec(transforms=val_transforms, shuffle=False, drop_last=False),
            "test":  SplitSpec(transforms=test_transforms, shuffle=False, drop_last=False),
        },
        gpu_postprocess=gpu_postprocess,
    )
