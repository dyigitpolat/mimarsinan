"""Build a :class:`PipelineSpec` from a :class:`DataProvider`'s surface."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.data_handling.ffcv.pipeline_spec import (
    FieldSpec,
    PipelineSpec,
    SplitSpec,
)


def _provider_id(provider) -> str:
    return type(provider).__name__.lower().replace("_dataprovider", "").replace("_provider", "")


class _AsRGB(torch.utils.data.Dataset):
    """Lift a (grayscale PIL, int) dataset to 3-channel RGB.

    FFCV's stock ``RGBImageField`` requires 3 channels at write time.
    Applied automatically when the provider's ``get_input_shape()``
    reports 1 channel.
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


_PIL_INTERP = {"bicubic": 3, "bilinear": 2, "nearest": 0}  # PIL.Image constants


class _PILResize(torch.utils.data.Dataset):
    """Resize the PIL image half of each ``(image, label)`` to a fixed square.

    FFCV's ``RGBImageField.max_resolution`` is an *upper bound* — it
    downscales images larger than the target but doesn't upscale. To get
    the beton stored at the size the decoder expects, the writer needs
    to receive PIL images already at that size.
    """

    def __init__(self, base, target_size: int, interpolation: str):
        self._base = base
        self._size = int(target_size)
        self._interp = _PIL_INTERP.get((interpolation or "bicubic").lower(), 3)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        img, label = self._base[idx]
        if hasattr(img, "resize"):
            img = img.resize((self._size, self._size), self._interp)
        return img, label


def _ffcv_config(provider) -> dict:
    """Return the provider's FFCV declaration, validated.

    Expected shape::

        {
            "beton_image_size": int | None,    # optional
            "splits": {"train": [...], "val": [...], "test": [...]},
        }

    where each split is a list of ``(ffcv_op_class_name, kwargs)`` tuples
    starting with the per-split decoder. Empty / missing return value
    means the provider opted out.
    """
    cfg = provider.ffcv_transforms()
    if not cfg:
        return {}
    if "splits" not in cfg:
        raise ValueError(
            f"{type(provider).__name__}.ffcv_transforms() must return a dict "
            f"with a 'splits' key (got keys: {sorted(cfg.keys())})"
        )
    splits = cfg["splits"]
    for split in ("train", "val", "test"):
        if split not in splits:
            raise ValueError(
                f"{type(provider).__name__}.ffcv_transforms()['splits'] missing "
                f"required key '{split}' (got keys: {sorted(splits.keys())})"
            )
    return cfg


def _beton_image_size(provider) -> int | None:
    """Resolve the beton storage size: provider's FFCV declaration, else preprocessing_spec."""
    cfg = _ffcv_config(provider)
    if cfg and "beton_image_size" in cfg and cfg["beton_image_size"] is not None:
        return int(cfg["beton_image_size"])
    preproc = getattr(provider, "_preprocessing_spec", None)
    if preproc is not None and preproc.resize_to is not None:
        return int(preproc.resize_to)
    return None


def raw_dataset_for(provider, split: str):
    """Return the raw dataset that will feed the FFCV writer for ``split``.

    Applies two structural wraps:

    * :class:`_AsRGB` if ``get_input_shape()`` says 1 channel.
    * :class:`_PILResize` to ``beton_image_size`` (from
      ``ffcv_transforms()['beton_image_size']`` if set, else
      ``_preprocessing_spec.resize_to``) so the beton stores at the size
      the decoder expects.
    """
    raw = provider.raw_datasets()[split]
    in_shape = provider.get_input_shape()
    channels = int(in_shape[0]) if len(in_shape) >= 3 else 3
    if channels == 1:
        raw = _AsRGB(raw)
    beton_size = _beton_image_size(provider)
    if beton_size is not None:
        preproc = getattr(provider, "_preprocessing_spec", None)
        interp = (preproc.interpolation if preproc is not None else None) or "bicubic"
        raw = _PILResize(raw, beton_size, interp)
    return raw


def infer_spec(provider) -> PipelineSpec:
    """Build a :class:`PipelineSpec` from a provider's data surface.

    Provider opts into FFCV by returning a structured ``ffcv_transforms()``
    config with per-split FFCV op chains (decoder is the first op). The
    structural tail — ``NormalizeImage`` (from
    ``_preprocessing_spec.{mean,std}``) and ``ToTensor → ToDevice →
    ToTorchImage`` — is synthesized here so both data paths consume the
    same model-input contract uniformly.
    """
    preproc = getattr(provider, "_preprocessing_spec", None)
    cfg = _ffcv_config(provider)
    provider_splits = cfg.get("splits", {})

    image_write_kwargs = {}
    beton_size = _beton_image_size(provider)
    if beton_size is not None:
        image_write_kwargs["max_resolution"] = int(beton_size)
    image_field = FieldSpec(
        name="image",
        write_type="RGBImageField",
        write_kwargs=image_write_kwargs,
    )
    label_field = FieldSpec(name="label", write_type="IntField")

    def _image_ops(split: str) -> tuple:
        return tuple(("image", cls_name, dict(kwargs))
                     for cls_name, kwargs in provider_splits.get(split, []))

    # Structural image tail: NormalizeImage (from preprocessing) then the
    # standard FFCV plumbing. Normalize runs CPU-side (before ToDevice)
    # so we don't pull cupy in as a dependency for the GPU path.
    image_tail_ops = []
    if preproc is not None and preproc.mean is not None and preproc.std is not None:
        mean_255 = np.asarray(preproc.mean, dtype=np.float32) * 255.0
        std_255  = np.asarray(preproc.std,  dtype=np.float32) * 255.0
        image_tail_ops.append(("NormalizeImage", {"mean": mean_255, "std": std_255, "type": np.float32}))
    image_tail_ops.extend([
        ("ToTensor", {}),
        ("ToDevice", {"non_blocking": True}),
        ("ToTorchImage", {}),
    ])
    image_tail = tuple(("image", cls, kw) for cls, kw in image_tail_ops)

    # Labels are bypassed at consume time (see ffcv/loader.py) but FFCV's
    # pipeline still has to materialize the label field — IntDecoder is
    # the first label op, then the standard int-label conversion tail.
    label_tail = (
        ("label", "IntDecoder", {}),
        ("label", "ToTensor", {}),
        ("label", "ToDevice", {}),
        ("label", "Squeeze", {}),
    )

    train_transforms = _image_ops("train") + image_tail + label_tail
    val_transforms   = _image_ops("val")   + image_tail + label_tail
    test_transforms  = _image_ops("test")  + image_tail + label_tail

    return PipelineSpec(
        id=_provider_id(provider),
        fields=(image_field, label_field),
        splits={
            "train": SplitSpec(transforms=train_transforms, shuffle=True, drop_last=True),
            "val":   SplitSpec(transforms=val_transforms, shuffle=False, drop_last=False),
            "test":  SplitSpec(transforms=test_transforms, shuffle=False, drop_last=False),
        },
    )
