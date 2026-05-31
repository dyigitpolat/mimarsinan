"""Build a :class:`PipelineSpec` from a :class:`DataProvider`'s surface."""

from __future__ import annotations

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
    Provider authors that want FFCV on a grayscale dataset wrap with this
    explicitly; the FFCV layer never auto-synthesizes the lift.
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
    """Return the raw dataset that will feed the FFCV writer for ``split``."""
    return provider.raw_datasets()[split]


def infer_spec(provider) -> PipelineSpec:
    """Build a :class:`PipelineSpec` from a provider's data surface.

    Reads two provider overrides — ``raw_datasets()`` (beton source) and
    ``ffcv_transforms()`` (per-split FFCV op chains) — plus the
    model-input contract from ``_preprocessing_spec`` (``resize_to`` →
    ``RGBImageField.max_resolution`` so the beton stores at the model's
    expected input size; no post-decode resize op needed). The provider's
    op chain lands verbatim on each split's image pipeline.
    """
    preproc = getattr(provider, "_preprocessing_spec", None)
    image_write_kwargs = {}
    if preproc is not None and preproc.resize_to is not None:
        image_write_kwargs["max_resolution"] = int(preproc.resize_to)
    image_field = FieldSpec(
        name="image",
        write_type="RGBImageField",
        write_kwargs=image_write_kwargs,
        decode_type="SimpleRGBImageDecoder",
    )
    label_field = FieldSpec(
        name="label", write_type="IntField", decode_type="IntDecoder",
    )

    provider_ffcv_tf = provider.ffcv_transforms()

    def _image_ops(split: str) -> tuple:
        return tuple(("image", cls_name, dict(kwargs))
                     for cls_name, kwargs in provider_ffcv_tf.get(split, []))

    # Labels are bypassed at consume time (see ffcv/loader.py) but FFCV's
    # pipeline still has to materialize the label field, so the standard
    # int-label tail is the one piece the spec builder owns end-to-end.
    label_tail = (
        ("label", "ToTensor", {}),
        ("label", "ToDevice", {}),
        ("label", "Squeeze", {}),
    )

    train_transforms = _image_ops("train") + label_tail
    val_transforms   = _image_ops("val")   + label_tail
    test_transforms  = _image_ops("test")  + label_tail

    return PipelineSpec(
        id=_provider_id(provider),
        fields=(image_field, label_field),
        splits={
            "train": SplitSpec(transforms=train_transforms, shuffle=True, drop_last=True),
            "val":   SplitSpec(transforms=val_transforms, shuffle=False, drop_last=False),
            "test":  SplitSpec(transforms=test_transforms, shuffle=False, drop_last=False),
        },
    )
