"""``infer_spec(provider)`` builds a complete PipelineSpec from the
provider's surface — provider declares per-split FFCV op chains
(decoder first), spec_builder synthesizes the structural normalize+tail.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ffcv", reason="optional ffcv backend not installed")


class _TinyDS(torch.utils.data.Dataset):
    def __init__(self, shape, n_classes, n=4):
        self.shape = shape; self.n_classes = n_classes; self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx): return torch.zeros(self.shape), idx % self.n_classes


def _default_ffcv_cfg():
    """Minimal valid FFCV config: decoder per split, no augments."""
    return {
        "splits": {
            "train": [("SimpleRGBImageDecoder", {})],
            "val":   [("SimpleRGBImageDecoder", {})],
            "test":  [("SimpleRGBImageDecoder", {})],
        },
    }


class _FakeProvider:
    def __init__(self, *, input_shape=(3, 32, 32), num_classes=10,
                 ffcv_cfg=None, preprocessing=None):
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._raw_map = {
            "train": _TinyDS(input_shape, num_classes),
            "val":   _TinyDS(input_shape, num_classes),
            "test":  _TinyDS(input_shape, num_classes),
        }
        self._ffcv_cfg = ffcv_cfg if ffcv_cfg is not None else _default_ffcv_cfg()
        from mimarsinan.data_handling.preprocessing import resolve_preprocessing
        self._preprocessing_spec = resolve_preprocessing(preprocessing)

    def get_input_shape(self):
        return self._input_shape

    def get_prediction_mode(self):
        from mimarsinan.data_handling.data_provider import ClassificationMode
        return ClassificationMode(self._num_classes)

    def raw_datasets(self):
        return self._raw_map

    def ffcv_transforms(self):
        return self._ffcv_cfg

    def enable_ffcv(self) -> bool:
        return bool(self._ffcv_cfg)


def test_infer_spec_returns_image_and_label_fields():
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec = infer_spec(_FakeProvider())
    assert {f.name for f in spec.fields} == {"image", "label"}


def test_infer_spec_id_stable_per_provider_class():
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    a = infer_spec(_FakeProvider())
    b = infer_spec(_FakeProvider())
    assert a.id == b.id


def test_infer_spec_max_resolution_from_explicit_beton_image_size():
    """``beton_image_size`` in the FFCV config wins over preprocessing.resize_to."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    cfg = _default_ffcv_cfg()
    cfg["beton_image_size"] = 256
    p = _FakeProvider(ffcv_cfg=cfg, preprocessing={"resize_to": 224})
    spec = infer_spec(p)
    image_field = next(f for f in spec.fields if f.name == "image")
    assert image_field.write_kwargs == {"max_resolution": 256}


def test_infer_spec_max_resolution_falls_back_to_preprocessing_resize_to():
    """When the FFCV config doesn't set ``beton_image_size``, the model-side
    ``_preprocessing_spec.resize_to`` is used as the default."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(preprocessing={"resize_to": 224})
    spec = infer_spec(p)
    image_field = next(f for f in spec.fields if f.name == "image")
    assert image_field.write_kwargs == {"max_resolution": 224}


def test_infer_spec_omits_max_resolution_when_neither_source_sets_it():
    """No preprocessing.resize_to + no FFCV beton_image_size → no max_resolution."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec = infer_spec(_FakeProvider(preprocessing=None))
    image_field = next(f for f in spec.fields if f.name == "image")
    assert image_field.write_kwargs == {}


def test_infer_spec_image_chain_is_provider_chain_then_synthesized_tail():
    """The image pipeline is: provider's per-split chain (decoder first +
    augments) → synthesized structural tail (ToTensor / ToDevice /
    ToTorchImage). Provider's chain lands verbatim."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    cfg = {
        "splits": {
            "train": [
                ("SimpleRGBImageDecoder", {}),
                ("RandomHorizontalFlip", {}),
                ("Cutout", {"crop_size": 8}),
            ],
            "val":  [("SimpleRGBImageDecoder", {})],
            "test": [("SimpleRGBImageDecoder", {})],
        },
    }
    p = _FakeProvider(ffcv_cfg=cfg)
    spec = infer_spec(p)
    train_image = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "image"]
    assert train_image == [
        ("SimpleRGBImageDecoder", {}),
        ("RandomHorizontalFlip", {}),
        ("Cutout", {"crop_size": 8}),
        ("ToTensor", {}),
        ("ToDevice", {"non_blocking": True}),
        ("ToTorchImage", {}),
    ]


def test_infer_spec_synthesizes_normalize_image_from_preprocessing_spec():
    """``_preprocessing_spec.{mean,std}`` synthesized into ``NormalizeImage``
    on the image tail, placed CPU-side (before ToTensor) so we don't pull
    cupy in for the GPU normalize path."""
    import numpy as np
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(preprocessing={
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    })
    spec = infer_spec(p)
    image_ops = [t for t in spec.splits["train"].transforms if t[0] == "image"]
    cls_names = [t[1] for t in image_ops]
    # Provider's SimpleRGBImageDecoder first, then synthesized tail.
    assert cls_names == ["SimpleRGBImageDecoder", "NormalizeImage", "ToTensor", "ToDevice", "ToTorchImage"]
    _, _, normalize_kwargs = image_ops[1]
    np.testing.assert_allclose(normalize_kwargs["mean"], np.array([0.485, 0.456, 0.406]) * 255.0)
    np.testing.assert_allclose(normalize_kwargs["std"],  np.array([0.229, 0.224, 0.225]) * 255.0)
    assert normalize_kwargs["type"] is np.float32


def test_infer_spec_skips_normalize_when_no_preprocessing_mean_std():
    """No normalize in preprocessing → tail omits NormalizeImage."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec = infer_spec(_FakeProvider(preprocessing=None))
    image_ops = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "image"]
    assert image_ops == [
        ("SimpleRGBImageDecoder", {}),
        ("ToTensor", {}),
        ("ToDevice", {"non_blocking": True}),
        ("ToTorchImage", {}),
    ]


def test_infer_spec_supports_per_split_decoder():
    """Each split's decoder is the first op in that split's provider
    chain — independent across splits."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    cfg = {
        "beton_image_size": 256,
        "splits": {
            "train": [("RandomResizedCropRGBImageDecoder", {"output_size": (224, 224)})],
            "val":   [("CenterCropRGBImageDecoder", {"output_size": (224, 224), "ratio": 0.875})],
            "test":  [("CenterCropRGBImageDecoder", {"output_size": (224, 224), "ratio": 0.875})],
        },
    }
    p = _FakeProvider(ffcv_cfg=cfg)
    spec = infer_spec(p)
    train_first = next(t for t in spec.splits["train"].transforms if t[0] == "image")
    val_first   = next(t for t in spec.splits["val"].transforms   if t[0] == "image")
    assert train_first[1] == "RandomResizedCropRGBImageDecoder"
    assert val_first[1]   == "CenterCropRGBImageDecoder"


def test_infer_spec_synthesizes_label_int_decoder():
    """Label tail (IntDecoder + ToTensor + ToDevice + Squeeze) is
    synthesized; provider doesn't declare it."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec = infer_spec(_FakeProvider())
    label_ops = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "label"]
    assert label_ops == [
        ("IntDecoder", {}),
        ("ToTensor", {}),
        ("ToDevice", {}),
        ("Squeeze", {}),
    ]


def test_infer_spec_rejects_ffcv_config_without_splits_key():
    """Hard-fail on malformed FFCV config (no silent default)."""
    import pytest
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(ffcv_cfg={"beton_image_size": 224})  # missing "splits"
    with pytest.raises(ValueError, match="'splits' key"):
        infer_spec(p)


def test_infer_spec_rejects_ffcv_config_missing_per_split_entries():
    """All three splits required when ffcv_transforms() is set."""
    import pytest
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(ffcv_cfg={"splits": {"train": [("SimpleRGBImageDecoder", {})]}})
    with pytest.raises(ValueError, match="missing required key"):
        infer_spec(p)


def test_raw_dataset_for_uses_ffcv_beton_size_when_set():
    """Explicit ``beton_image_size`` overrides preprocessing.resize_to for
    the PIL pre-resize wrap."""
    from mimarsinan.data_handling.ffcv.spec_builder import raw_dataset_for
    from PIL import Image
    cfg = _default_ffcv_cfg()
    cfg["beton_image_size"] = 256
    p = _FakeProvider(ffcv_cfg=cfg, preprocessing={"resize_to": 224})
    p._raw_map["train"] = type("DS", (), {
        "__len__": lambda self: 1,
        "__getitem__": lambda self, idx: (Image.new("RGB", (300, 300)), 7),
    })()
    wrapped = raw_dataset_for(p, "train")
    img, label = wrapped[0]
    assert img.size == (256, 256)
    assert label == 7


def test_raw_dataset_for_falls_back_to_preprocessing_resize_to():
    """When the FFCV config doesn't set beton_image_size, the wrap uses
    preprocessing.resize_to as the default."""
    from mimarsinan.data_handling.ffcv.spec_builder import raw_dataset_for
    from PIL import Image
    p = _FakeProvider(preprocessing={"resize_to": 224})
    p._raw_map["train"] = type("DS", (), {
        "__len__": lambda self: 1,
        "__getitem__": lambda self, idx: (Image.new("RGB", (32, 32)), 7),
    })()
    wrapped = raw_dataset_for(p, "train")
    img, _ = wrapped[0]
    assert img.size == (224, 224)


def test_raw_dataset_for_skips_resize_when_no_source_sets_size():
    from mimarsinan.data_handling.ffcv.spec_builder import raw_dataset_for
    from PIL import Image
    p = _FakeProvider(preprocessing=None)
    p._raw_map["train"] = type("DS", (), {
        "__len__": lambda self: 1,
        "__getitem__": lambda self, idx: (Image.new("RGB", (32, 32)), 7),
    })()
    wrapped = raw_dataset_for(p, "train")
    img, _ = wrapped[0]
    assert img.size == (32, 32)


def test_as_rgb_wrapper_lifts_grayscale_pil_to_rgb():
    from mimarsinan.data_handling.ffcv.spec_builder import _AsRGB

    class _FakePIL:
        def __init__(self, mode): self.mode = mode
        def convert(self, mode):
            return _FakePIL(mode)

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, idx):
            return _FakePIL("L"), idx

    wrapped = _AsRGB(_DS())
    img, label = wrapped[0]
    assert img.mode == "RGB"
    assert label == 0
