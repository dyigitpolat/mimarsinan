"""``infer_spec(provider)`` builds a complete PipelineSpec from the
provider's surface — no GPU postprocess synthesis, no magic tails.
"""

from __future__ import annotations

import torch


class _TinyDS(torch.utils.data.Dataset):
    def __init__(self, shape, n_classes, n=4):
        self.shape = shape; self.n_classes = n_classes; self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx): return torch.zeros(self.shape), idx % self.n_classes


class _FakeProvider:
    def __init__(self, *, input_shape=(3, 32, 32), num_classes=10,
                 ffcv_tf=None, preprocessing=None):
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._raw_map = {
            "train": _TinyDS(input_shape, num_classes),
            "val":   _TinyDS(input_shape, num_classes),
            "test":  _TinyDS(input_shape, num_classes),
        }
        self._ffcv_tf_map = ffcv_tf if ffcv_tf is not None else {
            "train": [], "val": [], "test": [],
        }
        # PreprocessingSpec from a dict-like config, mirroring DataProvider's init.
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
        return self._ffcv_tf_map

    def enable_ffcv(self) -> bool:
        return bool(self._ffcv_tf_map)


def test_infer_spec_returns_image_and_label_fields():
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec = infer_spec(_FakeProvider())
    assert {f.name for f in spec.fields} == {"image", "label"}


def test_infer_spec_id_stable_per_provider_class():
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    a = infer_spec(_FakeProvider())
    b = infer_spec(_FakeProvider())
    assert a.id == b.id


def test_infer_spec_sets_max_resolution_from_preprocessing_spec():
    """``_preprocessing_spec.resize_to`` is the canonical model-input
    contract; the FFCV layer reads it directly and sets ``max_resolution``
    on the beton ``RGBImageField`` so the on-disk image is stored at the
    size the model wants (no post-decode resize op needed)."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(preprocessing={"resize_to": 224})
    spec = infer_spec(p)
    image_field = next(f for f in spec.fields if f.name == "image")
    assert image_field.write_kwargs == {"max_resolution": 224}


def test_infer_spec_omits_max_resolution_when_no_preprocessing_resize():
    """No ``resize_to`` in preprocessing → beton stores at native size,
    no ``max_resolution`` kwarg."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(preprocessing=None)
    spec = infer_spec(p)
    image_field = next(f for f in spec.fields if f.name == "image")
    assert image_field.write_kwargs == {}


def test_infer_spec_appends_provider_chain_then_synthesized_tail():
    """The image pipeline is: provider's augments → synthesized structural
    tail (ToTensor / ToDevice / ToTorchImage). Provider declares augments
    only; tail is uniform across providers."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(ffcv_tf={
        "train": [("RandomHorizontalFlip", {}), ("Cutout", {"crop_size": 8})],
        "val":   [],
        "test":  [],
    })
    spec = infer_spec(p)
    train_image = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "image"]
    assert train_image == [
        ("RandomHorizontalFlip", {}),
        ("Cutout", {"crop_size": 8}),
        ("ToTensor", {}),
        ("ToDevice", {"non_blocking": True}),
        ("ToTorchImage", {}),
    ]
    # No augments on val → just the synthesized tail.
    val_image = [(t[1], t[2]) for t in spec.splits["val"].transforms if t[0] == "image"]
    assert val_image == [
        ("ToTensor", {}),
        ("ToDevice", {"non_blocking": True}),
        ("ToTorchImage", {}),
    ]


def test_infer_spec_synthesizes_normalize_image_from_preprocessing_spec():
    """``_preprocessing_spec.{mean,std}`` is the canonical model-input
    contract for both data paths. The FFCV layer synthesizes
    ``NormalizeImage(mean*255, std*255, np.float32)`` into the image tail,
    placed *before* ToTensor so it runs CPU-side on uint8 (FFCV's GPU
    normalize path would require cupy)."""
    import numpy as np
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(preprocessing={
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    })
    spec = infer_spec(p)
    image_ops = [t for t in spec.splits["train"].transforms if t[0] == "image"]
    cls_names = [t[1] for t in image_ops]
    assert cls_names == ["NormalizeImage", "ToTensor", "ToDevice", "ToTorchImage"]
    _, _, normalize_kwargs = image_ops[0]
    np.testing.assert_allclose(normalize_kwargs["mean"], np.array([0.485, 0.456, 0.406]) * 255.0)
    np.testing.assert_allclose(normalize_kwargs["std"],  np.array([0.229, 0.224, 0.225]) * 255.0)
    assert normalize_kwargs["type"] is np.float32


def test_infer_spec_skips_normalize_when_no_preprocessing_mean_std():
    """Without normalize in preprocessing the tail is just ToTensor /
    ToDevice / ToTorchImage."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec = infer_spec(_FakeProvider(preprocessing=None))
    image_ops = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "image"]
    assert image_ops == [
        ("ToTensor", {}),
        ("ToDevice", {"non_blocking": True}),
        ("ToTorchImage", {}),
    ]


def test_infer_spec_appends_only_the_label_tail():
    """Label tail (ToTensor/ToDevice/Squeeze) is also synthesized; FFCV
    must materialize the label field even though the loader bypasses it
    for the indexed-lookup label path."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec = infer_spec(_FakeProvider())
    label_ops = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "label"]
    assert label_ops == [
        ("ToTensor", {}),
        ("ToDevice", {}),
        ("Squeeze", {}),
    ]


def test_infer_spec_supports_distinct_per_split_chains():
    """Provider's augments differ per split; the synthesized tail is the
    same across splits."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(ffcv_tf={
        "train": [("RandomHorizontalFlip", {})],
        "val":   [],
        "test":  [("Cutout", {"crop_size": 4})],
    })
    spec = infer_spec(p)
    train = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "image"]
    test  = [(t[1], t[2]) for t in spec.splits["test"].transforms  if t[0] == "image"]
    assert train[0] == ("RandomHorizontalFlip", {})
    assert test[0]  == ("Cutout", {"crop_size": 4})
    # Both end with the same synthesized tail.
    assert train[1:] == test[1:] == [
        ("ToTensor", {}),
        ("ToDevice", {"non_blocking": True}),
        ("ToTorchImage", {}),
    ]


def test_raw_dataset_for_returns_provider_raw():
    from mimarsinan.data_handling.ffcv.spec_builder import raw_dataset_for
    p = _FakeProvider()
    assert raw_dataset_for(p, "train") is p.raw_datasets()["train"]


def test_raw_dataset_for_pre_resizes_pil_to_preprocessing_resize_to():
    """When ``_preprocessing_spec.resize_to`` is set, ``raw_dataset_for``
    wraps the dataset with a PIL resize so the beton stores at the model-
    input resolution (FFCV's ``max_resolution`` is an upper bound and
    doesn't upscale, so the writer needs to receive already-resized PIL)."""
    from mimarsinan.data_handling.ffcv.spec_builder import raw_dataset_for

    p = _FakeProvider(preprocessing={"resize_to": 224})
    # _FakeProvider's _TinyDS yields tensors, not PIL — patch one entry with
    # a tiny PIL image so we exercise the resize path end-to-end.
    from PIL import Image
    small = Image.new("RGB", (32, 32))
    p._raw_map["train"] = type("DS", (), {
        "__len__": lambda self: 1,
        "__getitem__": lambda self, idx: (small, 7),
    })()

    wrapped = raw_dataset_for(p, "train")
    img, label = wrapped[0]
    assert img.size == (224, 224)
    assert label == 7


def test_raw_dataset_for_skips_resize_when_no_preprocessing_resize_to():
    """No ``resize_to`` → no PIL resize wrap."""
    from mimarsinan.data_handling.ffcv.spec_builder import raw_dataset_for

    p = _FakeProvider(preprocessing=None)
    from PIL import Image
    small = Image.new("RGB", (32, 32))
    p._raw_map["train"] = type("DS", (), {
        "__len__": lambda self: 1,
        "__getitem__": lambda self, idx: (small, 7),
    })()

    wrapped = raw_dataset_for(p, "train")
    img, label = wrapped[0]
    assert img.size == (32, 32)


def test_as_rgb_wrapper_lifts_grayscale_pil_to_rgb():
    """``_AsRGB`` is the explicit wrapper providers can use when they have a
    grayscale source dataset. Never auto-applied by ``infer_spec``."""
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
