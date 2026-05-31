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
                 ffcv_tf=None, field_kwargs=None):
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
        self._field_kwargs = field_kwargs or {}

    def get_input_shape(self):
        return self._input_shape

    def get_prediction_mode(self):
        from mimarsinan.data_handling.data_provider import ClassificationMode
        return ClassificationMode(self._num_classes)

    def raw_datasets(self):
        return self._raw_map

    def ffcv_transforms(self):
        return self._ffcv_tf_map

    def ffcv_image_field_kwargs(self):
        return self._field_kwargs

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


def test_infer_spec_forwards_field_kwargs_to_image_field():
    """``ffcv_image_field_kwargs()`` lands verbatim on the ``image`` FieldSpec —
    the only way the provider controls beton-write parameters like
    ``max_resolution``."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(field_kwargs={"max_resolution": 224})
    spec = infer_spec(p)
    image_field = next(f for f in spec.fields if f.name == "image")
    assert image_field.write_kwargs == {"max_resolution": 224}


def test_infer_spec_image_chain_is_exactly_what_provider_declared():
    """No magic tail synthesis: the image op chain is exactly the provider's
    ``ffcv_transforms()[split]`` content, in declared order."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    provider_chain = [
        ("RandomHorizontalFlip", {}),
        ("ToTensor", {}),
        ("ToDevice", {"non_blocking": True}),
        ("ToTorchImage", {}),
        ("NormalizeImage", {"mean": [0.5], "std": [0.5], "type": "float32"}),
    ]
    p = _FakeProvider(ffcv_tf={
        "train": provider_chain, "val": provider_chain[1:], "test": provider_chain[1:],
    })
    spec = infer_spec(p)
    train_image_ops = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "image"]
    assert train_image_ops == [(c, k) for (c, k) in provider_chain]


def test_infer_spec_appends_only_the_label_tail():
    """Label tail (ToTensor/ToDevice/Squeeze) is the one piece the spec
    builder owns; FFCV must materialize the label field even though the
    loader bypasses it for the indexed-lookup label path."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec = infer_spec(_FakeProvider())
    label_ops = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "label"]
    assert label_ops == [
        ("ToTensor", {}),
        ("ToDevice", {}),
        ("Squeeze", {}),
    ]


def test_infer_spec_supports_distinct_per_split_chains():
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(ffcv_tf={
        "train": [("RandomHorizontalFlip", {})],
        "val":   [],
        "test":  [("Cutout", {"crop_size": 4})],
    })
    spec = infer_spec(p)
    train = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "image"]
    test  = [(t[1], t[2]) for t in spec.splits["test"].transforms  if t[0] == "image"]
    assert train == [("RandomHorizontalFlip", {})]
    assert test == [("Cutout", {"crop_size": 4})]


def test_raw_dataset_for_returns_provider_raw():
    from mimarsinan.data_handling.ffcv.spec_builder import raw_dataset_for
    p = _FakeProvider()
    assert raw_dataset_for(p, "train") is p.raw_datasets()["train"]


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
