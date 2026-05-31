"""Tests for ``infer_spec(provider)`` — generic FFCV spec inference.

The provider declares its FFCV opt-in and augment chain explicitly
(``enable_ffcv()`` / ``ffcv_train_augments()``); ``infer_spec`` walks
``_raw`` / ``_preprocessing_spec`` / ``get_input_shape`` /
``get_prediction_mode`` and produces a complete :class:`PipelineSpec`.
"""
from __future__ import annotations

import torch


# Provider lookalike for tests — just the attributes infer_spec reads.
class _FakeProvider:
    def __init__(self, *, input_shape=(3, 32, 32), num_classes=10,
                 preprocessing=None, ffcv_tf=None):
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._preprocessing_spec = preprocessing  # may be a PreprocessingSpec or None
        self._raw_map = {
            "train": _TinyDS(input_shape, num_classes),
            "val": _TinyDS(input_shape, num_classes),
            "test": _TinyDS(input_shape, num_classes),
        }
        # Default: empty per-split FFCV op chains (= opted in to FFCV, just
        # no augments). Override via ``ffcv_tf`` kwarg.
        self._ffcv_tf_map = ffcv_tf if ffcv_tf is not None else {
            "train": [], "val": [], "test": [],
        }

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


class _TinyDS(torch.utils.data.Dataset):
    def __init__(self, shape, n_classes, n=4):
        self.shape = shape; self.n_classes = n_classes; self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx): return torch.zeros(self.shape), idx % self.n_classes


def test_infer_spec_returns_pipeline_spec_with_image_and_label_fields():
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec = infer_spec(_FakeProvider())
    assert {f.name for f in spec.fields} == {"image", "label"}


def test_infer_spec_id_is_stable_and_derived_from_provider():
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec_a = infer_spec(_FakeProvider())
    spec_b = infer_spec(_FakeProvider())
    assert spec_a.id == spec_b.id  # same shape/classes → same id


def test_infer_spec_uses_provider_declared_per_split_ffcv_chain():
    """Each split's image ops come from the corresponding ``_ffcv_tf[split]``
    entry; the standard ToTensor/ToDevice/ToTorchImage tail is appended."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(ffcv_tf={
        "train": [("RandomHorizontalFlip", {}), ("Cutout", {"crop_size": 8})],
        "val":   [],
        "test":  [],
    })
    spec = infer_spec(p)
    train_ops = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "image"]
    val_ops   = [(t[1], t[2]) for t in spec.splits["val"].transforms   if t[0] == "image"]
    test_ops  = [(t[1], t[2]) for t in spec.splits["test"].transforms  if t[0] == "image"]
    # Provider's train chain leads, then the standard tail
    assert train_ops[0] == ("RandomHorizontalFlip", {})
    assert train_ops[1] == ("Cutout", {"crop_size": 8})
    # val/test have empty provider chain → identical tails
    assert val_ops == test_ops
    assert ("RandomHorizontalFlip", {}) not in val_ops


def test_infer_spec_empty_chains_make_train_match_val_match_test():
    """No augment declared anywhere → all three splits share the same image
    pipeline (just decode + ship to GPU)."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(ffcv_tf={"train": [], "val": [], "test": []})
    spec = infer_spec(p)
    train_img = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "image"]
    val_img   = [(t[1], t[2]) for t in spec.splits["val"].transforms   if t[0] == "image"]
    test_img  = [(t[1], t[2]) for t in spec.splits["test"].transforms  if t[0] == "image"]
    assert train_img == val_img == test_img


def test_infer_spec_supports_distinct_per_split_chains():
    """val/test may legitimately differ from train (in degenerate test setups);
    the inferred spec reflects whatever the provider declared per split."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(ffcv_tf={
        "train": [("RandomHorizontalFlip", {})],
        "val":   [],
        "test":  [("Cutout", {"crop_size": 4})],
    })
    spec = infer_spec(p)
    train_ops = [(t[1], t[2]) for t in spec.splits["train"].transforms if t[0] == "image"]
    test_ops  = [(t[1], t[2]) for t in spec.splits["test"].transforms  if t[0] == "image"]
    assert train_ops[0] == ("RandomHorizontalFlip", {})
    assert test_ops[0] == ("Cutout", {"crop_size": 4})


def test_infer_spec_uses_composite_resize_normalize():
    """``gpu_postprocess`` is the single ``GPUResizeNormalize`` composite —
    cheaper than the split Resize / Normalize pair if there's no augment to
    slot between them."""
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    spec = infer_spec(_FakeProvider())
    op_names = [op[0] for op in spec.gpu_postprocess]
    assert op_names == ["GPUResizeNormalize"]


def test_infer_spec_grayscale_for_one_channel_input():
    from mimarsinan.data_handling.ffcv.spec_builder import infer_spec
    p = _FakeProvider(input_shape=(1, 28, 28), num_classes=10)
    spec = infer_spec(p)
    # GPU postprocess must include a to_grayscale=True step so the 3-channel
    # FFCV beton is collapsed back to 1 channel for the model.
    found = False
    for cls_name, kwargs in spec.gpu_postprocess:
        if cls_name in ("GPUResizeNormalize", "GPUResize") and kwargs.get("to_grayscale") is True:
            found = True
            break
    assert found, f"expected to_grayscale=True somewhere in gpu_postprocess; got {spec.gpu_postprocess}"


def test_raw_dataset_factory_passes_through_for_rgb():
    from mimarsinan.data_handling.ffcv.spec_builder import raw_dataset_for
    p = _FakeProvider(input_shape=(3, 32, 32))
    ds = raw_dataset_for(p, "train")
    assert ds is p.raw_datasets()["train"]  # no wrapping needed


def test_raw_dataset_factory_lifts_to_rgb_for_grayscale():
    """1-channel datasets get wrapped so PIL ``.convert('RGB')`` runs at write time."""
    from mimarsinan.data_handling.ffcv.spec_builder import raw_dataset_for
    p = _FakeProvider(input_shape=(1, 28, 28))
    ds = raw_dataset_for(p, "train")
    assert ds is not p.raw_datasets()["train"]  # wrapped
    assert hasattr(ds, "__len__") and hasattr(ds, "__getitem__")
