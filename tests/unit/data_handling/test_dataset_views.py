"""Tests for the provider data-surface contract: ``raw_datasets`` +
``torch_transforms`` method overrides.

The base class assembles each split's torch dataset by composing the raw,
transform-free dataset (from ``raw_datasets()``) with the matching transform
(from ``torch_transforms()``). This decoupling is what lets the FFCV layer
reach in for the raw dataset (no transforms baked in, required for beton
writing + label preload) without each provider duplicating split-construction
logic.
"""
from __future__ import annotations

import torch


def test_apply_transform_wraps_and_invokes():
    from mimarsinan.data_handling.dataset_views import ApplyTransform

    class _Raw(torch.utils.data.Dataset):
        def __init__(self):
            self.items = [(torch.tensor([1.0]), 0), (torch.tensor([2.0]), 1)]
        def __len__(self): return len(self.items)
        def __getitem__(self, idx): return self.items[idx]

    raw = _Raw()
    view = ApplyTransform(raw, lambda x: x * 10)
    assert len(view) == 2
    x0, y0 = view[0]
    assert x0.item() == 10.0 and y0 == 0
    x1, y1 = view[1]
    assert x1.item() == 20.0 and y1 == 1


def test_apply_transform_passes_label_through():
    from mimarsinan.data_handling.dataset_views import ApplyTransform

    class _Raw(torch.utils.data.Dataset):
        def __len__(self): return 1
        def __getitem__(self, _): return torch.tensor([5.0]), 42

    view = ApplyTransform(_Raw(), lambda x: x + 1)
    x, y = view[0]
    assert x.item() == 6.0
    assert y == 42


def test_base_class_assembles_loaders_from_overridden_methods():
    """A subclass that overrides ``raw_datasets()`` + ``torch_transforms()``
    gets its per-split torch datasets assembled by the base class via
    ``ApplyTransform`` — no per-provider plumbing needed."""
    from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode

    class _ToyProvider(DataProvider):
        def __init__(self):
            super().__init__(datasets_path="", seed=0)
            items = [(torch.zeros(1), i) for i in range(6)]

            class _Raw(torch.utils.data.Dataset):
                def __init__(self, items): self.items = items
                def __len__(self): return len(self.items)
                def __getitem__(self, idx): return self.items[idx]

            self._raws = {
                "train": _Raw(items[:4]),
                "val":   _Raw(items[4:5]),
                "test":  _Raw(items[5:]),
            }

        def raw_datasets(self):
            return self._raws

        def torch_transforms(self):
            # Provider returns raw lists; the base class wraps them with
            # preprocessing via ``_wrap_with_preprocessing``. A lambda
            # masquerading as a torchvision transform is fine for this test
            # — ``Compose`` accepts any callable.
            return {
                "train": [lambda x: x + 1.0],
                "val":   [lambda x: x],
                "test":  [lambda x: x],
            }

        def get_prediction_mode(self): return ClassificationMode(2)

    p = _ToyProvider()
    train = p._get_training_dataset()
    val = p._get_validation_dataset()
    test = p._get_test_dataset()
    assert len(train) == 4 and len(val) == 1 and len(test) == 1

    # Transform was applied on train (added 1.0); val/test untouched.
    assert train[0][0].item() == 1.0
    assert val[0][0].item() == 0.0
    assert test[0][0].item() == 0.0
