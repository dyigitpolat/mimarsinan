"""Tests for gui.json_util."""

import math

import numpy as np
import torch

from mimarsinan.gui.json_util import to_json_safe


class TestJsonUtil:
    def test_nested_dict(self):
        assert to_json_safe({"a": [1, {"b": 2}]}) == {"a": [1, {"b": 2}]}

    def test_tensor_scalar(self):
        assert to_json_safe(torch.tensor(3.5)) == 3.5

    def test_numpy_array(self):
        assert to_json_safe(np.array([1, 2])) == [1, 2]

    def test_nan_inf_to_none(self):
        assert to_json_safe(float("nan")) is None
        assert to_json_safe(float("inf")) is None
        assert to_json_safe(float("-inf")) is None
        assert to_json_safe({"x": math.nan, "y": [np.inf]}) == {"x": None, "y": [None]}

    def test_nested_tensor_item(self):
        assert to_json_safe({"t": torch.tensor([1.0])}) == {"t": [1.0]}
