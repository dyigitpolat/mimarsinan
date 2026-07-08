"""Providers register their post-transform input value range (the D1 fix).

The framework never assumes [0,1]: providers declare their raw range (a
ToTensor fact) and the preprocessing spec transforms it; normalized datasets
therefore stop silently feeding unit-scale boundary math.
"""

import pytest

from mimarsinan.data_handling.data_provider import DataProvider
from mimarsinan.data_handling.data_providers.cifar10_data_provider import (
    CIFAR10_DataProvider,
)
from mimarsinan.data_handling.data_providers.mnist_data_provider import (
    MNIST_DataProvider,
)
from mimarsinan.data_handling.preprocessing import PreprocessingSpec


class TestTransformValueRange:
    def test_no_normalization_passes_the_raw_range_through(self):
        spec = PreprocessingSpec(resize_to=224)
        assert spec.transform_value_range((0.0, 1.0)) == (0.0, 1.0)

    def test_normalization_produces_the_per_channel_envelope(self):
        spec = PreprocessingSpec(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        lo, hi = spec.transform_value_range((0.0, 1.0))
        assert lo == pytest.approx(min((0 - 0.485) / 0.229, (0 - 0.456) / 0.224, (0 - 0.406) / 0.225))
        assert hi == pytest.approx(max((1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225))


class TestProviderInputValueRange:
    def _provider(self, provider_cls, preprocessing=None):
        provider = object.__new__(provider_cls)
        DataProvider.__init__(provider, "<test>", preprocessing=preprocessing)
        return provider

    def test_base_provider_makes_no_range_claim(self):
        provider = DataProvider("<test>")
        assert provider.workload_profile().input_value_range is None

    def test_totensor_provider_declares_unit_range(self):
        provider = self._provider(MNIST_DataProvider)
        profile = provider.workload_profile()
        assert profile.input_value_range == (0.0, 1.0)
        assert profile.config_updates()["input_data_scale"] == 1.0

    def test_normalized_provider_declares_the_transformed_range(self):
        provider = self._provider(
            CIFAR10_DataProvider, preprocessing={"normalize": "imagenet"}
        )
        profile = provider.workload_profile()
        assert profile.input_value_range is not None
        lo, hi = profile.input_value_range
        assert hi == pytest.approx((1 - 0.406) / 0.225)
        assert lo == pytest.approx((0 - 0.485) / 0.229)
        assert profile.config_updates()["input_data_scale"] == pytest.approx(hi)
