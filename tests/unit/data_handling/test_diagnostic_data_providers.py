"""Metadata-only tests for the cheap diagnostic providers (FashionMNIST, KMNIST, SVHN).

Each provider is constructed with torchvision's dataset class mocked out so the
test never touches the network; only declared #classes and input shape are
checked. The real torchvision download happens at run time, not in the test.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from mimarsinan.data_handling.data_provider import ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

from mimarsinan.data_handling.data_providers.fashion_mnist_data_provider import (
    FashionMNIST_DataProvider,
)
from mimarsinan.data_handling.data_providers.kmnist_data_provider import (
    KMNIST_DataProvider,
)
from mimarsinan.data_handling.data_providers.svhn_data_provider import (
    SVHN_DataProvider,
)


def _grayscale_dataset(size=20):
    """A no-network stand-in for a 1x28x28 torchvision dataset."""
    ds = MagicMock()
    ds.__len__ = lambda self: size

    def _getitem(_self, idx):
        return Image.new("L", (28, 28)), int(idx) % 10

    ds.__getitem__ = _getitem
    return ds


def _rgb_dataset(size=20):
    """A no-network stand-in for a 3x32x32 torchvision dataset."""
    ds = MagicMock()
    ds.__len__ = lambda self: size

    def _getitem(_self, idx):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB"), int(idx) % 10

    ds.__getitem__ = _getitem
    return ds


class TestFashionMNISTDataProvider:
    def test_metadata(self):
        target = (
            "mimarsinan.data_handling.data_providers."
            "fashion_mnist_data_provider.torchvision.datasets.FashionMNIST"
        )
        with patch(target, return_value=_grayscale_dataset()):
            dp = FashionMNIST_DataProvider("/tmp/does_not_matter")
            assert isinstance(dp.get_prediction_mode(), ClassificationMode)
            assert dp.get_output_shape() == 10
            assert tuple(dp.get_input_shape()) == (1, 28, 28)

    def test_registered(self):
        assert "FashionMNIST_DataProvider" in BasicDataProviderFactory._provider_registry


class TestKMNISTDataProvider:
    def test_metadata(self):
        target = (
            "mimarsinan.data_handling.data_providers."
            "kmnist_data_provider.torchvision.datasets.KMNIST"
        )
        with patch(target, return_value=_grayscale_dataset()):
            dp = KMNIST_DataProvider("/tmp/does_not_matter")
            assert isinstance(dp.get_prediction_mode(), ClassificationMode)
            assert dp.get_output_shape() == 10
            assert tuple(dp.get_input_shape()) == (1, 28, 28)

    def test_registered(self):
        assert "KMNIST_DataProvider" in BasicDataProviderFactory._provider_registry


class TestSVHNDataProvider:
    def test_metadata(self):
        target = (
            "mimarsinan.data_handling.data_providers."
            "svhn_data_provider.torchvision.datasets.SVHN"
        )
        with patch(target, return_value=_rgb_dataset()):
            dp = SVHN_DataProvider("/tmp/does_not_matter")
            assert isinstance(dp.get_prediction_mode(), ClassificationMode)
            assert dp.get_output_shape() == 10
            assert tuple(dp.get_input_shape()) == (3, 32, 32)

    def test_registered(self):
        assert "SVHN_DataProvider" in BasicDataProviderFactory._provider_registry


@pytest.mark.parametrize(
    "name",
    ["FashionMNIST_DataProvider", "KMNIST_DataProvider", "SVHN_DataProvider"],
)
def test_appears_in_list_registered(name):
    import mimarsinan.data_handling.data_providers  # noqa: F401 — populate registry

    ids = [e["id"] for e in BasicDataProviderFactory.list_registered()]
    assert name in ids
