"""The one table of absorbable activations: the analyzer's fold plan and both weight-layer mixins read it."""

from __future__ import annotations

import pytest
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import ACTIVATION_REGISTRY
from mimarsinan.torch_mapping.converter_handlers.conv_mixin import ConvConvertMixin
from mimarsinan.torch_mapping.converter_handlers.linear_mixin import LinearConvertMixin
from mimarsinan.torch_mapping.representability_analyzer import (
    ABSORBABLE_ACTIVATIONS,
    absorbed_activation_name,
)


class _ReLUSubclass(nn.ReLU):
    pass


@pytest.mark.parametrize("act,name", [
    (nn.ReLU(), "ReLU"),
    (nn.LeakyReLU(0.2), "LeakyReLU"),
    (nn.GELU(), "GELU"),
    (_ReLUSubclass(), "ReLU"),
    (nn.SiLU(), None),
    (nn.ReLU6(), None),
    (nn.Identity(), None),
    (None, None),
])
def test_absorbed_activation_name(act, name):
    assert absorbed_activation_name(act) == name


def test_the_table_lists_exactly_the_absorbable_activations():
    assert ABSORBABLE_ACTIVATIONS == (nn.ReLU, nn.LeakyReLU, nn.GELU)


def test_every_absorbable_activation_is_rebuildable_by_name():
    """A Perceptron rebuilds an absorbed activation from its name; an unknown name would fall back to ReLU."""
    for cls in ABSORBABLE_ACTIVATIONS:
        assert absorbed_activation_name(cls()) in ACTIVATION_REGISTRY


def test_the_mixins_keep_no_private_copy():
    for mixin in (LinearConvertMixin, ConvConvertMixin):
        assert not hasattr(mixin, "_activation_to_name")
        assert not hasattr(mixin, "_conv_activation_to_name")
