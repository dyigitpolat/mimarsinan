"""torchvision DenseNet-121 (D1), GoogLeNet and Inception-v3 (D10, eps 1e-3) convert exactly at fp64.

Built without weights (random init), BatchNorm statistics randomized; small inputs keep the unit tier fast.
"""

from __future__ import annotations

import pytest
import torch
import torchvision.models as tvm

from . import witnesses as W

_ZOO = {
    "densenet121": (lambda: tvm.densenet121(weights=None), (3, 64, 64)),
    "googlenet": (lambda: tvm.googlenet(weights=None, aux_logits=False, init_weights=True), (3, 64, 64)),
    "inception_v3": (lambda: tvm.inception_v3(weights=None, aux_logits=False, init_weights=True), (3, 80, 80)),
}


@pytest.mark.timeout(240)
@pytest.mark.parametrize("contract", ["mvm", "spiking"])
@pytest.mark.parametrize("name", sorted(_ZOO))
def test_torchvision_member_converts_exactly(name, contract):
    factory, shape = _ZOO[name]
    torch.manual_seed(0)
    model = W.fp64_eval_copy(factory())
    assert W.converted_error(model, shape, contract) <= W.PARITY_BOUND
