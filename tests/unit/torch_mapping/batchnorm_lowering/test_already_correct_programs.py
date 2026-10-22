"""Programs whose BatchNorms all fold keep today's form: every BN folded, slots carry the source's state bitwise.

Before/after IR digests of these fixtures are recorded in the wave-0 report; these pins keep the form fixed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torchvision.models as tvm

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.models.cifar_models import cifar_resnet20, cifar_vgg8
from mimarsinan.models.families.densenet_bc import CifarDenseNetBC
from mimarsinan.models.vit_leaf import cifar_vit_leaf
from mimarsinan.torch_mapping.converter import check_representability
from mimarsinan.torch_mapping.converter_handlers.batchnorm_lowering import BatchNormAffine

from . import witnesses as W
from .census import ResMixer

_FIXTURES = {
    "cifar_resnet20": (lambda: cifar_resnet20(num_classes=W.NC), (3, 32, 32)),
    "cifar_vgg8": (lambda: cifar_vgg8(num_classes=W.NC), (3, 32, 32)),
    "cifar_vit_leaf": (lambda: cifar_vit_leaf(num_classes=W.NC), (3, 32, 32)),
    "densenet_bc_post_activation": (
        lambda: CifarDenseNetBC((3, 16, 16), W.NC, depth=10, growth_rate=4, layer_order="post_activation"), (3, 16, 16)),
    "tv_resnet18": (lambda: tvm.resnet18(weights=None, num_classes=W.NC), (3, 32, 32)),
    "residual_mixer": (ResMixer, (3, 32, 32)),
    "conv_bn_no_activation": (lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.Flatten(),
                                                    nn.Linear(8 * 64, W.NC)), W.SHAPE),
}


def _sources_by_name(model):
    return {name: m for name, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)}


@pytest.mark.timeout(240)
@pytest.mark.parametrize("contract", ["mvm", "spiking"])
@pytest.mark.parametrize("key", sorted(_FIXTURES))
def test_every_batchnorm_folds_and_slots_copy_the_source(key, contract):
    factory, shape = _FIXTURES[key]
    torch.manual_seed(0)
    model = W.fp64_eval_copy(factory())
    report = check_representability(model, shape, packaging=W.CONTRACTS[contract])
    assert W.batchnorm_node_names(model, shape) <= set(report.absorption_plan)

    flow = W.convert_quietly(model, shape, contract)
    assert not [m for m in flow.get_mapper_repr().execution_order()
                if isinstance(m, ComputeOpMapper) and isinstance(m.module, BatchNormAffine)]
    sources = list(_sources_by_name(model).values())
    for perceptron in flow.get_perceptrons():
        slot = perceptron.normalization
        if not isinstance(slot, nn.modules.batchnorm._BatchNorm):
            continue
        matches = [src for src in sources
                   if all(torch.equal(slot.state_dict()[k], v) for k, v in src.state_dict().items())]
        assert matches, f"{perceptron.name}: slot state matches no source BatchNorm"
        assert (slot.eps, slot.momentum) == (matches[0].eps, matches[0].momentum)
    x = W.seeded_inputs(shape)
    with torch.no_grad():
        assert W.relative_error(flow(x), model(x)) <= W.PARITY_BOUND
