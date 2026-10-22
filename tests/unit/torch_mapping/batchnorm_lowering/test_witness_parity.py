"""fp64 parity at 1e-12 of every BatchNorm witness: converted model in both contracts and the MVM value program."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.models.families.densenet_bc import CifarDenseNetBC
from mimarsinan.models.families.pyramidnet import CifarPyramidNet
from mimarsinan.torch_mapping.converter import check_representability
from mimarsinan.torch_mapping.converter_handlers.batchnorm_lowering import BatchNormAffine

from . import witnesses as W

_D1_D8 = {**W.D1_WITNESSES, **W.D8_WITNESSES}
_FAMILIES = {
    "densenet_bc_L10_k4_published": (lambda: CifarDenseNetBC((3, 16, 16), W.NC, depth=10, growth_rate=4), (3, 16, 16)),
    "densenet_bc_L16_k4_published": (lambda: CifarDenseNetBC((3, 16, 16), W.NC, depth=16, growth_rate=4), (3, 16, 16)),
    "densenet_bc_L10_k4_post_activation": (
        lambda: CifarDenseNetBC((3, 16, 16), W.NC, depth=10, growth_rate=4, layer_order="post_activation"), (3, 16, 16)),
    "pyramidnet_d8_a12": (lambda: CifarPyramidNet((3, 16, 16), W.NC, depth=8, alpha=12), (3, 16, 16)),
    "pyramidnet_d14_a24": (lambda: CifarPyramidNet((3, 16, 16), W.NC, depth=14, alpha=24), (3, 16, 16)),
}


# The named instances of E271 / families2 section 5 at their published geometry (3 x 32 x 32, 100 classes).
_PUBLISHED = {
    "densenet_bc_L22_k12_published": lambda: CifarDenseNetBC((3, 32, 32), 100, depth=22, growth_rate=12),
    "densenet_bc_L40_k6_published": lambda: CifarDenseNetBC((3, 32, 32), 100, depth=40, growth_rate=6),
    "pyramidnet_n2_a24": lambda: CifarPyramidNet((3, 32, 32), 100, depth=14, alpha=24),
}


def _model(factory):
    torch.manual_seed(0)
    return W.fp64_eval_copy(factory())


def _host_affine_ops(flow):
    return [m for m in flow.get_mapper_repr().execution_order()
            if isinstance(m, ComputeOpMapper) and isinstance(m.module, BatchNormAffine)]


@pytest.mark.parametrize("contract", ["mvm", "spiking"])
@pytest.mark.parametrize("key", sorted({**_D1_D8, **W.SPIKING_ONLY_WITNESSES, **_FAMILIES}))
def test_converted_model_equals_native(key, contract):
    factory, shape = {**_D1_D8, **W.SPIKING_ONLY_WITNESSES, **_FAMILIES}[key]
    assert W.converted_error(_model(factory), shape, contract) <= W.PARITY_BOUND


@pytest.mark.parametrize("key", sorted({**_D1_D8, **W.SPIKING_ONLY_WITNESSES, **_FAMILIES}))
def test_mvm_value_program_equals_native(key):
    factory, shape = {**_D1_D8, **W.SPIKING_ONLY_WITNESSES, **_FAMILIES}[key]
    assert W.value_program_error(_model(factory), shape) <= W.PARITY_BOUND


@pytest.mark.parametrize("contract", ["mvm", "spiking"])
@pytest.mark.parametrize("key", sorted(W.D1_WITNESSES))
def test_every_unfolded_batchnorm_is_one_host_affine_op(key, contract):
    factory, shape = W.D1_WITNESSES[key]
    model = _model(factory)
    flow = W.convert_quietly(model, shape, contract)
    host_names = {m.name for m in _host_affine_ops(flow)}
    report = check_representability(model, shape, packaging=W.CONTRACTS[contract])
    assert host_names == W.batchnorm_node_names(model, shape) - set(report.absorption_plan)
    assert host_names, "every D1 witness has a BatchNorm that cannot fold"


@pytest.mark.parametrize("contract", ["mvm", "spiking"])
@pytest.mark.parametrize("kind", W.EPS_KINDS)
@pytest.mark.parametrize("eps", W.EPS_VALUES)
def test_batchnorm_eps_after_a_weight_layer_is_carried(kind, eps, contract):
    model = _model(lambda: W.WeightBnEps(kind, eps))
    assert W.converted_error(model, W.SHAPE, contract) <= W.PARITY_BOUND
    flow = W.convert_quietly(model, W.SHAPE, contract)
    eps_seen = {float(p.normalization.eps) for p in flow.get_perceptrons()
                if isinstance(p.normalization, nn.modules.batchnorm._BatchNorm)}
    assert eps_seen == {eps}


@pytest.mark.parametrize("kind", W.EPS_KINDS)
@pytest.mark.parametrize("eps", W.EPS_VALUES)
def test_batchnorm_eps_value_program(kind, eps):
    assert W.value_program_error(_model(lambda: W.WeightBnEps(kind, eps)), W.SHAPE) <= W.PARITY_BOUND


@pytest.mark.parametrize("kind", W.EPS_KINDS)
def test_affine_free_batchnorm_after_a_weight_layer_maps_exactly(kind):
    """The fold slot of a BatchNorm(affine=False) carries unit scale: converted and value program exact."""
    model = _model(lambda: W.WeightBnEps(kind, 1e-3, affine=False))
    for contract in ("mvm", "spiking"):
        assert W.converted_error(model, W.SHAPE, contract) <= W.PARITY_BOUND, contract
    assert W.value_program_error(model, W.SHAPE) <= W.PARITY_BOUND


@pytest.mark.timeout(240)
@pytest.mark.parametrize("key", sorted(_PUBLISHED))
def test_published_family_instances_convert_and_map_exactly(key):
    model = _model(_PUBLISHED[key])
    for contract in ("mvm", "spiking"):
        assert W.converted_error(model, (3, 32, 32), contract) <= W.PARITY_BOUND, contract
    assert W.value_program_error(model, (3, 32, 32)) <= W.PARITY_BOUND
