"""The two BatchNorm lowerings: the exact host affine op and the Perceptron normalization slot (D1, D10)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.layers import ChannelsLastBatchNorm1d
from mimarsinan.torch_mapping.converter_handlers.batchnorm_lowering import (
    BatchNormAffine,
    copy_batchnorm_state,
    perceptron_normalization,
)

from .witnesses import randomize_batchnorm


def _bn(cls, n=6, **kwargs):
    return randomize_batchnorm(cls(n, **kwargs).double()).eval()


class TestBatchNormAffine:
    @pytest.mark.parametrize("cls, shape", [
        (nn.BatchNorm1d, (3, 6)),
        (nn.BatchNorm1d, (3, 6, 5)),
        (nn.BatchNorm2d, (3, 6, 4, 5)),
        (ChannelsLastBatchNorm1d, (3, 6)),
        (ChannelsLastBatchNorm1d, (3, 5, 6)),
    ])
    @pytest.mark.parametrize("eps", [1e-5, 1e-3, 0.1])
    def test_equals_eval_batchnorm_in_fp64(self, cls, shape, eps):
        bn = _bn(cls, eps=eps)
        op = BatchNormAffine.from_batchnorm(bn)
        x = torch.randn(*shape, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            want, got = bn(x), op(x)
        assert got.dtype == torch.float64
        assert float((got - want).abs().max()) <= 1e-14 * max(1.0, float(want.abs().max()))

    def test_is_the_affine_u_x_plus_c(self):
        bn = _bn(nn.BatchNorm2d, eps=0.1)
        op = BatchNormAffine.from_batchnorm(bn)
        u = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
        torch.testing.assert_close(op.scale, u, rtol=0, atol=0)
        torch.testing.assert_close(op.shift, bn.bias.detach() - u * bn.running_mean, rtol=0, atol=0)

    def test_non_affine_batchnorm_has_unit_scale_terms(self):
        bn = _bn(nn.BatchNorm2d, affine=False)
        op = BatchNormAffine.from_batchnorm(bn)
        x = torch.randn(2, 6, 3, 3, dtype=torch.float64)
        torch.testing.assert_close(op(x), bn(x), rtol=1e-14, atol=1e-14)

    def test_mode_does_not_change_the_function_and_nothing_trains(self):
        op = BatchNormAffine.from_batchnorm(_bn(nn.BatchNorm2d))
        x = torch.randn(4, 6, 3, 3, dtype=torch.float64)
        with torch.no_grad():
            evaluated = op.eval()(x)
            trained = op.train()(x)
        assert torch.equal(evaluated, trained)
        assert list(op.parameters()) == []
        assert set(dict(op.named_buffers())) == {"scale", "shift"}

    def test_refuses_a_batchnorm_without_running_statistics(self):
        with pytest.raises(ValueError, match="running statistics"):
            BatchNormAffine.from_batchnorm(nn.BatchNorm2d(4, track_running_stats=False))

    def test_follows_dtype_and_device_moves(self):
        op = BatchNormAffine.from_batchnorm(_bn(nn.BatchNorm2d))
        assert op.scale.dtype == torch.float64
        assert op.float().scale.dtype == torch.float32


class TestPerceptronNormalization:
    @pytest.mark.parametrize("eps", [1e-5, 1e-3, 0.1])
    def test_batchnorm2d_becomes_batchnorm1d_with_its_eps_momentum_and_state(self, eps):
        src = _bn(nn.BatchNorm2d, eps=eps, momentum=0.3)
        slot = perceptron_normalization(src)
        assert type(slot) is nn.BatchNorm1d
        assert (slot.num_features, slot.eps, slot.momentum) == (6, eps, 0.3)
        for name, tensor in src.state_dict().items():
            assert torch.equal(slot.state_dict()[name], tensor), name
            assert slot.state_dict()[name].dtype == tensor.dtype, name

    def test_default_batchnorm2d_slot_is_todays_slot(self):
        src = _bn(nn.BatchNorm2d)
        slot = perceptron_normalization(src)
        reference = nn.BatchNorm1d(6).double()
        copy_batchnorm_state(reference, src)
        assert slot.extra_repr() == reference.extra_repr()
        assert slot.state_dict().keys() == reference.state_dict().keys()
        for name, tensor in reference.state_dict().items():
            assert torch.equal(slot.state_dict()[name], tensor), name

    @pytest.mark.parametrize("cls", [nn.BatchNorm1d, ChannelsLastBatchNorm1d])
    def test_batchnorm1d_is_copied_with_its_layout(self, cls):
        src = _bn(cls, eps=1e-3)
        slot = perceptron_normalization(src)
        assert type(slot) is cls and slot is not src and slot.eps == 1e-3
        for name, tensor in src.state_dict().items():
            assert torch.equal(slot.state_dict()[name], tensor), name

    @pytest.mark.parametrize("cls", [nn.BatchNorm1d, ChannelsLastBatchNorm1d, nn.BatchNorm2d])
    def test_affine_free_source_gets_an_affine_slot_of_unit_scale(self, cls):
        """One rule for every kind: the slot always carries weight and bias (1 and 0 for affine=False)."""
        src = cls(6, eps=1e-3, momentum=0.2, affine=False).double()
        with torch.no_grad():
            src.running_mean.uniform_(-0.2, 0.2)
            src.running_var.uniform_(0.5, 2.0)
        slot = perceptron_normalization(src)
        assert type(slot) is (nn.BatchNorm1d if cls is nn.BatchNorm2d else cls)
        assert slot.affine and (slot.eps, slot.momentum) == (1e-3, 0.2)
        assert torch.equal(slot.weight, torch.ones(6, dtype=torch.float64))
        assert torch.equal(slot.bias, torch.zeros(6, dtype=torch.float64))
        assert torch.equal(slot.running_mean, src.running_mean) and torch.equal(slot.running_var, src.running_var)
        x = torch.randn(4, 6, 5, dtype=torch.float64) if cls is not ChannelsLastBatchNorm1d \
            else torch.randn(4, 5, 6, dtype=torch.float64)
        with torch.no_grad():
            native = src.eval()(x.unsqueeze(-1)).squeeze(-1) if cls is nn.BatchNorm2d else src.eval()(x)
            assert torch.equal(slot.eval()(x), native)

    @pytest.mark.parametrize("eps", [1e-5, 1e-3, 0.1])
    def test_slot_reproduces_the_source_on_channel_first_input(self, eps):
        src = _bn(nn.BatchNorm2d, eps=eps)
        slot = perceptron_normalization(src).eval()
        x = torch.randn(2, 6, 4, 4, dtype=torch.float64)
        with torch.no_grad():
            got = slot(x.view(2, 6, -1)).view_as(x)
        assert torch.equal(got, src(x))

    def test_copy_batchnorm_state_follows_the_source_dtype(self):
        src = _bn(nn.BatchNorm2d)
        dst = nn.BatchNorm1d(6)
        copy_batchnorm_state(dst, src)
        assert dst.running_var.dtype == torch.float64 and torch.equal(dst.running_var, src.running_var)
        untouched = copy.deepcopy(dst)
        copy_batchnorm_state(nn.Identity(), src)
        assert torch.equal(untouched.running_mean, dst.running_mean)
