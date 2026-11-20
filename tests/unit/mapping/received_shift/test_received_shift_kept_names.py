"""[U-RSHIFT] The kept names keep today's behaviour on a second call.

``_bake_shift_into_host_bias`` writes a correction no source names, at every call (as on the fold unit's revision 3: two
calls add twice); ``apply_negative_shift_bias`` is one source over every axon (today's once-only bake: a second call
writes nothing).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.tuning.orchestration.signed_seam_install import _bake_shift_into_host_bias


def _carrier(kind: str) -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(6, 4) if kind == "linear" else nn.MultiheadAttention(6, 2)


def _bias_and_weight(module: nn.Module) -> tuple:
    if isinstance(module, nn.Linear):
        return module.bias, module.weight
    return module.in_proj_bias, module.in_proj_weight


@pytest.mark.parametrize("kind", ["linear", "attention"])
def test_the_host_bake_adds_at_every_call_as_today(kind):
    module = _carrier(kind)
    bias, weight = _bias_and_weight(module)
    trained, w = bias.detach().clone(), weight.detach().clone()
    assert _bake_shift_into_host_bias(module, 0.37)
    after_one = bias.detach().clone()
    assert _bake_shift_into_host_bias(module, 0.21)
    want_one = trained - 0.37 * w.sum(dim=1)
    want_two = want_one - 0.21 * w.sum(dim=1)
    assert torch.allclose(after_one, want_one, atol=1e-6), float((after_one - want_one).abs().max())
    assert torch.allclose(bias.detach(), want_two, atol=1e-6), (
        f"the second call is off the two-call sum by {float((bias.detach() - want_two).abs().max()):.4g}")


def test_the_perceptron_bake_over_every_axon_stays_once_only():
    torch.manual_seed(0)
    perceptron = Perceptron(4, 6)
    trained, w = perceptron.layer.bias.detach().clone(), perceptron.layer.weight.detach().clone()
    apply_negative_shift_bias(perceptron, torch.tensor(0.5))
    after_one = perceptron.layer.bias.detach().clone()
    apply_negative_shift_bias(perceptron, torch.tensor(0.3))
    assert torch.allclose(after_one, trained - 0.5 * w.sum(dim=1), atol=1e-6)
    assert torch.equal(perceptron.layer.bias.detach(), after_one)
