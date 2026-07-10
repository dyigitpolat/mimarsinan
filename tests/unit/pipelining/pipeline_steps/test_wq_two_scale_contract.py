"""Two-scale WQ pipeline contracts: the post-install grid verification and the
platform capability gate.

The verification is the parity anchor: BOTH sides of the torch<->chip contract
must see the same two-scale semantics — the bias integer on its own ±q_max
register grid AND on the weight-grid lattice the chip export emits
(``bias * parameter_scale = r * bias_int``).
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
    assert_effective_parameters_on_chip_grid,
)
from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import (
    resolve_wq_two_scale_projection,
)
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)

BITS = 5
Q_MAX = (2 ** (BITS - 1)) - 1


def _perceptron(two_scale):
    torch.manual_seed(0)
    p = Perceptron(4, 8, normalization=nn.Identity())
    p.set_activation_scale(1.0)
    with torch.no_grad():
        p.layer.weight.data.uniform_(-0.06, 0.06)
        p.layer.bias.data.copy_(torch.tensor([0.75, 0.5, 0.3, 0.1]))
    NormalizationAwarePerceptronQuantization(
        bits=BITS, device="cpu", rate=1.0, two_scale=two_scale
    ).transform(p)
    return p


class TestChipGridVerification:
    @pytest.mark.parametrize("two_scale", [False, True])
    def test_quantized_perceptron_passes(self, two_scale):
        assert_effective_parameters_on_chip_grid(_perceptron(two_scale), Q_MAX)

    def test_off_grid_bias_fails(self):
        p = _perceptron(True)
        with torch.no_grad():
            # Half a bias-grid step off: on neither lattice.
            p.layer.bias.data[0] += 0.5 / float(p.bias_scale)
        with pytest.raises(AssertionError):
            assert_effective_parameters_on_chip_grid(p, Q_MAX)

    def test_bias_beyond_its_register_range_fails(self):
        p = _perceptron(True)
        with torch.no_grad():
            # Exactly on the bias grid but outside the ±q_max register range.
            p.layer.bias.data[0] += 4.0 * (Q_MAX / float(p.bias_scale))
        with pytest.raises(AssertionError):
            assert_effective_parameters_on_chip_grid(p, Q_MAX)

    def test_off_grid_weight_fails(self):
        p = _perceptron(True)
        with torch.no_grad():
            p.layer.weight.data[0, 0] += 0.4 / float(p.parameter_scale)
        with pytest.raises(AssertionError):
            assert_effective_parameters_on_chip_grid(p, Q_MAX)

    def test_pre_two_scale_perceptron_without_bias_scale_passes(self):
        p = _perceptron(False)
        del p._parameters["bias_scale"]
        assert_effective_parameters_on_chip_grid(p, Q_MAX)


class TestTwoScaleCapabilityGate:
    def _config(self, *, flag, has_bias):
        return {
            "wq_two_scale_projection": flag,
            "cores": [
                {"max_axons": 256, "max_neurons": 256, "count": 4, "has_bias": has_bias},
            ],
        }

    def test_on_chip_bias_platform_honors_the_flag(self):
        assert resolve_wq_two_scale_projection(self._config(flag=True, has_bias=True))

    def test_param_encoded_bias_platform_falls_back_to_shared_grid(self):
        # The always-on bias row must obey the ±q_max weight-register contract
        # on the weight grid; two-scale is not mappable there.
        assert not resolve_wq_two_scale_projection(
            self._config(flag=True, has_bias=False)
        )

    def test_flag_off_is_off_everywhere(self):
        assert not resolve_wq_two_scale_projection(self._config(flag=False, has_bias=True))
        assert not resolve_wq_two_scale_projection({})
