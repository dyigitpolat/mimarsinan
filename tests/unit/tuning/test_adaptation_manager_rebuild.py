"""The finalize-rebuild decorator stack under conversion-family subsumption.

F1 lock (spiking_deployment_calculus.md §8, PR1 resolved): the activation-
replacement decorator (activation_adaptation_rate) rides a buffer-backed
carrier that persists in the cached manager at alpha 1.0 even when the float
reads 0.0 — and it was the ONLY decorator family with no subsumption gate, so
every lif_active/ttfs_active rebuild replaced the installed LIF's output with
LeakyGradReLU(input) (measured: t2_04 full-transform 0.60-expected -> 0.0156).
Once the conversion family owns the node, the chip-ReLU replacement must
vanish exactly like clamp/quant/shift; before it installs, AA behavior is
unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.autograd import LeakyGradReLU
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFBlendActivation

_T = 16


def _perceptron_with_installed_lif(theta: float = 1.0):
    p = Perceptron(4, 4, normalization=nn.Identity())
    p.set_activation_scale(theta)
    lif = LIFActivation(T=_T, activation_scale=p.activation_scale)
    p.base_activation = LIFBlendActivation(LeakyGradReLU(), lif, 1.0)
    return p


def _manager_with_persisted_aa_buffer() -> AdaptationManager:
    """The torch-vehicle cached-manager state: AA ramp completed (buffer 1.0),
    float rate reset to 0.0 — the buffer alone keeps the decorator active."""
    manager = AdaptationManager()
    manager.bind_rate_buffer("activation_adaptation_rate").set(1.0)
    manager.activation_adaptation_rate = 0.0
    return manager


def _sweep(theta: float) -> torch.Tensor:
    return torch.linspace(-theta, 2 * theta, 512).reshape(8, 64)


class TestConversionFamilySubsumesReplacement:
    def test_lif_active_rebuild_is_transparent_over_the_blend(self):
        manager = _manager_with_persisted_aa_buffer()
        manager.lif_active = True
        p = _perceptron_with_installed_lif()
        manager.update_activation({"spiking_mode": "lif", "target_tq": 32}, p)
        assert list(p.activation.decorators) == []
        z = _sweep(1.0)
        with torch.no_grad():
            torch.testing.assert_close(p.activation(z), p.base_activation(z))

    def test_ttfs_active_rebuild_is_transparent_over_the_blend(self):
        manager = _manager_with_persisted_aa_buffer()
        manager.ttfs_active = True
        p = _perceptron_with_installed_lif()
        manager.update_activation(
            {"spiking_mode": "ttfs_cycle_based", "target_tq": 32}, p,
        )
        assert list(p.activation.decorators) == []

    def test_replacement_still_applies_before_the_family_installs(self):
        """AA-phase behavior preserved: with no conversion family active, the
        buffer-carried replacement decorator applies at its alpha (1.0)."""
        manager = _manager_with_persisted_aa_buffer()
        p = _perceptron_with_installed_lif()
        manager.update_activation({"spiking_mode": "lif", "target_tq": 32}, p)
        assert len(list(p.activation.decorators)) == 1
        z = _sweep(1.0)
        with torch.no_grad():
            torch.testing.assert_close(
                p.activation(z), LeakyGradReLU()(z),
            )
