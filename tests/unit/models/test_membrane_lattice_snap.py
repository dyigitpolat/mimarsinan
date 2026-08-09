"""[nevresim parity] Integer-chip membrane lattice: float summation noise must
never decide a threshold tie — the 2026-08-09 NF↔SCM 13/788 catch (a true tie
fired on one twin and not the other)."""

import torch

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.spiking.cycle_policy import LIFCyclePolicy

THETA = 24.0


class TestNfLatticeSnap:
    def _lif(self, lattice):
        lif = LIFActivation(T=4, activation_scale=THETA)
        lif.eval()
        lif.set_cycle_accurate(True)
        if lattice:
            lif.set_membrane_lattice(THETA)
        return lif

    def test_true_tie_fires_and_resets_exactly(self):
        """Charges 41/24 then 7/24: after the first fire the pre-fire value at
        cycle 1 is EXACTLY theta (17+7=24). With the lattice armed the tie
        fires and the membrane lands at exactly zero."""
        lif = self._lif(lattice=True)
        with torch.no_grad():
            s0 = lif(torch.tensor([[41.0]]))
            s1 = lif(torch.tensor([[7.0]]))
        assert float(s0) == THETA and float(s1) == THETA
        assert float(lif.if_node.v) == 0.0

    def test_unarmed_keeps_default_class_behavior(self):
        lif = self._lif(lattice=False)
        assert lif.if_node.lattice_scale is None
        with torch.no_grad():
            lif(torch.tensor([[41.0]]))

    def test_strict_comparator_holds_the_true_tie(self):
        """The t0_45 catch: under '<' a TRUE tie must NOT fire; float noise
        (+3e-8) fired it in the NF while the exact SCM held it. With the
        lattice armed the tie value is exact and the strict comparator holds."""
        lif = LIFActivation(T=4, activation_scale=THETA, thresholding_mode="<")
        lif.eval()
        lif.set_cycle_accurate(True)
        lif.set_membrane_lattice(THETA)
        with torch.no_grad():
            s0 = lif(torch.tensor([[41.0]]))   # 41 > 24: fires, memb 17/24
            s1 = lif(torch.tensor([[7.0]]))    # 17+7 == 24: TIE — must hold
        assert float(s0) == THETA
        assert float(s1) == 0.0
        assert float(lif.if_node.v) == 1.0


class TestArmFromParameterScale:
    def test_arm_integer_membrane_lattice_stamps_from_ps(self):
        import torch.nn as nn

        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
        from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice

        class _M(nn.Module):
            def __init__(self, ps):
                super().__init__()
                self.p = Perceptron(4, 8, normalization=nn.Identity())
                self.p.set_activation_scale(1.0)
                lif = LIFActivation(T=4, activation_scale=self.p.activation_scale)
                self.p.base_activation = lif
                self.p.activation = lif
                self.p.set_parameter_scale(torch.tensor(float(ps)))

            def get_perceptrons(self):
                return [self.p]

        m = _M(24.0)
        assert arm_integer_membrane_lattice(m) == 1
        assert m.p.activation.if_node.lattice_scale == 48.0

        frac = _M(24.37)
        assert arm_integer_membrane_lattice(frac) == 0


class TestScmLatticeSnap:
    def _advance(self, contribution, *, integer_lattice):
        policy = LIFCyclePolicy("Default", integer_lattice=integer_lattice)
        state = policy.make_state(1, 1, "cpu", torch.float32)
        out = policy.advance(
            state,
            torch.tensor([[contribution]]),
            torch.tensor([THETA]),
            thresholding_mode="<=",
        )
        return float(out), float(state["memb"])

    def test_noise_above_tie_snaps_to_exact_tie(self):
        fired, memb = self._advance(24.0000305, integer_lattice=True)
        assert fired == 1.0
        assert memb == 0.0

    def test_noise_below_tie_snaps_up_and_fires(self):
        fired, memb = self._advance(23.9999695, integer_lattice=True)
        assert fired == 1.0
        assert memb == 0.0

    def test_without_lattice_noise_decides_the_tie(self):
        fired, _ = self._advance(23.9999695, integer_lattice=False)
        assert fired == 0.0


class TestNapqStampsLattice:
    def test_transform_arms_the_activation(self):
        import torch.nn as nn

        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
        from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
            NormalizationAwarePerceptronQuantization,
        )

        torch.manual_seed(0)
        p = Perceptron(4, 8, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        lif = LIFActivation(T=4, activation_scale=p.activation_scale)
        p.base_activation = lif
        p.activation = lif
        NormalizationAwarePerceptronQuantization(
            8, "cpu", rate=1.0, two_scale=False,
        ).transform(p)
        expected = 2.0 * float(p.parameter_scale)
        assert lif.if_node.lattice_scale == expected
