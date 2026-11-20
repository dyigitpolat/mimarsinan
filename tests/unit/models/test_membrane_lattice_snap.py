"""[nevresim parity] Integer-chip membrane lattice: float summation noise must
never decide a threshold tie — the 2026-08-09 NF↔SCM 13/788 catch (a true tie
fired on one twin and not the other)."""

import torch

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import measurement_plane
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
        with torch.no_grad(), measurement_plane():
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
        with torch.no_grad(), measurement_plane():
            s0 = lif(torch.tensor([[41.0]]))   # 41 > 24: fires, memb 17/24
            s1 = lif(torch.tensor([[7.0]]))    # 17+7 == 24: TIE — must hold
        assert float(s0) == THETA
        assert float(s1) == 0.0
        assert float(lif.if_node.v) == 1.0


class TestMultiStepLatticeSnap:
    """The n7 t8 catch (2026-08-10): the encoding layer's RATE-mode forward
    runs spikingjelly's fused eval multi-step kernel, which bypasses the
    armed single-step snap — batch-shape GEMM dust then decides an exact
    staircase tie (rate 1.5/8 read as 1/8 or 2/8 by batch size)."""

    THETA = 1.0

    def _lif(self, thresholding_mode="<"):
        lif = LIFActivation(
            T=8, activation_scale=self.THETA, thresholding_mode=thresholding_mode,
        )
        lif.eval()
        lif.set_membrane_lattice(82.0)
        return lif

    def test_multi_step_tie_is_dust_invariant(self):
        """Constant charge z with 8z == 2*theta EXACTLY (z = 41/164 on the
        armed 1/164 lattice): the snapped multi-step count must equal the
        exact strict-hold count (1 fire), with dust of either sign."""
        for dust in (-3e-8, 0.0, +3e-8):
            lif = self._lif()
            z = 41.0 / 164.0 + dust
            with torch.no_grad(), measurement_plane():
                out = lif(torch.full((1, 3), z))
            count = round(float(out[0, 0]) / self.THETA * 8)
            assert count == 1, f"dust={dust}: count={count}"

    def test_multi_step_equals_single_step_loop(self):
        """Armed 'm'-mode forward must be bit-equal to the armed per-cycle
        's'-mode loop (the snap must not depend on step mode)."""
        torch.manual_seed(0)
        x = torch.randn(2, 5)
        m = self._lif()
        with torch.no_grad(), measurement_plane():
            rate_m = m(x)
        s = self._lif()
        s.set_cycle_accurate(True)
        spikes = []
        with torch.no_grad(), measurement_plane():
            for _ in range(8):
                spikes.append(s(x))
        rate_s = torch.stack(spikes).mean(dim=0)
        assert torch.equal(rate_m, rate_s)

    def test_unarmed_multi_step_keeps_default_path(self):
        lif = LIFActivation(T=8, activation_scale=self.THETA)
        lif.eval()
        assert lif.if_node.lattice_scale is None
        with torch.no_grad():
            lif(torch.randn(2, 4))

    def test_outside_the_measurement_plane_stays_continuous(self):
        """The chip lattice belongs to the EXPLICIT measurement plane
        (parity twins, certificates): outside it, an armed node is
        bit-identical to an unarmed one — snapped training corrupted the
        surrogate chain, and snapped tuning TELEMETRY steered keep-best /
        floor decisions onto worse trajectories (t0_01 fresh 0.9771 vs
        0.9799, n8d 2026-08-10)."""
        torch.manual_seed(5)
        x = torch.rand(8, 2, 3)
        armed = LIFActivation(T=8, activation_scale=self.THETA,
                              thresholding_mode="<")
        armed.set_membrane_lattice(82.0)
        plain = LIFActivation(T=8, activation_scale=self.THETA,
                              thresholding_mode="<")
        assert plain.if_node.lattice_scale is None
        out_a = armed.if_node(x)          # grad-enabled path
        out_p = plain.if_node(x)
        assert torch.equal(out_a, out_p)
        with torch.no_grad():             # no_grad but OUTSIDE the plane
            armed.if_node.v = 0.0
            plain.if_node.v = 0.0
            assert torch.equal(armed.if_node(x), plain.if_node(x))

    def test_encoder_grid_tie_is_dust_invariant(self):
        """The n7 t8 sample-0 catch: encoder charges live on 1/(ps*T) (input
        quantizer grid divides by T), so membranes hit HALF-points of the
        2*ps lattice (163.5/164) and the snap itself rounds on dust. With
        the encoder-armed 2*ps*T lattice the walk is exact: strict '<' holds
        cycle 0 (1308/1312 < theta) and fires cycles 1-7 — count 7, any dust."""
        for dust in (-3e-8, 0.0, +3e-8):
            lif = LIFActivation(T=8, activation_scale=self.THETA,
                                thresholding_mode="<")
            lif.eval()
            lif.set_membrane_lattice(82.0 * 8)   # encoding layer: ps * T
            z = 163.5 / 164.0 + dust
            with torch.no_grad(), measurement_plane():
                out = lif(torch.full((1, 2), z))
            count = round(float(out[0, 0]) / self.THETA * 8)
            assert count == 7, f"dust={dust}: count={count}"


class TestArmFromTheLatticeRecord:
    def test_arm_integer_membrane_lattice_derives_from_the_record_and_the_persisted_scale(self):
        """[U-POST2] The arming derives the lattice from the projection's record and the persisted parameter_scale,
        never from an integer parameter_scale alone, and caches nothing: an unrecorded perceptron stays unarmed; a
        recorded core arms at 2 ps and follows its persisted scale; a fractional scale arms nothing; an encoding layer
        is recorded on the entry grid (2 ps T) only when its input carries the 1/T entry grid."""
        import torch.nn as nn

        from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer
        from mimarsinan.models.perceptron_mixer.membrane_lattice import (
            CORE_LATTICE,
            ENTRY_GRID_LATTICE,
            record_membrane_lattice_kind,
        )
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

        unrecorded = _M(24.0)
        assert arm_integer_membrane_lattice(unrecorded) == 0
        assert unrecorded.p.activation.if_node.lattice_scale is None

        m = _M(24.0)
        assert record_membrane_lattice_kind(m.p) == CORE_LATTICE
        assert m.p.activation.if_node.lattice_scale is None  # the record arms nothing
        assert arm_integer_membrane_lattice(m) == 1
        assert m.p.activation.if_node.lattice_scale == 48.0
        m.p.set_parameter_scale(torch.tensor(10.0))  # a restore writes the persisted scale back
        assert arm_integer_membrane_lattice(m) == 1
        assert m.p.activation.if_node.lattice_scale == 20.0
        m.p.membrane_lattice_kind = None  # a restore to a state no projection reached
        assert arm_integer_membrane_lattice(m) == 0
        assert m.p.activation.if_node.lattice_scale is None

        frac = _M(24.37)
        assert record_membrane_lattice_kind(frac.p) == CORE_LATTICE
        assert arm_integer_membrane_lattice(frac) == 0
        assert frac.p.activation.if_node.lattice_scale is None

        enc = _M(24.0)
        enc.p.is_encoding_layer = True
        assert record_membrane_lattice_kind(enc.p) is None  # a float input: no lattice
        assert arm_integer_membrane_lattice(enc) == 0
        enc.p.append_input_wire_op(ChipInputQuantizer(T=4, activation_scale=1.0))
        assert record_membrane_lattice_kind(enc.p) == ENTRY_GRID_LATTICE
        assert arm_integer_membrane_lattice(enc) == 1
        # Encoding layers integrate the input-quantized pre-activation:
        # grid 1/(ps*T) => lattice 2*ps*T.
        assert enc.p.activation.if_node.lattice_scale == 48.0 * 4


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


class TestNapqRecordsLattice:
    def test_transform_records_the_lattice_and_the_arming_derives_it(self):
        """[U-POST2] The projection records the lattice's kind and arms nothing (never cached); the arming derives the
        lattice from the record and the persisted scale."""
        import torch.nn as nn

        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
        from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice
        from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
            NormalizationAwarePerceptronQuantization,
        )

        class _M(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p

            def get_perceptrons(self):
                return [self.p]

        torch.manual_seed(0)
        p = Perceptron(4, 8, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        lif = LIFActivation(T=4, activation_scale=p.activation_scale)
        p.base_activation = lif
        p.activation = lif
        NormalizationAwarePerceptronQuantization(
            8, "cpu", rate=1.0, two_scale=False,
        ).transform(p)
        assert lif.if_node.lattice_scale is None
        assert arm_integer_membrane_lattice(_M(p)) == 1
        expected = 2.0 * float(p.parameter_scale)
        assert lif.if_node.lattice_scale == expected
