"""[w2-parity, defect B] The torch-versus-deployed-sim decision gate reads both twins inside the one measurement plane.

On a weight-quantized LIF program the deployed executor decides an exact threshold tie on the integer membrane lattice
by exact arithmetic, every cycle (``membrane_integer_lattice``; its host-run lattice nodes too, in any plane, since the
fix round: ``test_deployed_host_op_tie_rule.py``); the torch twin's armed lattice node snaps only inside
``measurement_plane()`` (training and tuning telemetry stay continuous, n8b/n8d). The C5 decision gate read both twins
outside the plane, so float32 dust decided the ties on the torch side (SCM = NF + 1 on 1,088 of 16,384 first-core
window counts of one sample), as the streamed exactness gate never did (it reads both twins in the plane). The gate
now reads both twins in the plane too: one tie rule for every NF-versus-SCM comparison that reads a LIF twin.

The tests hold the wiring (both reads in the plane), the tie itself on one layer (the C5 first-core values: theta_int
3, 4-bit weights), and the gate on the weight-quantized encoding-skip program, where the two defects meet as on C5:
agreement 1.0 with the logits equal, in both deployable LIF semantics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import in_measurement_plane, measurement_plane
from mimarsinan.models.spiking.cycle_policy import LIFCyclePolicy
from mimarsinan.pipelining.core.nf_scm_parity import assert_torch_vs_deployed_sim_parity_or_raise

from unit.spiking.encoding_skip_program import (
    DEPLOYED_SEMANTICS,
    SEMANTICS,
    T,
    WQ_BITS,
    WQ_THETA_ENC,
    WQ_THETAS,
    NFModel,
    build_program,
    count_mismatches,
    deployed_flow,
    nf_window_counts,
    samples,
    scm_window_counts,
)


class _Recording(nn.Module):
    """A twin as the gate sees it, recording whether each read ran in the measurement plane and what it returned."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.planes: list = []
        self.outputs: list = []

    def forward(self, x):
        self.planes.append(in_measurement_plane())
        out = self.inner(x)
        self.outputs.append(out.detach().clone())
        return out


def test_the_gate_evaluates_both_twins_inside_the_measurement_plane():
    torch.manual_seed(0)
    model, flow = _Recording(nn.Linear(3, 2)), _Recording(nn.Linear(3, 2))
    assert_torch_vs_deployed_sim_parity_or_raise(model, flow, torch.rand(4, 3), min_agreement=0.0)
    assert model.planes == [True] and flow.planes == [True]
    assert not in_measurement_plane(), "the plane closes with the gate"


# ---------------------------------------------------------------------------
# The tie on one layer (the C5 first core): exact on the chip, dust on the torch node outside the plane
# ---------------------------------------------------------------------------

N_AXONS, N_NEURONS, STEPS = 48, 512, 15
THETA, THETA_IN, THETA_INT = 0.5244447588920593, 2.289466619491577, 3.0


def _tie_layer():
    torch.manual_seed(3)
    q = torch.randint(-7, 8, (N_NEURONS, N_AXONS)).double()
    b_int = torch.randint(-4, 5, (N_NEURONS,)).double()
    return q, b_int


class _TorchLayerTwin(nn.Module):
    """The converted layer's torch twin: float32 weights on the NAPQ grid ``theta / (theta_int * theta_in)``, the
    producer's trains at ``theta_in``, the armed LIF node; window counts per neuron."""

    def __init__(self, q, b_int):
        super().__init__()
        self.register_buffer("w", (q * THETA / (THETA_INT * THETA_IN)).float())
        self.register_buffer("b", (b_int * THETA / THETA_INT).float())
        self.lif = LIFActivation(T=STEPS, activation_scale=THETA, thresholding_mode="<=")
        self.lif.set_membrane_lattice(THETA_INT)

    def forward(self, trains):  # (B, STEPS, N_AXONS) binary
        self.lif.set_cycle_accurate(True)
        total = torch.zeros(trains.shape[0], N_NEURONS)
        for t in range(STEPS):
            total = total + self.lif((trains[:, t].float() * THETA_IN) @ self.w.T + self.b) / THETA
        self.lif.set_cycle_accurate(False)
        return total.round()


class _ChipLayerTwin(nn.Module):
    """The deployed core: integer charges, threshold theta_int, the chip lattice on every cycle."""

    def __init__(self, q, b_int):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("b_int", b_int)
        self.policy = LIFCyclePolicy("Default", integer_lattice=True)

    def forward(self, trains):
        state = self.policy.make_state(trains.shape[0], N_NEURONS, "cpu", torch.float64)
        threshold = torch.full((N_NEURONS,), THETA_INT, dtype=torch.float64)
        counts = torch.zeros(trains.shape[0], N_NEURONS, dtype=torch.float64)
        for t in range(STEPS):
            counts += self.policy.step(state, self.q, trains[:, t].double(), threshold, hw_bias=self.b_int,
                                       thresholding_mode="<=")
        return counts.float()


def _trains(n=4):
    torch.manual_seed(5)
    return (torch.rand(n, STEPS, N_AXONS) < 0.3).float()


def test_the_twins_part_on_exact_ties_outside_the_plane_and_agree_inside():
    """The physics the gate must respect (the teeth of the test below): outside the plane float dust decides exact
    ties on the torch twin (it reads one count low), inside the plane both twins decide them exactly."""
    q, b_int = _tie_layer()
    torch_twin, chip_twin = _TorchLayerTwin(q, b_int), _ChipLayerTwin(q, b_int)
    trains = _trains()
    with torch.no_grad():
        chip = chip_twin(trains)
        outside = torch_twin(trains)
        with measurement_plane():
            inside = torch_twin(trains)
    assert int((outside != chip).sum()) > 0
    assert set((outside - chip)[outside != chip].tolist()) == {-1.0}
    assert torch.equal(inside, chip)


def test_the_gate_decides_exact_threshold_ties_on_both_twins_alike():
    q, b_int = _tie_layer()
    model, flow = _Recording(_TorchLayerTwin(q, b_int)), _Recording(_ChipLayerTwin(q, b_int))
    agreement = assert_torch_vs_deployed_sim_parity_or_raise(model, flow, _trains(), min_agreement=0.0)
    differ = int((model.outputs[0] != flow.outputs[0]).sum())
    assert differ == 0, f"{differ} of {model.outputs[0].numel()} window counts differ between the gate's two reads"
    assert agreement == 1.0


# ---------------------------------------------------------------------------
# The gate on the weight-quantized encoding-skip program (defects A and B together, as on C5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("semantics", DEPLOYED_SEMANTICS)
def test_the_weight_quantized_program_is_bit_exact_in_the_plane_only(semantics):
    """The C5 shape at minimum size: the first core differs outside the plane (defect B) and agrees inside it."""
    retimed, _, policy = SEMANTICS[semantics]
    program = build_program(WQ_THETA_ENC, retimed=retimed, thetas=WQ_THETAS, weight_bits=WQ_BITS)
    x = samples(64)
    flow = deployed_flow(program, semantics, membrane_integer_lattice=True)
    first = min(program.core_perceptron_indices().values())
    _, nf_outside = nf_window_counts(program, x, policy())
    _, scm, _ = scm_window_counts(program, x, flow)
    with measurement_plane():
        _, nf_inside = nf_window_counts(program, x, policy())
        _, scm_inside, _ = scm_window_counts(program, x, flow)
    assert count_mismatches(nf_outside, scm)[first] > 0
    assert count_mismatches(nf_inside, scm_inside)[first] == 0


@pytest.mark.parametrize("semantics", DEPLOYED_SEMANTICS)
def test_the_gate_holds_the_weight_quantized_encoding_skip_program_bit_exact(semantics):
    retimed, _, policy = SEMANTICS[semantics]
    program = build_program(WQ_THETA_ENC, retimed=retimed, thetas=WQ_THETAS, weight_bits=WQ_BITS)
    model = _Recording(NFModel(program, policy()))
    flow = _Recording(deployed_flow(program, semantics, membrane_integer_lattice=True))
    agreement = assert_torch_vs_deployed_sim_parity_or_raise(model, flow, samples(64), min_agreement=0.0)
    nf_counts = model.outputs[0].double() * T / program.theta_out
    scm_counts = flow.outputs[0].double()
    differ = int((torch.abs(nf_counts - scm_counts) > 1e-4).sum())
    assert differ == 0 and agreement == 1.0, (
        f"{semantics}: agreement {agreement:.4f}, {differ} of {scm_counts.numel()} output counts differ "
        f"(max |T*NF - SCM| {float((nf_counts - scm_counts).abs().max()):.3g})")
    np.testing.assert_allclose(nf_counts.numpy(), scm_counts.numpy(), atol=1e-4, rtol=0)
