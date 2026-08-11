"""The host-op tread tie is decided by the lattice, not by GEMM dust.

t0_05 root cause (2026-08-11): LSQ exact-QAT trains pre-activations ONTO the
``theta/T`` tread edge; the encoding host ComputeOp then decides that tie by
whatever float dust its device/batch reduction produced. The nevresim runner
computed the op on CUDA at batch-25 (dust below the tread: hold) while the
certificate twin ran it on CPU at batch-2 (dust above: fire) — one count in,
17 windows out. The fix: chip probe AND certificate twin both execute inside
``measurement_plane()`` (and on one device), where the armed lattice node
snaps the membrane so the tie is decided by the exact value — like the chip's
all-integer arithmetic, where charge == theta under strict "<" HOLDS."""

from __future__ import annotations

import torch

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import measurement_plane

THETA = 1.8761228322982788  # t0_05's encoder scale (theta_int = 328)
THETA_INT = 328.0
T = 4


def _counts(lif: LIFActivation, x: torch.Tensor) -> torch.Tensor:
    """Rate-mode forward decoded back to integer window counts."""
    with torch.no_grad():
        value = lif(x)
    return torch.round(value / THETA * T)


def _armed_lif() -> LIFActivation:
    lif = LIFActivation(T, THETA, thresholding_mode="<")
    lif.set_membrane_lattice(THETA_INT)
    return lif


class TestHostOpTreadTie:
    def test_exact_tie_holds_under_strict_regardless_of_dust(self):
        # Pre-activation exactly on the first tread (theta/T) plus dust of
        # BOTH signs at f32-reduction magnitude: inside the measurement
        # plane all three decide identically, and like the chip (charge ==
        # theta never fires under strict "<").
        lif = _armed_lif()
        tread = THETA / T
        dust = 6e-8  # measured CUDA-vs-CPU GEMM delta magnitude on t0_05
        x = torch.tensor(
            [[tread, tread - dust, tread + dust]], dtype=torch.float32,
        )
        with measurement_plane():
            counts = _counts(lif, x)
        assert counts.tolist() == [[0.0, 0.0, 0.0]]

    def test_off_tie_charges_are_untouched(self):
        # The snap is an exact projection, not a tolerance: values a full
        # lattice step past the tread keep their honest counts.
        lif = _armed_lif()
        step = THETA / (2.0 * THETA_INT)
        x = torch.tensor(
            [[THETA / T + 2 * step, 2 * THETA / T + 2 * step, THETA]],
            dtype=torch.float32,
        )
        with measurement_plane():
            counts = _counts(lif, x)
        # One step above tread k: fires k times over T cycles (strict "<").
        assert counts.tolist() == [[1.0, 2.0, 3.0]]

    def test_tie_decision_is_batch_shape_invariant_in_plane(self):
        # The certificate twin runs a different batch shape than the probe;
        # inside the plane the tie decision must not depend on it.
        lif = _armed_lif()
        tread = THETA / T
        row = [tread, tread + 5e-8, 0.6 * THETA]
        small = torch.tensor([row], dtype=torch.float32)
        big = torch.tensor([row] * 25, dtype=torch.float32)
        with measurement_plane():
            c_small = _counts(lif, small)
            c_big = _counts(lif, big)
        assert torch.equal(c_big[:1], c_small)
