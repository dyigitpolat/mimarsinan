"""Kernel-level LIF exactness locks (lif_deployment_exactness.md §7.5).

Theorem 0 (charge conservation): ``Q_t = theta*c_t + m_t`` at every cycle under
the Default (subtractive) reset — the identity behind the membrane-augmented
readout. Novena breaks it (discards ``m - theta`` per fire).

Theorem 2 (commutation): for constant per-cycle drive the deployed count is the
comparator staircase of the terminal charge, ``c_T = clamp(F(Q), 0, T)`` with
``F = floor`` for ``"<="`` and ``F(x) = ceil(x-1)`` for ``"<"``.

A1 encode locks: ``to_uniform_spikes`` delivers exactly ``round(r*T)`` spikes
and every live channel pulses at cycle 0.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.recording.spike_modes import to_uniform_spikes
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
from mimarsinan.models.spiking.lif_core_step import lif_core_contribute_and_fire
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

DTYPE = torch.float64


def staircase(x: torch.Tensor, thresholding_mode: str) -> torch.Tensor:
    """Comparator staircase F: thresholds passed by charge ``x`` (theta = 1)."""
    if thresholding_mode == "<=":
        return torch.floor(x)
    return torch.ceil(x - 1.0)


class TestUniformEncodeLocks:
    """A1: exact count and cycle-0 anchoring of the Uniform encoder."""

    def test_exact_count_round_r_times_t(self):
        torch.manual_seed(1234)
        for T in (4, 8, 16, 32):
            r = torch.rand(512, dtype=DTYPE)
            train = torch.stack(
                [to_uniform_spikes(r, t, T) for t in range(T)]
            ).to(DTYPE)
            assert torch.equal(train.sum(0), torch.round(r * T))

    def test_live_channels_pulse_at_cycle_zero(self):
        torch.manual_seed(1234)
        for T in (4, 8, 16, 32):
            r = torch.rand(512, dtype=DTYPE)
            first = to_uniform_spikes(r, 0, T).to(DTYPE)
            live = torch.round(r * T) >= 1
            assert torch.all(first[live] == 1.0)


class TestTheorem0ChargeConservation:
    """Q_t = theta*c_t + m_t at every cycle under Default reset."""

    def _run(self, firing_mode: str, thresholding_mode: str):
        torch.manual_seed(7)
        T, B, n_in, n_out = 8, 16, 12, 10
        W = torch.randn(n_out, n_in, dtype=DTYPE) * 0.7
        b = torch.randn(n_out, dtype=DTYPE) * 0.1
        r = torch.rand(B, n_in, dtype=DTYPE)
        train = torch.stack([to_uniform_spikes(r, t, T) for t in range(T)]).to(DTYPE)
        theta = torch.tensor(1.0, dtype=DTYPE)
        memb = torch.zeros(B, n_out, dtype=DTYPE)
        counts = torch.zeros(B, n_out, dtype=DTYPE)
        Q = torch.zeros(B, n_out, dtype=DTYPE)
        residuals = []
        for t in range(T):
            Q = Q + torch.matmul(W, train[t].T).T + b
            fired = lif_core_contribute_and_fire(
                memb, W, train[t], theta,
                hw_bias=b,
                thresholding_mode=thresholding_mode,
                firing_mode=firing_mode,
                output_dtype=DTYPE,
            )
            counts = counts + fired
            residuals.append(float((Q - (theta * counts + memb)).abs().max()))
        return residuals

    def test_default_reset_holds_identity_every_cycle(self):
        # The independent Q accumulator rounds differently than the kernel's
        # interleaved add/subtract, so the arithmetic identity is held to
        # float64 accumulation noise (1e-12), 9+ orders under Novena's discard.
        for mode in ("<", "<="):
            residuals = self._run("Default", mode)
            assert max(residuals) < 1e-12, (
                f"charge conservation broken under Default reset ({mode}): "
                f"max |Q - (theta*c + m)| = {max(residuals)}"
            )

    def test_novena_reset_breaks_identity(self):
        residuals = self._run("Novena", "<=")
        assert max(residuals) > 1e-3, (
            "Novena zero-reset must discard charge (m - theta per fire); the "
            "identity unexpectedly held — reset semantics changed?"
        )


def _constant_drive_mapping(weights: np.ndarray) -> HybridHardCoreMapping:
    """Single latency-0 core; neuron i driven by weight ``weights[i]`` from one
    always-firing input channel (rate 1.0) — constant per-cycle drive."""
    n = len(weights)
    core = HardCore(1, n, has_bias_capability=False)
    core.core_matrix = np.asarray(weights, dtype=np.float64).reshape(1, n)
    core.axon_sources = [SpikeSource(-2, 0, is_input=True)]
    core.available_axons = 0
    core.available_neurons = 0
    core.threshold = 1.0
    core.latency = 0

    segment = HardCoreMapping([])
    segment.cores = [core]
    segment.output_sources = np.asarray(
        [SpikeSource(0, i) for i in range(n)], dtype=object
    )
    stage = HybridStage(
        kind="neural",
        name="constant_drive",
        hard_core_mapping=segment,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=n)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray(
            [IRSource(node_id=0, index=i) for i in range(n)], dtype=object
        ),
    )


def make_flow(hybrid, *, T, thresholding_mode="<=", **kwargs) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow(
        input_shape=(1,),
        hybrid_mapping=hybrid,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode=thresholding_mode,
        spiking_mode="lif",
        cycle_accurate_lif_forward=True,
        **kwargs,
    ).eval()


class TestTheorem2ConstantDriveCountLaw:
    """Deployed executor count == clamp(F(T*w), 0, T) for constant drives,
    including exact-integer charges (the tie convention per comparator)."""

    # Drives whose terminal charge T*w crosses integers exactly at T=8:
    # 0.5*8 = 4.0 locks the "<=" (fires) vs "<" (silent) tie convention.
    WEIGHTS = np.array([-0.3, 0.0, 0.11, 0.25, 0.5, 0.73, 1.0, 1.31], dtype=np.float64)

    def _deployed_counts(self, thresholding_mode: str, T: int) -> torch.Tensor:
        hybrid = _constant_drive_mapping(self.WEIGHTS)
        flow = make_flow(hybrid, T=T, thresholding_mode=thresholding_mode)
        x = torch.ones(1, 1, dtype=torch.float32)
        with torch.no_grad():
            return flow(x)[0].to(DTYPE)

    def test_count_equals_staircase_of_terminal_charge(self):
        for mode in ("<", "<="):
            for T in (4, 8, 16, 32):
                got = self._deployed_counts(mode, T)
                Q = torch.tensor(self.WEIGHTS, dtype=DTYPE) * T
                want = torch.clamp(staircase(Q, mode), 0.0, float(T))
                assert torch.equal(got, want), (
                    f"mode={mode} T={T}: executor counts {got.tolist()} != "
                    f"staircase {want.tolist()}"
                )

    def test_comparators_differ_exactly_on_integer_lattice(self):
        """The strict-'<' comparator loses exactly one level at exact-integer
        charges (the V9 lattice hazard in kernel form)."""
        T = 8
        le = self._deployed_counts("<=", T)
        lt = self._deployed_counts("<", T)
        Q = torch.tensor(self.WEIGHTS, dtype=DTYPE) * T
        on_lattice = (Q == Q.round()) & (Q > 0) & (Q <= T)
        assert torch.equal((le - lt) != 0, on_lattice)
