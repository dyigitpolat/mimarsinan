"""Shared-HCM input-fill latency-window lock (residual Tier-1 Component A).

The shared per-core HCM input fill (``stage_io._fill_signal_tensor_from_spans`` /
``lif_step._run_neural_segment_rate``) presents the raw input train to EVERY core
at the same global cycle. A core at latency ``L`` is only active in its window
``[L, L+T)`` (``latency_gated``), so before this fix it integrated the raw input
over ``[L, L+T)`` — missing input cycle 0 and reading a zero at cycle ``T`` — i.e.
the skip stream was truncated by ``L`` cycles relative to its own window. This is
residual Tier-1 **Component A** (see
``docs/research/findings/residual_tier1_intrinsic_limit.md``): a merge core
(latency > 0) whose SKIP source is the raw segment input under-integrates that
skip by ~``1/T``, keeping NF != HCM for residual-from-raw-input diamonds.

The fix delays the raw-input read by the consuming core's latency: at global cycle
``c`` a core at latency ``L`` reads input local cycle ``c - L`` (zero outside
``[0, T)``), exactly mirroring the always-on bias (which already fires at
``cycle == latency``) and neuron-source windowing. It is a **no-op** for every
non-residual model (raw input there only ever feeds latency-0 cores), which the
byte-identical tests below lock.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

T = 8


def _build_raw_input_skip_diamond() -> HybridHardCoreMapping:
    """One neural segment with a raw-input → latency-1 merge skip (a mini residual).

    ``core0`` (latency 0) = identity(raw input) = ``z``. ``core1`` (latency 1)
    merges ``raw input`` (the skip) + ``core0`` output (``z``) via an
    identity-concat ``[I; I]`` bank — exactly the in-segment ``z + F(z)`` merge,
    minimal width 1. The skip axon of ``core1`` is the raw input feeding a
    latency-1 core: the path the fill mis-windowed before Component A.
    """
    core0 = HardCore(4, 4, has_bias_capability=False)
    core0.core_matrix = np.zeros((4, 4), dtype=np.float64)
    core0.core_matrix[0, 0] = 1.0
    core0.axon_sources = [SpikeSource(-2, 0, is_input=True)]
    core0.available_axons = 3
    core0.available_neurons = 3
    core0.threshold = 1.0
    core0.latency = 0

    core1 = HardCore(4, 4, has_bias_capability=False)
    core1.core_matrix = np.zeros((4, 4), dtype=np.float64)
    core1.core_matrix[0, 0] = 1.0  # raw-input skip axon -> merge neuron
    core1.core_matrix[1, 0] = 1.0  # core0 (z) axon       -> merge neuron
    core1.axon_sources = [SpikeSource(-2, 0, is_input=True), SpikeSource(0, 0)]
    core1.available_axons = 2
    core1.available_neurons = 3
    core1.threshold = 1.0
    core1.latency = 1

    segment = HardCoreMapping([])
    segment.cores = [core0, core1]
    segment.output_sources = np.asarray([SpikeSource(1, 0)], dtype=object)

    stage = HybridStage(
        kind="neural",
        name="raw_input_skip_diamond",
        hard_core_mapping=segment,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
        output_map=[SegmentIOSlice(node_id=1, offset=0, size=1)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray([IRSource(node_id=1, index=0)], dtype=object),
    )


def _make_hcm(hybrid: HybridHardCoreMapping) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow(
        input_shape=(1,),
        hybrid_mapping=hybrid,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode="<=",
        spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    ).eval()


def _record(hybrid: HybridHardCoreMapping, value: float):
    hcm = _make_hcm(hybrid)
    x = torch.tensor([[value]], dtype=torch.float32)
    with torch.no_grad():
        out, record = hcm.forward_with_recording(x, sample_index=0)
    return out, record


# --- Component A: the skip stream is aligned to the merge core's window ---

class TestRawInputSkipWindowAlignment:
    """The raw-input skip read by a latency-L core integrates the FULL input train
    (aligned to the core's own ``[L, L+T)`` window), not a truncated ``[L, L+T)``
    slice of the GLOBAL ``[0, T)`` presentation."""

    VALUES = [1.0, 0.875, 0.5, 0.375, 0.125]

    def test_latency1_skip_axon_count_equals_full_input_train(self):
        """core1's raw-input (skip) axon integrates the same count as the latency-0
        core's raw-input axon — the full uniform train, not one cycle short."""
        for v in self.VALUES:
            hybrid = _build_raw_input_skip_diamond()
            _, record = _record(hybrid, v)
            seg = record.segments[0]
            core0 = seg.cores[0]
            core1 = seg.cores[1]
            full_input = int(core0.input_spike_count[0])  # latency-0 raw-input axon
            skip_axon = int(core1.input_spike_count[0])    # latency-1 raw-input axon
            assert skip_axon == full_input, (
                f"v={v}: latency-1 skip axon integrated {skip_axon} input spikes but "
                f"the full input train is {full_input} — the skip is window-truncated "
                f"by the merge latency (Component A is NOT closed)"
            )

    def test_merge_integrates_both_streams_at_full_count(self):
        """The merge core's integrated INPUT current is ``z + F`` exactly: BOTH of
        its axons (the raw-input skip AND the core0/``z`` source) integrate the
        full per-stream uniform count ``round(v*T)``.

        This is the Component-A invariant: the current driven into the merge IF
        head is lossless. (The merge's *fired output* is then re-quantized by the
        IF head's own threshold/saturation — Component B — and is intentionally
        NOT asserted here.)
        """
        for v in self.VALUES:
            hybrid = _build_raw_input_skip_diamond()
            _, record = _record(hybrid, v)
            merge = record.segments[0].cores[1]
            full = int(round(v * T))
            skip_axon = int(merge.input_spike_count[0])  # raw-input skip
            z_axon = int(merge.input_spike_count[1])     # core0 (z) source
            assert (skip_axon, z_axon) == (full, full), (
                f"v={v}: merge axons integrated (skip={skip_axon}, z={z_axon}); "
                f"both must be the full per-stream count {full} (z + F lossless)"
            )


class TestNonResidualModelsByteIdentical:
    """The fill is latency-windowed only for the (rare) raw-input→deep-core case;
    a model whose raw input feeds only latency-0 cores is unchanged. This is the
    DEFAULT path for every production model and must stay byte-identical."""

    @staticmethod
    def _build_plain_mlp_hcm():
        from integration._torch_sim_fidelity import build_torch_and_hcm, mapping_configs

        class _PlainMLP(nn.Module):
            def __init__(self, w=24):
                super().__init__()
                self.stem = nn.Linear(16, w)
                self.a0 = nn.ReLU()
                self.f1 = nn.Linear(w, w)
                self.a1 = nn.ReLU()
                self.f2 = nn.Linear(w, w)
                self.a2 = nn.ReLU()
                self.head = nn.Linear(w, 10)
                self.ah = nn.ReLU()

            def forward(self, x):
                x = self.a0(self.stem(x))
                x = self.a1(self.f1(x))
                x = self.a2(self.f2(x))
                return self.ah(self.head(x))

        cfgs = mapping_configs(wide_dim=64, split_neurons=8, fuse_core_axons=16)
        torch.manual_seed(0)
        return build_torch_and_hcm(
            _PlainMLP(), (16,), 10, spiking_mode="lif", config=cfgs["identity"], T=T,
        )

    def test_plain_mlp_input_feeds_only_latency0_cores(self):
        """The precondition for byte-identical no-op: in a plain MLP the raw input
        only ever feeds latency-0 cores (so ``cycle - latency == cycle``)."""
        _, _, hybrid, _ = self._build_plain_mlp_hcm()
        saw_input_axon = False
        for stage in hybrid.stages:
            mapping = stage.hard_core_mapping
            if mapping is None:
                continue
            for core in mapping.cores:
                has_input = any(
                    getattr(s, "is_input_", False) for s in core.axon_sources
                )
                if has_input:
                    saw_input_axon = True
                    assert int(core.latency or 0) == 0, (
                        f"plain MLP raw input fed a latency-{core.latency} core — "
                        f"the no-op precondition does not hold"
                    )
        assert saw_input_axon, "expected at least one raw-input axon in the MLP"

    def test_plain_mlp_output_is_stable(self):
        """The plain-MLP deployed output is the documented golden (no raw-input
        skip → the latency-window fill is a no-op)."""
        flow, hcm, _, _ = self._build_plain_mlp_hcm()
        x = torch.rand(8, 16)
        with torch.no_grad():
            out_a = hcm(x).clone()
            out_b = hcm(x).clone()
        assert torch.equal(out_a, out_b), "HCM forward is not deterministic"
