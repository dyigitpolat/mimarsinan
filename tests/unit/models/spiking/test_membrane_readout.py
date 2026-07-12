"""Membrane-augmented readout (C2, lif_deployment_exactness.md §4/§7.1).

By Theorem 0 the terminal charge is ``Q = theta*c_T + m_T``; decoding output
cores as ``counts + m_T/theta - 1/2`` (half-step charge removed) recovers the
exact, unquantized, sign-carrying pre-activation. Realizability: the residual
membrane is claimable ONLY for output cores whose decoded value feeds nothing
but the network output (a host-side read); anything re-consumed on-chip keeps
the count decode.

Currency contract: the correction is a HOST-READ decode applied strictly at
the flow's decode-to-logits boundary. The per-neuron spike-count currency —
segment ``output_counts``, ``seg_output_spike_count`` records, per-core
records — must stay the raw clamp counts (nonnegative integers) so parity
gathers and cross-sim comparisons remain bit-identical readout-on vs off.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline

from mimarsinan.chip_simulation.recording.spike_recorder import compare_records
from mimarsinan.pipelining.core import simulation_factory
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
from mimarsinan.models.spiking.hybrid.membrane_readout import (
    final_only_output_nodes,
    membrane_readout_ranges,
)

from .test_lif_exactness_locks import _constant_drive_mapping, make_flow, staircase

DTYPE = torch.float64

WEIGHTS = np.array([-0.4, 0.0, 0.11, 0.25, 0.5, 0.73, 1.31], dtype=np.float64)
T = 8


def _forward(hybrid, **kwargs) -> torch.Tensor:
    flow = make_flow(hybrid, T=T, **kwargs)
    x = torch.ones(1, 1, dtype=torch.float32)
    with torch.no_grad():
        return flow(x)[0].to(DTYPE)


class TestFlagOffByteIdentical:
    def test_default_construction_has_no_membrane_term(self):
        """Without the knob the readout is the pure count decode (byte-identical
        to the pre-C2 executor)."""
        base = _forward(_constant_drive_mapping(WEIGHTS))
        explicit_off = _forward(
            _constant_drive_mapping(WEIGHTS), membrane_readout=False,
        )
        counts = torch.clamp(
            staircase(torch.tensor(WEIGHTS, dtype=DTYPE) * T, "<="), 0.0, float(T),
        )
        assert torch.equal(base, counts)
        assert torch.equal(explicit_off, counts)


class TestChargeIdentityReadout:
    def test_readout_recovers_exact_terminal_charge(self):
        """logits = Q/theta - 1/2 exactly (theta=1, constant drive => Q = T*w)."""
        got = _forward(
            _constant_drive_mapping(WEIGHTS),
            membrane_readout=True,
            membrane_readout_half_step=True,
        )
        want = torch.tensor(WEIGHTS, dtype=DTYPE) * T - 0.5
        assert torch.allclose(got, want, atol=1e-6), (
            f"membrane readout {got.tolist()} != Q - 1/2 {want.tolist()}"
        )

    def test_readout_is_sign_carrying(self):
        """Negative pre-activations survive the readout (the count decode is
        ReLU-positive; the charge readout is not)."""
        got = _forward(
            _constant_drive_mapping(WEIGHTS), membrane_readout=True,
        )
        assert float(got[0]) < 0.0

    def test_half_step_charge_removal_is_conditional(self):
        """Without the half-step bake the raw charge is reported (no -1/2)."""
        got = _forward(
            _constant_drive_mapping(WEIGHTS),
            membrane_readout=True,
            membrane_readout_half_step=False,
        )
        want = torch.tensor(WEIGHTS, dtype=DTYPE) * T
        assert torch.allclose(got, want, atol=1e-6)

    def test_strict_comparator_readout_matches_charge(self):
        got = _forward(
            _constant_drive_mapping(WEIGHTS),
            thresholding_mode="<",
            membrane_readout=True,
        )
        want = torch.tensor(WEIGHTS, dtype=DTYPE) * T - 0.5
        assert torch.allclose(got, want, atol=1e-6)


def _two_stage_shared_node_mapping() -> HybridHardCoreMapping:
    """Stage 0 (node 0) feeds BOTH stage 1 (node 1) and the final output; only
    node 1 is final-only, so only its slice may claim the membrane."""

    def _identity_core(weight: float) -> HardCore:
        core = HardCore(1, 1, has_bias_capability=False)
        core.core_matrix = np.asarray([[weight]], dtype=np.float64)
        core.axon_sources = [SpikeSource(-2, 0, is_input=True)]
        core.available_axons = 0
        core.available_neurons = 0
        core.threshold = 1.0
        core.latency = 0
        return core

    def _stage(node_id: int, input_node: int, weight: float) -> HybridStage:
        segment = HardCoreMapping([])
        segment.cores = [_identity_core(weight)]
        segment.output_sources = np.asarray([SpikeSource(0, 0)], dtype=object)
        return HybridStage(
            kind="neural",
            name=f"stage_{node_id}",
            hard_core_mapping=segment,
            input_map=[SegmentIOSlice(node_id=input_node, offset=0, size=1)],
            output_map=[SegmentIOSlice(node_id=node_id, offset=0, size=1)],
        )

    return HybridHardCoreMapping(
        stages=[_stage(0, -2, 0.73), _stage(1, 0, 0.73)],
        output_sources=np.asarray(
            [IRSource(node_id=0, index=0), IRSource(node_id=1, index=0)],
            dtype=object,
        ),
    )


class TestRealizabilityGating:
    def test_final_only_output_nodes(self):
        hybrid = _two_stage_shared_node_mapping()
        assert final_only_output_nodes(hybrid) == {1}

    def test_membrane_readout_ranges_per_stage(self):
        hybrid = _two_stage_shared_node_mapping()
        assert membrane_readout_ranges(hybrid, hybrid.stages[0]) == []
        assert membrane_readout_ranges(hybrid, hybrid.stages[1]) == [(0, 1)]

    def test_reconsumed_node_keeps_count_decode(self):
        """The shared node's final read stays an integer count; the final-only
        node's read carries the residual membrane."""
        got = _forward(
            _two_stage_shared_node_mapping(),
            membrane_readout=True,
            membrane_readout_half_step=True,
        )
        # Node 0: count decode of Q = 0.73 * 8 = 5.84 -> floor = 5 (integer).
        assert float(got[0]) == 5.0
        # Node 1: charge readout of Q = 0.73 * 5 (5 input spikes) - 1/2.
        assert torch.allclose(got[1], torch.tensor(0.73 * 5 - 0.5, dtype=DTYPE),
                              atol=1e-6)

    def test_input_kind_final_output_not_claimed(self):
        """A final output sourced from the raw input has no membrane to export;
        the correction must not claim it."""
        hybrid = _constant_drive_mapping(WEIGHTS[:1])
        hybrid.output_sources = np.asarray(
            [IRSource(node_id=0, index=0), IRSource(node_id=-2, index=0)],
            dtype=object,
        )
        flow = make_flow(hybrid, T=T, membrane_readout=True)
        x = torch.full((1, 1), 0.5, dtype=torch.float32)
        with torch.no_grad():
            got = flow(x)[0].to(DTYPE)
        # Raw-input passthrough: rate 0.5 * T, untouched by the readout.
        assert float(got[1]) == 0.5 * T


def _raw_clamp_counts() -> torch.Tensor:
    """Deployed count law for the constant-drive fixture: clamp(floor(T*w), 0, T)."""
    return torch.clamp(
        staircase(torch.tensor(WEIGHTS, dtype=DTYPE) * T, "<="), 0.0, float(T),
    )


class TestSpikeCountCurrencySeparation:
    """The readout must never leak into the spike-count currency (the measured
    cluster failure: negative seg_output values on the compress_spike_sources
    output_sources gather path tripping the SANA-FE/Loihi parity gate)."""

    def test_seg_output_record_is_raw_clamp_counts_when_armed(self):
        """With the readout armed, ``seg_output_spike_count`` equals the raw
        clamp counts (nonnegative, integer) while the logits carry the
        membrane term."""
        flow = make_flow(
            _constant_drive_mapping(WEIGHTS), T=T,
            membrane_readout=True, membrane_readout_half_step=True,
        )
        x = torch.ones(1, 1, dtype=torch.float32)
        with torch.no_grad():
            out, record = flow.forward_with_recording(x)

        seg = record.segments[0]
        want_counts = _raw_clamp_counts().numpy().astype(np.int64)
        np.testing.assert_array_equal(seg.seg_output_spike_count, want_counts)
        assert (seg.seg_output_spike_count >= 0).all()

        want_logits = torch.tensor(WEIGHTS, dtype=DTYPE) * T - 0.5
        assert torch.allclose(out[0].to(DTYPE), want_logits, atol=1e-6)

    def test_records_bit_identical_armed_vs_off(self):
        """The full parity gathering path (seg input/output, per-core counts)
        is bit-identical readout-on vs readout-off."""
        x = torch.ones(1, 1, dtype=torch.float32)
        records = {}
        for armed in (False, True):
            flow = make_flow(
                _constant_drive_mapping(WEIGHTS), T=T, membrane_readout=armed,
            )
            with torch.no_grad():
                _, records[armed] = flow.forward_with_recording(x)
        assert compare_records(records[False], records[True]) == []

    def test_segment_runner_counts_untouched_correction_stashed(self):
        """``_run_neural_segment_rate`` returns raw clamp counts even when
        armed; the membrane term travels the side channel keyed by node id."""
        flow = make_flow(
            _constant_drive_mapping(WEIGHTS), T=T,
            membrane_readout=True, membrane_readout_half_step=True,
        )
        batch = 3
        train = torch.ones(T, batch, 1, dtype=torch.float64)
        corrections: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            counts = flow._run_neural_segment_rate(
                flow.hybrid_mapping.stages[0],
                input_spike_train=train,
                readout_corrections=corrections,
            )
        assert set(corrections) == {0}
        want_counts = _raw_clamp_counts()
        want_decoded = torch.tensor(WEIGHTS, dtype=DTYPE) * T - 0.5
        for b in range(batch):
            assert torch.equal(counts[b].to(DTYPE), want_counts)
            decoded = counts[b].to(DTYPE) + corrections[0][b].to(DTYPE)
            assert torch.allclose(decoded, want_decoded, atol=1e-6)

    def test_segment_runner_without_side_channel_keeps_raw_counts(self):
        """A caller that does not opt into the decode side channel gets pure
        counts — never a silently mutated currency."""
        flow = make_flow(
            _constant_drive_mapping(WEIGHTS), T=T, membrane_readout=True,
        )
        train = torch.ones(T, 1, 1, dtype=torch.float64)
        with torch.no_grad():
            counts = flow._run_neural_segment_rate(
                flow.hybrid_mapping.stages[0], input_spike_train=train,
            )
        assert torch.equal(counts[0].to(DTYPE), _raw_clamp_counts())


def _lif_pipeline(T: int, **config_overrides) -> "MockPipeline":
    pipeline = MockPipeline()
    pipeline.config.update({
        "spiking_mode": "lif",
        "firing_mode": "Default",
        "spike_generation_mode": "Uniform",
        "thresholding_mode": "<=",
        "input_shape": (1,),
        "simulation_steps": T,
        "device": "cpu",
        "cycle_accurate_lif_forward": True,
        "lif_membrane_readout": True,
        "lif_half_step_bias": True,
    })
    pipeline.config.update(config_overrides)
    return pipeline


class TestDeployedDecodeDomainParity:
    """R8 verdict lock: the chip exports spike counts ONLY (nevresim's
    ``SpikingExecution`` returns the accumulated output buffer; the membrane
    has no read port), so every deployed-read flow built by the metric factory
    keeps the counts decode even when ``lif_membrane_readout`` is armed. The
    membrane decode exists solely as the explicit torch-side diagnostic.
    Decode-domain parity: NF (counts) == SCM (counts) == HCM (counts)."""

    def _forward_logits(self, flow) -> torch.Tensor:
        x = torch.ones(1, 1, dtype=torch.float32)
        with torch.no_grad():
            return flow(x)[0].to(DTYPE)

    def test_deployed_metric_flow_keeps_counts_decode_when_armed(self):
        pipeline = _lif_pipeline(T)
        flow = simulation_factory.build_spiking_hybrid_flow(
            pipeline, _constant_drive_mapping(WEIGHTS),
        )
        assert flow.membrane_readout is False
        assert torch.equal(self._forward_logits(flow), _raw_clamp_counts())

    def test_diagnostic_flow_carries_membrane_decode(self):
        pipeline = _lif_pipeline(T)
        flow = simulation_factory.build_spiking_hybrid_flow(
            pipeline, _constant_drive_mapping(WEIGHTS),
            membrane_readout_diagnostic=True,
        )
        assert flow.membrane_readout is True
        want = torch.tensor(WEIGHTS, dtype=DTYPE) * T - 0.5
        assert torch.allclose(self._forward_logits(flow), want, atol=1e-6)

    def test_diagnostic_flow_respects_half_step_flag(self):
        pipeline = _lif_pipeline(T, lif_half_step_bias=False)
        flow = simulation_factory.build_spiking_hybrid_flow(
            pipeline, _constant_drive_mapping(WEIGHTS),
            membrane_readout_diagnostic=True,
        )
        want = torch.tensor(WEIGHTS, dtype=DTYPE) * T
        assert torch.allclose(self._forward_logits(flow), want, atol=1e-6)

    def test_diagnostic_request_is_inert_off_lif(self):
        """The diagnostic decode is LIF-only; other modes never arm it."""
        pipeline = _lif_pipeline(
            T,
            spiking_mode="ttfs_quantized",
            firing_mode="TTFS",
            spike_generation_mode="TTFS",
        )
        flow = simulation_factory.build_spiking_hybrid_flow(
            pipeline, _constant_drive_mapping(WEIGHTS),
            membrane_readout_diagnostic=True,
        )
        assert flow.membrane_readout is False


class _RecordingReporter:
    def __init__(self):
        self.events = []

    def event(self, kind, payload):
        self.events.append((kind, payload))


class TestMembraneReadoutDiagnostic:
    """The engagement reporter (R8 defect 1): an armed readout must announce
    itself — eligible nodes, measured logit delta, prediction flips — and say
    plainly that deployed reads exclude it."""

    def _stats(self, pipeline, samples=None):
        if samples is None:
            samples = torch.ones(4, 1, dtype=torch.float32)
        return simulation_factory.run_membrane_readout_diagnostic(
            pipeline, _constant_drive_mapping(WEIGHTS), samples,
        )

    def test_gated_off_when_knob_unarmed(self):
        pipeline = _lif_pipeline(T, lif_membrane_readout=False)
        assert self._stats(pipeline) is None

    def test_gated_off_for_non_lif_modes(self):
        pipeline = _lif_pipeline(
            T,
            spiking_mode="ttfs_quantized",
            firing_mode="TTFS",
            spike_generation_mode="TTFS",
        )
        assert self._stats(pipeline) is None

    def test_engagement_stats_measure_the_decode_delta(self):
        pipeline = _lif_pipeline(T)
        stats = self._stats(pipeline)
        assert stats is not None
        assert stats["eligible_nodes"] == 1
        assert stats["samples"] == 4
        assert stats["engaged"] is True
        want_delta = float(
            (torch.tensor(WEIGHTS, dtype=DTYPE) * T - 0.5 - _raw_clamp_counts())
            .abs().max()
        )
        assert stats["max_abs_correction"] == pytest.approx(want_delta, abs=1e-6)
        assert isinstance(stats["pred_flips"], int)

    def test_engagement_line_and_reporter_event_emitted(self, capsys):
        pipeline = _lif_pipeline(T)
        reporter = _RecordingReporter()
        pipeline.reporter = reporter
        stats = self._stats(pipeline)
        out = capsys.readouterr().out
        assert "[C2] membrane-readout diagnostic" in out
        assert "counts decode" in out
        assert [(k, p) for k, p in reporter.events
                if k == "membrane_readout_diagnostic"] == [
            ("membrane_readout_diagnostic", stats),
        ]

    def test_diagnostic_never_mutates_deployed_decode(self):
        """Running the diagnostic must not flip the deployed flow's decode."""
        pipeline = _lif_pipeline(T)
        self._stats(pipeline)
        flow = simulation_factory.build_spiking_hybrid_flow(
            pipeline, _constant_drive_mapping(WEIGHTS),
        )
        assert flow.membrane_readout is False


class TestTheorem0OnExecutorState:
    def test_count_plus_membrane_equals_charge_random_hop(self):
        """Executor-level Theorem-0 lock: for uniform-encoded random rates the
        readout (counts + m/theta) equals the exact terminal charge computed
        from the delivered spike counts."""
        torch.manual_seed(3)
        n = 6
        weights = (torch.rand(n, dtype=DTYPE) * 2.0 - 0.5).numpy()
        hybrid = _constant_drive_mapping(weights)
        flow = make_flow(hybrid, T=T, membrane_readout=True,
                         membrane_readout_half_step=False)
        for rate in (0.125, 0.5, 0.875, 1.0):
            x = torch.full((1, 1), rate, dtype=torch.float32)
            with torch.no_grad():
                got = flow(x)[0].to(DTYPE)
            n_spikes = round(rate * T)
            want = torch.tensor(weights, dtype=DTYPE) * n_spikes
            assert torch.allclose(got, want, atol=1e-6), (
                f"rate={rate}: readout {got.tolist()} != charge {want.tolist()}"
            )
