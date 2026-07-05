"""Rate-path segment tensors + latency are retained under a byte budget (W3 wall)."""

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

T = 4


def _one_core_segment(seed):
    rng = np.random.default_rng(seed)
    core = HardCore(4, 4, has_bias_capability=False)
    core.core_matrix = np.abs(rng.normal(size=(4, 4))) * 0.4
    core.axon_sources = [
        SpikeSource(-2, i, is_input=True) for i in range(4)
    ]
    core.available_axons = 0
    core.available_neurons = 0
    core.threshold = 1.0
    core.latency = 0
    segment = HardCoreMapping([])
    segment.cores = [core]
    segment.output_sources = np.asarray(
        [SpikeSource(0, i) for i in range(4)], dtype=object,
    )
    return segment


def _two_stage_hybrid():
    seg0 = _one_core_segment(0)
    seg1 = _one_core_segment(1)
    stages = [
        HybridStage(
            kind="neural", name="s0", hard_core_mapping=seg0,
            input_map=[SegmentIOSlice(node_id=-2, offset=0, size=4)],
            output_map=[SegmentIOSlice(node_id=0, offset=0, size=4)],
        ),
        HybridStage(
            kind="neural", name="s1", hard_core_mapping=seg1,
            input_map=[SegmentIOSlice(node_id=0, offset=0, size=4)],
            output_map=[SegmentIOSlice(node_id=1, offset=0, size=4)],
        ),
    ]
    return HybridHardCoreMapping(
        stages=stages,
        output_sources=np.asarray(
            [IRSource(node_id=1, index=i) for i in range(4)], dtype=object,
        ),
    )


def _make_flow():
    return SpikingHybridCoreFlow(
        input_shape=(4,),
        hybrid_mapping=_two_stage_hybrid(),
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode="<=",
        spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    ).eval()


def _forward(flow, x):
    with torch.no_grad():
        return flow(x)


class TestSegmentCacheRetention:
    def test_cache_retains_all_stages_after_a_forward(self):
        flow = _make_flow()
        _forward(flow, torch.rand(2, 4))
        assert len(flow._segment_tensor_cache) == 2

    def test_second_forward_reuses_the_uploaded_tensors(self):
        flow = _make_flow()
        _forward(flow, torch.rand(2, 4))
        first = {
            k: v["core_params"][0] for k, v in flow._segment_tensor_cache.items()
        }
        _forward(flow, torch.rand(2, 4))
        for k, tensor in flow._segment_tensor_cache.items():
            assert tensor["core_params"][0] is first[k]

    def test_outputs_identical_across_cache_reuse(self):
        flow = _make_flow()
        x = torch.rand(3, 4)
        out_first = _forward(flow, x)
        out_second = _forward(flow, x)
        assert torch.equal(out_first, out_second)

    def test_over_budget_falls_back_to_eviction(self, monkeypatch):
        import mimarsinan.models.spiking.hybrid.stage_io as stage_io_mod

        monkeypatch.setattr(stage_io_mod, "_SEGMENT_CACHE_MAX_BYTES", 0)
        flow = _make_flow()
        x = torch.rand(2, 4)
        out = _forward(flow, x)
        assert len(flow._segment_tensor_cache) == 0
        fresh = _make_flow()
        assert torch.equal(out, _forward(fresh, x))

    def test_latency_memoized_in_the_segment_entry(self, monkeypatch):
        import mimarsinan.models.spiking.hybrid.lif_step as lif_step_mod

        flow = _make_flow()
        calls = []
        original = lif_step_mod.ChipLatency

        class _Counting(original):
            def calculate(self):
                calls.append(1)
                return super().calculate()

        monkeypatch.setattr(lif_step_mod, "ChipLatency", _Counting)
        x = torch.rand(2, 4)
        _forward(flow, x)
        first_calls = len(calls)
        assert first_calls == 2
        _forward(flow, x)
        assert len(calls) == first_calls, (
            "cached stages must reuse the memoized latency"
        )
