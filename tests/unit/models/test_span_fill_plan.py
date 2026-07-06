"""Golden fill-equality: the precomputed gather plan is bit-equal to the span walk.

W3b: 81% of the SCM/HCM identity-sim wall was ~18.6M per-span Python slice
assignments in the per-core-per-cycle axon fill. The ``SpanFillPlan`` replaces
the walk with a handful of precomputed index copies per core-cycle; it must be
bit-equal to ``fill_signal_from_spans`` on every span-kind combination,
including single-spike latency gating.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
import torch

from mimarsinan.models.spiking.signal_spans import (
    SpanFillPlan,
    fill_signal_from_spans,
)


@dataclass(frozen=True)
class _Span:
    kind: str
    src_core: int
    src_start: int
    dst_start: int
    dst_end: int

    @property
    def src_end(self) -> int:
        return self.src_start + (self.dst_end - self.dst_start)


def _legacy_fill(out, spans, *, input_spikes, buffers, cycle, single_spike, latency):
    """The pre-W3b stage_io lambdas, verbatim semantics."""

    def _single_spike_always_on(d0: int, d1: int) -> None:
        out[:, d0:d1].fill_(1.0 if cycle == latency else 0.0)

    fill_signal_from_spans(
        out,
        spans,
        read_input=lambda sp: out.__setitem__(
            (slice(None), slice(int(sp.dst_start), int(sp.dst_end))),
            input_spikes[:, int(sp.src_start) : int(sp.src_end)],
        ),
        read_upstream=lambda sp: out.__setitem__(
            (slice(None), slice(int(sp.dst_start), int(sp.dst_end))),
            buffers[int(sp.src_core)][:, int(sp.src_start) : int(sp.src_end)],
        ),
        on_always_on=_single_spike_always_on if single_spike else None,
        cycle=cycle,
    )


def _random_spans(rng, n_dst, n_cores, input_width, core_width):
    """Partition [0, n_dst) into random-kind contiguous spans (the RLE shape)."""
    spans = []
    d = 0
    while d < n_dst:
        length = rng.randint(1, max(1, min(5, n_dst - d)))
        kind = rng.choice(["off", "on", "input", "core", "core"])
        if kind == "input":
            s0 = rng.randint(0, input_width - length)
            spans.append(_Span("input", -2, s0, d, d + length))
        elif kind == "core":
            core = rng.randint(0, n_cores - 1)
            s0 = rng.randint(0, core_width - length)
            spans.append(_Span("core", core, s0, d, d + length))
        else:
            spans.append(_Span(kind, -1, 0, d, d + length))
        d += length
    return spans


def _random_sources(rng_seed, batch, n_cores, input_width, core_width, dtype):
    g = torch.Generator().manual_seed(rng_seed)
    input_spikes = torch.rand(batch, input_width, generator=g, dtype=torch.float64).to(dtype)
    buffers = [
        torch.rand(batch, core_width, generator=g, dtype=torch.float64).to(dtype)
        for _ in range(n_cores)
    ]
    return input_spikes, buffers


class TestGoldenEquality:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    def test_randomized_span_sets_bit_equal(self, seed, dtype):
        rng = random.Random(seed)
        n_dst, n_cores, input_width, core_width, batch = 37, 4, 16, 11, 3
        spans = _random_spans(rng, n_dst, n_cores, input_width, core_width)
        input_spikes, buffers = _random_sources(
            seed, batch, n_cores, input_width, core_width, dtype,
        )
        expected = torch.full((batch, n_dst), 7.0, dtype=dtype)
        got = torch.full((batch, n_dst), 7.0, dtype=dtype)
        _legacy_fill(
            expected, spans, input_spikes=input_spikes, buffers=buffers,
            cycle=0, single_spike=False, latency=0,
        )
        plan = SpanFillPlan(spans, torch.device("cpu"))
        plan.apply(got, input_spikes=input_spikes, buffers=buffers, on_value=1.0)
        assert torch.equal(expected, got)

    @pytest.mark.parametrize("cycle,latency", [(0, 0), (1, 0), (3, 3), (5, 3)])
    def test_single_spike_latency_gating(self, cycle, latency):
        rng = random.Random(99)
        n_dst, n_cores, input_width, core_width, batch = 24, 3, 8, 8, 2
        spans = _random_spans(rng, n_dst, n_cores, input_width, core_width)
        input_spikes, buffers = _random_sources(
            99, batch, n_cores, input_width, core_width, torch.float64,
        )
        expected = torch.zeros(batch, n_dst, dtype=torch.float64)
        got = torch.zeros(batch, n_dst, dtype=torch.float64)
        _legacy_fill(
            expected, spans, input_spikes=input_spikes, buffers=buffers,
            cycle=cycle, single_spike=True, latency=latency,
        )
        plan = SpanFillPlan(spans, torch.device("cpu"))
        plan.apply(
            got, input_spikes=input_spikes, buffers=buffers,
            on_value=1.0 if cycle == latency else 0.0,
        )
        assert torch.equal(expected, got)

    def test_all_off_leaves_zeros(self):
        spans = [_Span("off", -1, 0, 0, 6)]
        out = torch.full((2, 6), 3.0, dtype=torch.float64)
        plan = SpanFillPlan(spans, torch.device("cpu"))
        plan.apply(out, input_spikes=torch.rand(2, 4), buffers=[], on_value=1.0)
        assert torch.equal(out, torch.zeros(2, 6, dtype=torch.float64))

    def test_empty_span_list_zeroes(self):
        out = torch.full((2, 5), 3.0, dtype=torch.float64)
        plan = SpanFillPlan([], torch.device("cpu"))
        plan.apply(out, input_spikes=torch.rand(2, 4), buffers=[], on_value=1.0)
        assert torch.equal(out, torch.zeros(2, 5, dtype=torch.float64))

    def test_multi_span_same_core_uses_one_group(self):
        # Two non-adjacent spans from the same core must land exactly.
        spans = [
            _Span("core", 0, 3, 0, 2),
            _Span("off", -1, 0, 2, 3),
            _Span("core", 0, 0, 3, 6),
        ]
        buffers = [torch.arange(10, dtype=torch.float64).reshape(1, 10)]
        out = torch.zeros(1, 6, dtype=torch.float64)
        plan = SpanFillPlan(spans, torch.device("cpu"))
        plan.apply(out, input_spikes=torch.zeros(1, 1), buffers=buffers, on_value=1.0)
        assert out.tolist() == [[3.0, 4.0, 0.0, 0.0, 1.0, 2.0]]

    def test_plan_tensors_accounted(self):
        rng = random.Random(3)
        spans = _random_spans(rng, 30, 3, 8, 8)
        plan = SpanFillPlan(spans, torch.device("cpu"))
        tensors = list(plan.tensors())
        assert all(isinstance(t, torch.Tensor) for t in tensors)


class TestStageIoUsesPlan:
    def test_fill_signal_tensor_prefers_plan_and_matches_span_walk(self):
        from mimarsinan.models.spiking.hybrid.stage_io import HybridStageIOMixin

        rng = random.Random(7)
        n_dst, n_cores, input_width, core_width, batch = 21, 3, 8, 8, 2
        spans = _random_spans(rng, n_dst, n_cores, input_width, core_width)
        input_spikes, buffers = _random_sources(
            7, batch, n_cores, input_width, core_width, torch.float64,
        )
        host = HybridStageIOMixin.__new__(HybridStageIOMixin)
        expected = torch.zeros(batch, n_dst, dtype=torch.float64)
        got = torch.zeros(batch, n_dst, dtype=torch.float64)
        _legacy_fill(
            expected, spans, input_spikes=input_spikes, buffers=buffers,
            cycle=2, single_spike=True, latency=2,
        )
        host._fill_signal_tensor_from_spans(
            got, input_spikes=input_spikes, buffers=buffers,
            cycle=2, single_spike=True, latency=2,
            plan=SpanFillPlan(spans, torch.device("cpu")),
        )
        assert torch.equal(expected, got)
