"""The per-core reference loop's span arms, each pinned by its own value.

``accumulate_output_spans`` and ``build_cycle_activity_plan`` carry five arms
that no existing fixture reached — the always-on span's one-event-per-cycle
contribution, the raw-input span, the off span, a core with no latency, and
the ungated policy's every-core-every-cycle plan. Each is a semantic claim
about what a cycle DELIVERS, so each is pinned by the number it produces.
"""

from __future__ import annotations

import torch

from mimarsinan.mapping.support.spike_source_spans import SpikeSourceSpan
from mimarsinan.models.spiking.hybrid.executors.reference_loop import (
    accumulate_output_spans,
    allocate_record_rasters,
    build_cycle_activity_plan,
)

T = 4


class _Core:
    def __init__(self, latency, neurons=2, axons=3):
        self.latency = latency
        self.neurons_per_core = neurons
        self.available_neurons = 0
        self.axons_per_core = axons
        self.available_axons = 0


def _span(kind, *, dst_start, length, src_core=0, src_start=0):
    return SpikeSourceSpan(
        kind=kind, src_core=src_core, src_start=src_start, length=length,
        dst_start=dst_start,
    )


def _accumulate(spans, cores, *, cycles, buffers, input_spikes, width):
    counts = torch.zeros(1, width, dtype=torch.float64)
    for cycle in range(cycles):
        accumulate_output_spans(
            counts, spans, cores, cycle=cycle, T=T, buffers=buffers,
            input_spikes=input_spikes)
    return counts


def test_an_always_on_span_delivers_exactly_one_event_per_in_window_cycle():
    """The kill for the ``+= 1.0`` mutant: an always-on axon is ONE event per
    cycle, so its window count is exactly T — not 2T, and not T+1."""
    counts = _accumulate(
        [_span("on", dst_start=0, length=2)], [_Core(0)],
        cycles=T + 3, buffers=[torch.zeros(1, 2, dtype=torch.float64)],
        input_spikes=torch.zeros(1, 3, dtype=torch.float64), width=2)
    assert counts.tolist() == [[float(T), float(T)]]


def test_an_input_span_forwards_the_raw_input_only_inside_the_window():
    spikes = torch.tensor([[2.0, 5.0, 7.0]], dtype=torch.float64)
    counts = _accumulate(
        [_span("input", dst_start=1, length=2, src_start=1)], [_Core(0)],
        cycles=T + 2, buffers=[torch.zeros(1, 2, dtype=torch.float64)],
        input_spikes=spikes, width=3)
    assert counts.tolist() == [[0.0, 5.0 * T, 7.0 * T]]


def test_an_off_span_contributes_nothing_at_any_cycle():
    counts = _accumulate(
        [_span("off", dst_start=0, length=2)], [_Core(0)],
        cycles=T + 2, buffers=[torch.ones(1, 2, dtype=torch.float64)],
        input_spikes=torch.ones(1, 3, dtype=torch.float64), width=2)
    assert counts.tolist() == [[0.0, 0.0]]


def test_a_core_with_no_latency_never_contributes():
    """An unplaced producer has no firing window; reading its buffer would
    publish charge from a core the schedule never ran."""
    counts = _accumulate(
        [_span("core", dst_start=0, length=2)], [_Core(None)],
        cycles=T + 2, buffers=[torch.ones(1, 2, dtype=torch.float64)],
        input_spikes=torch.zeros(1, 3, dtype=torch.float64), width=2)
    assert counts.tolist() == [[0.0, 0.0]]


def test_a_core_span_accumulates_only_inside_its_own_latency_window():
    counts = _accumulate(
        [_span("core", dst_start=0, length=2)], [_Core(2)],
        cycles=2 + T + 3, buffers=[torch.full((1, 2), 3.0, dtype=torch.float64)],
        input_spikes=torch.zeros(1, 3, dtype=torch.float64), width=2)
    assert counts.tolist() == [[3.0 * T, 3.0 * T]]


def test_an_ungated_policy_steps_and_fills_every_core_every_cycle():
    cores = [_Core(0), _Core(5), _Core(None)]
    active, fill = build_cycle_activity_plan(
        {}, cores=cores, stepable=[0, 1], cycles=3, T=T, latency_gated=False)
    assert active == [[0, 1]] * 3
    assert fill == [[0, 1, 2]] * 3


def test_a_gated_plan_is_memoized_on_the_segment_and_windowed():
    seg: dict = {}
    cores = [_Core(0), _Core(2)]
    active, fill = build_cycle_activity_plan(
        seg, cores=cores, stepable=[0, 1], cycles=T + 2, T=T,
        latency_gated=True)
    assert active is fill is seg["active_by_cycle"]
    assert active[0] == [0] and active[T + 1] == [1]
    again, _ = build_cycle_activity_plan(
        seg, cores=cores, stepable=[0, 1], cycles=T + 2, T=T,
        latency_gated=True)
    assert again is active


def test_record_rasters_are_allocated_per_core_at_window_width():
    rasters = allocate_record_rasters([_Core(0, neurons=3), _Core(1)], T,
                                      torch.device("cpu"))
    assert [tuple(r.shape) for r in rasters] == [(T, 3), (T, 2)]
    assert all(r.dtype is torch.int64 for r in rasters)
