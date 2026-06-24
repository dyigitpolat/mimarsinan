"""Weight-reuse scheduling round-1: the phase-classification keystone.

A deployment SCHEDULE over IR cores currently models every pass as a REPROGRAM
(load X params onto Y cores). The TIME-DOMAIN WEIGHT-REUSE mode recognises that
a conv kernel's banks are loaded ONCE across cores (fixed mapping) and the spatial
positions are TIME-MULTIPLEXED through them — so the first pass over a distinct
weight bank is a (costly) reprogram and every subsequent pass that resolves to the
SAME resident bank is a (cheap) reuse pass.

The reuse-vs-reprogram boundary is ALREADY in the IR and currently DISCARDED:
``conv2d_mapper`` registers one bank per conv then attaches every spatial-position
softcore to that one ``weight_bank_id``. ``{core.weight_bank_id for core in nodes}``
recovers the grouping in O(cores).

This locks the keystone:

* classifying a list of NeuralCores into ``(reprogram_passes, reuse_passes)`` from
  the ``weight_bank_id`` grouping — N reprogram = #distinct banks, M reuse = rest;
* the per-segment plan also carries ``params_reloaded`` (Σ over reprogram passes of
  the resident bank's weight count) — the quantity the cost term charges per reload;
* a multi-segment :class:`WeightReusePlan` aggregating N / M / params across segments;
* the conv-over-a-big-map shape: one bank, ``n_positions`` cores ⇒ 1 reprogram +
  ``(n_positions - 1)`` reuse passes (the VGG@224 collapse the design quantifies).
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.mapping.ir.graph import IRGraph
from mimarsinan.mapping.ir.types import IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.weight_reuse import (
    SegmentReusePhases,
    WeightReusePlan,
    classify_segment_phases,
    format_weight_reuse_summary,
    weight_reuse_plan_from_graph,
)


def _src(node_id=-2, index=0):
    return IRSource(node_id=node_id, index=index)


def _bank(bank_id, axons, neurons):
    return WeightBank(id=bank_id, core_matrix=np.zeros((axons, neurons), dtype=np.float32))


def _shared_core(core_id, bank_id, n_inputs, out):
    """A bank-backed NeuralCore (one position-softcore referencing a shared bank)."""
    sources = np.array([_src(index=i) for i in range(n_inputs)], dtype=object)
    return NeuralCore(
        id=core_id,
        name=f"core{core_id}",
        input_sources=sources,
        core_matrix=None,
        weight_bank_id=bank_id,
        weight_row_slice=(0, out),
    )


def _owned_core(core_id, axons, neurons):
    """A NeuralCore with its own core_matrix (no shared bank — an FC-style core)."""
    sources = np.array([_src(index=i) for i in range(axons)], dtype=object)
    return NeuralCore(
        id=core_id,
        name=f"owned{core_id}",
        input_sources=sources,
        core_matrix=np.zeros((axons, neurons), dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# classify_segment_phases — the keystone: N reprogram + M reuse from banks.
# --------------------------------------------------------------------------- #

class TestClassifySegmentPhases:
    def test_conv_one_bank_many_positions(self):
        # The conv-over-a-big-map shape: ONE bank, n positions ⇒ 1 reprogram + (n-1)
        # reuse. This is the VGG features_6 collapse: 74 reprogram → 1 + 73 reuse.
        bank = _bank(0, axons=27, neurons=64)
        cores = [_shared_core(i, bank_id=0, n_inputs=27, out=64) for i in range(74)]
        phases = classify_segment_phases(cores, {0: bank})
        assert phases.reprogram_passes == 1
        assert phases.reuse_passes == 73
        assert phases.total_passes == 74

    def test_no_cores_is_no_passes(self):
        phases = classify_segment_phases([], {})
        assert phases.reprogram_passes == 0
        assert phases.reuse_passes == 0
        assert phases.total_passes == 0

    def test_single_pass_is_one_reprogram_zero_reuse(self):
        bank = _bank(0, 27, 64)
        phases = classify_segment_phases([_shared_core(0, 0, 27, 64)], {0: bank})
        assert phases.reprogram_passes == 1
        assert phases.reuse_passes == 0

    def test_two_distinct_banks_are_two_reprogram(self):
        # output_tiled conv: two banks (the channel groups). Each distinct bank is a
        # reprogram; positions within a group reuse.
        banks = {0: _bank(0, 27, 32), 1: _bank(1, 27, 32)}
        cores = (
            [_shared_core(i, 0, 27, 32) for i in range(10)]
            + [_shared_core(100 + i, 1, 27, 32) for i in range(10)]
        )
        phases = classify_segment_phases(cores, banks)
        assert phases.reprogram_passes == 2
        assert phases.reuse_passes == 18
        assert phases.total_passes == 20

    def test_owned_core_is_its_own_reprogram(self):
        # An FC core with its OWN core_matrix (no shared bank) cannot be reused — it
        # is a distinct reprogram pass (one per owned core).
        cores = [_owned_core(0, 64, 32), _owned_core(1, 64, 32)]
        phases = classify_segment_phases(cores, {})
        assert phases.reprogram_passes == 2
        assert phases.reuse_passes == 0

    def test_mixed_owned_and_shared(self):
        bank = _bank(0, 27, 64)
        cores = (
            [_shared_core(i, 0, 27, 64) for i in range(5)]  # 1 reprogram + 4 reuse
            + [_owned_core(50, 64, 32)]                       # 1 reprogram
        )
        phases = classify_segment_phases(cores, {0: bank})
        assert phases.reprogram_passes == 2
        assert phases.reuse_passes == 4
        assert phases.total_passes == 6

    def test_params_reloaded_counts_resident_bank_weights_once(self):
        # params_reloaded charges the resident bank ONCE (the reprogram), not once per
        # reused position — that is the whole point of the mode.
        bank = _bank(0, axons=27, neurons=64)  # 27*64 = 1728 weights
        cores = [_shared_core(i, 0, 27, 64) for i in range(74)]
        phases = classify_segment_phases(cores, {0: bank})
        assert phases.params_reloaded == 27 * 64

    def test_params_reloaded_sums_over_distinct_banks(self):
        banks = {0: _bank(0, 27, 32), 1: _bank(1, 27, 32)}
        cores = (
            [_shared_core(i, 0, 27, 32) for i in range(10)]
            + [_shared_core(100 + i, 1, 27, 32) for i in range(10)]
        )
        phases = classify_segment_phases(cores, banks)
        assert phases.params_reloaded == 27 * 32 + 27 * 32

    def test_params_reloaded_counts_owned_core_matrix(self):
        cores = [_owned_core(0, 64, 32)]  # owns 64*32 weights
        phases = classify_segment_phases(cores, {})
        assert phases.params_reloaded == 64 * 32

    def test_reuse_fraction(self):
        bank = _bank(0, 27, 64)
        cores = [_shared_core(i, 0, 27, 64) for i in range(100)]
        phases = classify_segment_phases(cores, {0: bank})
        assert phases.reuse_fraction == pytest.approx(0.99)

    def test_reuse_fraction_zero_passes(self):
        assert classify_segment_phases([], {}).reuse_fraction == 0.0


# --------------------------------------------------------------------------- #
# WeightReusePlan — aggregate N / M / params across a graph's segments.
# --------------------------------------------------------------------------- #

class TestWeightReusePlanFromGraph:
    def _conv_segment_graph(self):
        # One conv: bank 0, 74 position-softcores (a maximal reuse run).
        bank = _bank(0, axons=27, neurons=64)
        cores = [_shared_core(i, 0, 27, 64) for i in range(74)]
        graph = IRGraph(
            nodes=list(cores),
            output_sources=np.array([_src(node_id=0, index=0)], dtype=object),
            weight_banks={0: bank},
        )
        return graph

    def test_single_conv_collapse(self):
        plan = weight_reuse_plan_from_graph(self._conv_segment_graph())
        assert plan.reprogram_passes == 1
        assert plan.reuse_passes == 73
        assert plan.total_passes == 74
        assert plan.params_reloaded == 27 * 64

    def test_two_conv_graph_aggregates(self):
        b0, b1 = _bank(0, 27, 64), _bank(1, 64 * 9, 128)
        conv0 = [_shared_core(i, 0, 27, 64) for i in range(74)]
        conv1 = [_shared_core(200 + i, 1, 64 * 9, 128) for i in range(20)]
        graph = IRGraph(
            nodes=conv0 + conv1,
            output_sources=np.array([_src(node_id=0, index=0)], dtype=object),
            weight_banks={0: b0, 1: b1},
        )
        plan = weight_reuse_plan_from_graph(graph)
        # N = 2 distinct banks ⇒ 2 reprogram; M = (74-1)+(20-1) = 92 reuse.
        assert plan.reprogram_passes == 2
        assert plan.reuse_passes == 73 + 19
        assert plan.total_passes == 94
        assert plan.params_reloaded == 27 * 64 + 64 * 9 * 128

    def test_empty_graph(self):
        graph = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
        plan = weight_reuse_plan_from_graph(graph)
        assert plan.reprogram_passes == 0
        assert plan.reuse_passes == 0
        assert plan.params_reloaded == 0

    def test_plan_combines_phases(self):
        a = SegmentReusePhases(reprogram_passes=1, reuse_passes=73, params_reloaded=1728)
        b = SegmentReusePhases(reprogram_passes=2, reuse_passes=18, params_reloaded=900)
        combined = WeightReusePlan.from_segments([a, b])
        assert combined.reprogram_passes == 3
        assert combined.reuse_passes == 91
        assert combined.params_reloaded == 2628
        assert combined.total_passes == 94

    def test_sync_barrier_count_is_total_passes_minus_one(self):
        # Every pass writes its positions at the segment-exit sync point; the number
        # of sync barriers between passes is total_passes - 1 (the design's (M+N-1)).
        plan = weight_reuse_plan_from_graph(self._conv_segment_graph())
        assert plan.sync_barrier_count == plan.total_passes - 1


# --------------------------------------------------------------------------- #
# format_weight_reuse_summary — the SCM-gate report ("N reprogram + M reuse").
# --------------------------------------------------------------------------- #

class TestFormatSummary:
    def test_summary_states_n_reprogram_m_reuse(self):
        plan = WeightReusePlan(reprogram_passes=16, reuse_passes=142, params_reloaded=0)
        summary = format_weight_reuse_summary(plan)
        assert "16 reprogram" in summary
        assert "142 reuse" in summary
        assert "158" in summary  # total passes

    def test_summary_handles_empty(self):
        plan = WeightReusePlan(reprogram_passes=0, reuse_passes=0, params_reloaded=0)
        summary = format_weight_reuse_summary(plan)
        assert "0 reprogram" in summary
        assert "0 reuse" in summary


# --------------------------------------------------------------------------- #
# VGG16@224 quantification — the headline collapse: per-conv positions become a
# single reprogram + many reuse passes, so the schedule's reprogram reload events
# drop ~10× (the design's N≈16 / M≈142). Composed from per-segment phase math (not
# 700k materialised softcores — the unit under test is the classification, and one
# segment's split is fully determined by (#banks, #positions)).
# --------------------------------------------------------------------------- #

class TestVgg224Quantification:
    # VGG16 conv stack at 224×224 (spatial map per conv block, post the pools that
    # precede each block) and the 3 FC layers. (out_channels, h*w positions) per conv.
    _VGG_CONVS = [
        (64, 224 * 224), (64, 224 * 224),
        (128, 112 * 112), (128, 112 * 112),
        (256, 56 * 56), (256, 56 * 56), (256, 56 * 56),
        (512, 28 * 28), (512, 28 * 28), (512, 28 * 28),
        (512, 14 * 14), (512, 14 * 14), (512, 14 * 14),
    ]  # 13 convs

    def _vgg_plan(self):
        # One conv = one shared bank over ``positions`` softcores ⇒ 1 reprogram +
        # (positions - 1) reuse, params_reloaded = one bank (out_ch*9 axons × out_ch).
        segments = [
            SegmentReusePhases(
                reprogram_passes=1,
                reuse_passes=positions - 1,
                params_reloaded=out_ch * 9 * out_ch,
            )
            for out_ch, positions in self._VGG_CONVS
        ]
        # 3 FC layers: each an owned-core reprogram, no reuse.
        segments += [
            SegmentReusePhases(reprogram_passes=1, reuse_passes=0, params_reloaded=4096 * 4096)
            for _ in range(3)
        ]
        return WeightReusePlan.from_segments(segments)

    def test_reprogram_phases_collapse_to_distinct_weights(self):
        plan = self._vgg_plan()
        # N = 13 convs + 3 FC = 16 weight-distinct reprogram phases (the design's N≈16).
        assert plan.reprogram_passes == 16

    def test_most_passes_become_reuse(self):
        plan = self._vgg_plan()
        total_positions = sum(p for _, p in self._VGG_CONVS) + 3
        assert plan.total_passes == total_positions
        # M = total - N; with positions in the hundreds-of-thousands, ~all passes reuse.
        assert plan.reuse_passes == total_positions - 16
        assert plan.reuse_fraction > 0.999

    def test_reprogram_reload_events_drop_versus_per_position(self):
        # Pre-mode every position is its own reprogram (total_passes reloads); the mode
        # collapses to N=16 — a >1000× drop in reprogram reload events for VGG@224.
        plan = self._vgg_plan()
        per_position_reprograms = plan.total_passes
        assert plan.reprogram_passes < per_position_reprograms / 1000

    def test_widest_conv_segment_collapses_to_one_reprogram(self):
        # The widest-segment collapse, materialised on a representative position count:
        # n shared-bank position-softcores ⇒ n reprograms → 1 reprogram + (n-1) reuse
        # (the design's features_6: 50176 reprograms → 1 + 50175 reuse, same shape).
        n_positions = 4096
        bank = _bank(0, axons=64 * 9, neurons=64)
        cores = [_shared_core(i, 0, 64 * 9, 64) for i in range(n_positions)]
        phases = classify_segment_phases(cores, {0: bank})
        assert phases.reprogram_passes == 1
        assert phases.reuse_passes == n_positions - 1
