"""SANA-FE replays a carried raster, so a scheduled segment measures the real program.

SANA-FE is the cost-measuring backend: if it collapsed a pass boundary it would model a
different computation from the one the record claims, and its energy number would be for
the wrong chip. Both primitives it needs already existed — a per-neuron trace out and an
arbitrary per-axon train in — so carrying is a gather, not a new capability.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.models.spiking.hybrid.carry import (
    apply_carried_input,
    pass_transfer_for_backend,
)
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM


class _Slice:
    def __init__(self, node_id, offset, size):
        self.node_id, self.offset, self.size = node_id, offset, size


class _Stage:
    def __init__(self, inputs):
        self.input_map = [_Slice(n, o, s) for n, o, s in inputs]


class TestSanafeCarriesVerbatim:
    def test_sanafe_is_declared_a_verbatim_backend(self):
        assert pass_transfer_for_backend("sanafe") == VERBATIM


class TestReplayingACarriedTrain:
    """``encoded`` is (1, size, T); a carried raster is time-first (T, size)."""

    def _encoded(self, size, T):
        return np.zeros((1, size, T), dtype=np.float32)

    def test_a_published_train_overwrites_its_own_slice(self):
        encoded = self._encoded(4, 3)
        train = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        assert apply_carried_input(encoded, _Stage([(7, 1, 2)]), {7: train})
        assert encoded[0, 1, :].tolist() == [1, 0, 1]
        assert encoded[0, 2, :].tolist() == [0, 1, 1]

    def test_slices_without_a_published_train_are_left_re_encoded(self):
        """A host boundary in the same input map keeps its uniform encode."""
        encoded = self._encoded(4, 3)
        encoded[0, 3, :] = 1.0
        apply_carried_input(encoded, _Stage([(7, 0, 1), (-2, 3, 1)]),
                            {7: np.ones((3, 1), dtype=np.float32)})
        assert encoded[0, 3, :].tolist() == [1, 1, 1]

    def test_nothing_published_means_nothing_replayed(self):
        encoded = self._encoded(4, 3)
        assert apply_carried_input(encoded, _Stage([(7, 0, 2)]), {}) is False
        assert encoded.sum() == 0

    def test_the_transpose_is_not_silently_wrong(self):
        """A (T, size) raster read as (size, T) would still 'work' on a square
        window and be wrong everywhere else, so the shapes are pinned asymmetric."""
        encoded = self._encoded(2, 4)
        train = np.array([[1, 0], [0, 0], [0, 0], [0, 1]], dtype=np.float32)
        apply_carried_input(encoded, _Stage([(7, 0, 2)]), {7: train})
        assert encoded[0, 0, :].tolist() == [1, 0, 0, 0]
        assert encoded[0, 1, :].tolist() == [0, 0, 0, 1]


class _Gather:
    """The mixin method under test, bound to nothing else it needs."""

    from mimarsinan.chip_simulation.sanafe.runner.segment_io import (
        SanafeSegmentIOMixin as _Mixin,
    )
    _compute_seg_output_raster = _Mixin._compute_seg_output_raster


class TestTheSanafeGather:
    """The produce side: SANA-FE's trace is (neurons, T) in ABSOLUTE cycle time and
    net-group row order; a carried raster is (T, out_dim) in PRODUCER-LOCAL time and
    segment-output order. Getting either wrong hands the next pass someone else's
    spikes, or the right spikes at the wrong moment."""

    def _sources(self, specs):
        from mimarsinan.code_generation.cpp_chip_model_types import SpikeSource

        return np.array([SpikeSource(**spec) for spec in specs], dtype=object)

    def test_a_core_span_is_gathered_per_cycle(self):
        # Core 0 neurons 0..1; trace rows 0..1 over 4 absolute cycles.
        raster = _Gather()._compute_seg_output_raster(
            self._sources([{"core": 0, "neuron": 0}, {"core": 0, "neuron": 1}]),
            seg_raster=np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.uint8),
            core_rows={0: 0}, core_latency={0: 0}, T=4,
        )
        assert raster.tolist() == [[1, 0], [0, 1], [1, 1], [0, 0]]

    def test_the_producers_latency_sets_local_time_zero(self):
        """A core at latency 1 emits its FIRST spike at absolute cycle 1, and the
        consuming pass reads that as its own step 0."""
        raster = _Gather()._compute_seg_output_raster(
            self._sources([{"core": 0, "neuron": 0}]),
            seg_raster=np.array([[9, 1, 0, 1]], dtype=np.uint8),
            core_rows={0: 0}, core_latency={0: 1}, T=3,
        )
        assert raster[:, 0].tolist() == [1, 0, 1], "cycle 0 belongs to no local step"

    def test_an_always_on_source_spikes_every_local_step(self):
        raster = _Gather()._compute_seg_output_raster(
            self._sources([{"core": 0, "neuron": 0, "is_always_on": True}]),
            seg_raster=np.zeros((1, 3), dtype=np.uint8),
            core_rows={0: 0}, core_latency={0: 0}, T=3,
        )
        assert raster[:, 0].tolist() == [1, 1, 1]

    def test_an_off_source_never_spikes(self):
        raster = _Gather()._compute_seg_output_raster(
            self._sources([{"core": 0, "neuron": 0, "is_off": True}]),
            seg_raster=np.ones((1, 3), dtype=np.uint8),
            core_rows={0: 0}, core_latency={0: 0}, T=3,
        )
        assert raster[:, 0].tolist() == [0, 0, 0]

    def test_no_trace_means_no_carry_rather_than_a_zero_one(self):
        assert _Gather()._compute_seg_output_raster(
            self._sources([{"core": 0, "neuron": 0}]),
            seg_raster=None, core_rows={0: 0}, core_latency={0: 0}, T=3,
        ) is None
