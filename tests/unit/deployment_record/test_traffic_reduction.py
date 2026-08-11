"""Boundary-traffic reduction: {node: (B, n)} -> totals/maxima, eagerly (W4 stage 2)."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from mimarsinan.deployment_record.build.from_certificates import (
    boundary_traffic_from_node_counts,
)
from mimarsinan.deployment_record.schema import BoundaryTrafficRecord


class TestReduction:
    def test_totals_and_maxima_are_exact(self):
        counts = {
            7: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            3: torch.tensor([[0.0, 5.0, 1.0]]),
        }
        records = boundary_traffic_from_node_counts(counts)
        assert [r.node_id for r in records] == [3, 7]  # sorted by node id
        by_node = {r.node_id: r for r in records}
        assert by_node[7] == BoundaryTrafficRecord(
            node_id=7, producing_stage_index=None,
            neurons=2, samples=2, total_count=10, max_neuron_count=4,
        )
        assert by_node[3] == BoundaryTrafficRecord(
            node_id=3, producing_stage_index=None,
            neurons=3, samples=1, total_count=6, max_neuron_count=5,
        )

    def test_stage_indices_thread_through_when_known(self):
        counts = {1: torch.ones(2, 2), 9: torch.ones(2, 2)}
        records = boundary_traffic_from_node_counts(
            counts, stage_index_by_node={1: 4},
        )
        by_node = {r.node_id: r for r in records}
        assert by_node[1].producing_stage_index == 4
        assert by_node[9].producing_stage_index is None  # unknown stays None

    def test_empty_tensor_reduces_to_zero_counts(self):
        records = boundary_traffic_from_node_counts({0: torch.zeros(0, 4)})
        assert records[0].samples == 0
        assert records[0].total_count == 0
        assert records[0].max_neuron_count == 0

    def test_wrong_rank_fails_loud(self):
        with pytest.raises(ValueError, match=r"\(B, n\)"):
            boundary_traffic_from_node_counts({0: torch.zeros(3)})

    def test_reduction_is_eager_and_retains_no_tensor(self):
        tensor = torch.tensor([[1.0, 2.0]])
        ref = weakref.ref(tensor)
        records = boundary_traffic_from_node_counts({5: tensor})
        del tensor
        gc.collect()
        assert ref() is None, "the reduction must not retain the raw tensor"
        assert records[0].total_count == 3

    def test_record_fields_are_plain_ints(self):
        (record,) = boundary_traffic_from_node_counts({2: torch.ones(3, 4)})
        for value in (record.node_id, record.neurons, record.samples,
                      record.total_count, record.max_neuron_count):
            assert type(value) is int
