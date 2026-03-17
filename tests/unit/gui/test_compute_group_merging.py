"""Unit tests for compute-op grouping in IR graph and hardware snapshots."""

import pytest

from mimarsinan.gui.snapshot.builders import (
    _group_consecutive_compute_stages,
    _merge_consecutive_compute_groups,
)


class TestGroupConsecutiveComputeStages:
    """Tests for _group_consecutive_compute_stages (hardware snapshot)."""

    def test_empty_input(self):
        assert _group_consecutive_compute_stages([]) == []

    def test_single_neural_stage_unchanged(self):
        stages = [{"index": 0, "kind": "neural", "name": "seg0"}]
        result = _group_consecutive_compute_stages(stages)
        assert len(result) == 1
        assert result[0]["kind"] == "neural"

    def test_single_compute_stage_unchanged(self):
        stages = [
            {"index": 0, "kind": "compute", "op_type": "gelu", "op_name": "gelu_0",
             "name": "gelu_0", "input_shape": [4], "output_shape": [4]},
        ]
        result = _group_consecutive_compute_stages(stages)
        assert len(result) == 1
        assert result[0]["kind"] == "compute"

    def test_two_consecutive_compute_stages_grouped(self):
        stages = [
            {"index": 0, "kind": "compute", "op_type": "layer_norm", "op_name": "ln_0",
             "name": "ln_0", "input_shape": [8], "output_shape": [8]},
            {"index": 1, "kind": "compute", "op_type": "gelu", "op_name": "gelu_0",
             "name": "gelu_0", "input_shape": [8], "output_shape": [8]},
        ]
        result = _group_consecutive_compute_stages(stages)
        assert len(result) == 1
        assert result[0]["kind"] == "compute_group"
        assert result[0]["num_ops"] == 2
        assert len(result[0]["ops"]) == 2
        assert result[0]["ops"][0]["op_type"] == "layer_norm"
        assert result[0]["ops"][1]["op_type"] == "gelu"
        assert result[0]["is_barrier"] is True

    def test_three_consecutive_compute_stages_grouped(self):
        stages = [
            {"index": 0, "kind": "compute", "op_type": "layer_norm", "op_name": "ln",
             "name": "ln", "input_shape": [4], "output_shape": [4]},
            {"index": 1, "kind": "compute", "op_type": "gelu", "op_name": "gelu",
             "name": "gelu", "input_shape": [4], "output_shape": [4]},
            {"index": 2, "kind": "compute", "op_type": "add", "op_name": "add",
             "name": "add", "input_shape": [4], "output_shape": [4]},
        ]
        result = _group_consecutive_compute_stages(stages)
        assert len(result) == 1
        assert result[0]["kind"] == "compute_group"
        assert result[0]["num_ops"] == 3

    def test_compute_between_neural_stages_isolated(self):
        stages = [
            {"index": 0, "kind": "neural", "name": "seg0"},
            {"index": 1, "kind": "compute", "op_type": "gelu", "op_name": "gelu",
             "name": "gelu", "input_shape": [4], "output_shape": [4]},
            {"index": 2, "kind": "neural", "name": "seg1"},
        ]
        result = _group_consecutive_compute_stages(stages)
        assert len(result) == 3
        assert result[0]["kind"] == "neural"
        assert result[1]["kind"] == "compute"
        assert result[2]["kind"] == "neural"

    def test_mixed_neural_and_compute_groups(self):
        stages = [
            {"index": 0, "kind": "neural", "name": "seg0"},
            {"index": 1, "kind": "compute", "op_type": "ln", "op_name": "ln",
             "name": "ln", "input_shape": None, "output_shape": None},
            {"index": 2, "kind": "compute", "op_type": "gelu", "op_name": "gelu",
             "name": "gelu", "input_shape": None, "output_shape": None},
            {"index": 3, "kind": "neural", "name": "seg1"},
            {"index": 4, "kind": "compute", "op_type": "add", "op_name": "add",
             "name": "add", "input_shape": None, "output_shape": None},
        ]
        result = _group_consecutive_compute_stages(stages)
        assert len(result) == 4
        assert result[0]["kind"] == "neural"
        assert result[1]["kind"] == "compute_group"
        assert result[1]["num_ops"] == 2
        assert result[2]["kind"] == "neural"
        assert result[3]["kind"] == "compute"

    def test_op_types_are_deduplicated_preserving_order(self):
        stages = [
            {"index": 0, "kind": "compute", "op_type": "gelu", "op_name": "g1",
             "name": "g1", "input_shape": None, "output_shape": None},
            {"index": 1, "kind": "compute", "op_type": "gelu", "op_name": "g2",
             "name": "g2", "input_shape": None, "output_shape": None},
        ]
        result = _group_consecutive_compute_stages(stages)
        assert result[0]["op_types"] == ["gelu"]


class TestMergeConsecutiveComputeGroups:
    """Tests for _merge_consecutive_compute_groups (IR graph snapshot)."""

    def test_empty_input(self):
        assert _merge_consecutive_compute_groups([]) == []

    def test_single_neural_group_unchanged(self):
        groups = [{"key": "fc1", "order": 0, "type": "neural", "num_cores": 2,
                   "num_ops": 0, "node_ids": [0, 1], "op_types": []}]
        result = _merge_consecutive_compute_groups(groups)
        assert len(result) == 1
        assert result[0]["type"] == "neural"

    def test_single_compute_group_unchanged(self):
        groups = [{"key": "gelu_0", "order": 0, "type": "compute", "num_cores": 0,
                   "num_ops": 1, "node_ids": [5], "op_types": ["gelu"]}]
        result = _merge_consecutive_compute_groups(groups)
        assert len(result) == 1
        assert result[0]["type"] == "compute"

    def test_two_consecutive_compute_groups_merged(self):
        groups = [
            {"key": "ln_0", "order": 0, "type": "compute", "num_cores": 0,
             "num_ops": 1, "node_ids": [5], "op_types": ["layer_norm"],
             "threshold_range": None, "latency_range": None,
             "axon_range": None, "neuron_range": None},
            {"key": "gelu_0", "order": 1, "type": "compute", "num_cores": 0,
             "num_ops": 1, "node_ids": [6], "op_types": ["gelu"],
             "threshold_range": None, "latency_range": None,
             "axon_range": None, "neuron_range": None},
        ]
        result = _merge_consecutive_compute_groups(groups)
        assert len(result) == 1
        assert result[0]["type"] == "compute_group"
        assert result[0]["num_ops"] == 2
        assert set(result[0]["node_ids"]) == {5, 6}
        assert result[0]["sub_keys"] == ["ln_0", "gelu_0"]
        assert "layer_norm" in result[0]["op_types"]
        assert "gelu" in result[0]["op_types"]

    def test_isolated_compute_not_merged(self):
        groups = [
            {"key": "fc1", "order": 0, "type": "neural", "num_cores": 1,
             "num_ops": 0, "node_ids": [0], "op_types": []},
            {"key": "gelu", "order": 1, "type": "compute", "num_cores": 0,
             "num_ops": 1, "node_ids": [1], "op_types": ["gelu"]},
            {"key": "fc2", "order": 2, "type": "neural", "num_cores": 1,
             "num_ops": 0, "node_ids": [2], "op_types": []},
        ]
        result = _merge_consecutive_compute_groups(groups)
        assert len(result) == 3
        assert result[1]["type"] == "compute"

    def test_order_is_renumbered(self):
        groups = [
            {"key": "fc1", "order": 0, "type": "neural", "num_cores": 1,
             "num_ops": 0, "node_ids": [0], "op_types": []},
            {"key": "ln", "order": 1, "type": "compute", "num_cores": 0,
             "num_ops": 1, "node_ids": [1], "op_types": ["layer_norm"],
             "threshold_range": None, "latency_range": None,
             "axon_range": None, "neuron_range": None},
            {"key": "gelu", "order": 2, "type": "compute", "num_cores": 0,
             "num_ops": 1, "node_ids": [2], "op_types": ["gelu"],
             "threshold_range": None, "latency_range": None,
             "axon_range": None, "neuron_range": None},
            {"key": "fc2", "order": 3, "type": "neural", "num_cores": 1,
             "num_ops": 0, "node_ids": [3], "op_types": []},
        ]
        result = _merge_consecutive_compute_groups(groups)
        assert len(result) == 3
        orders = [g["order"] for g in result]
        assert orders == [0, 1, 2]

    def test_virtual_groups_not_merged_with_compute(self):
        groups = [
            {"key": "input", "order": 0, "type": "virtual", "num_cores": 0,
             "num_ops": 0, "node_ids": [], "op_types": []},
            {"key": "ln", "order": 1, "type": "compute", "num_cores": 0,
             "num_ops": 1, "node_ids": [1], "op_types": ["layer_norm"],
             "threshold_range": None, "latency_range": None,
             "axon_range": None, "neuron_range": None},
            {"key": "gelu", "order": 2, "type": "compute", "num_cores": 0,
             "num_ops": 1, "node_ids": [2], "op_types": ["gelu"],
             "threshold_range": None, "latency_range": None,
             "axon_range": None, "neuron_range": None},
        ]
        result = _merge_consecutive_compute_groups(groups)
        assert len(result) == 2
        assert result[0]["type"] == "virtual"
        assert result[1]["type"] == "compute_group"

    def test_merged_key_contains_sub_keys(self):
        groups = [
            {"key": "a", "order": 0, "type": "compute", "num_cores": 0,
             "num_ops": 1, "node_ids": [10], "op_types": ["add"],
             "threshold_range": None, "latency_range": None,
             "axon_range": None, "neuron_range": None},
            {"key": "b", "order": 1, "type": "compute", "num_cores": 0,
             "num_ops": 2, "node_ids": [11, 12], "op_types": ["gelu"],
             "threshold_range": None, "latency_range": None,
             "axon_range": None, "neuron_range": None},
        ]
        result = _merge_consecutive_compute_groups(groups)
        assert result[0]["key"] == "a + b"
        assert result[0]["num_ops"] == 3
        assert result[0]["node_ids"] == [10, 11, 12]
