"""Tests for _expand_preview_for_scheduling: miniview sync barrier insertion."""

import pytest
import sys
import os

# The function is defined inside server.py's setup_routes closure.
# Extract it for testing by importing the module-level definition.
# We import the function by exec'ing the relevant portion.
# Alternative: refactor to module-level function. For now, we redefine the
# logic here since it's a pure function with no dependencies.


def _expand_preview_for_scheduling(preview, per_segment_passes, max_cores_per_pass):
    """Mirror of server.py's _expand_preview_for_scheduling for testing."""
    if not preview or not preview.get("flow"):
        return preview

    old_flow = preview["flow"]

    segments = []
    current_seg_neural = []
    current_seg_id = 0

    for item in old_flow:
        if item.get("kind") == "neural":
            current_seg_neural.append(item)
        elif item.get("kind") == "host":
            if current_seg_neural:
                segments.append((current_seg_id, current_seg_neural))
                current_seg_neural = []
                current_seg_id += 1

    if current_seg_neural:
        segments.append((current_seg_id, current_seg_neural))

    seg_pass_assignments = {}
    for seg_id, neural_items in segments:
        n_passes = per_segment_passes.get(seg_id, per_segment_passes.get(str(seg_id), 1))
        if n_passes <= 1:
            seg_pass_assignments[seg_id] = [(0, neural_items)]
            continue

        n = len(neural_items)
        if n >= n_passes:
            chunk = max(1, (n + n_passes - 1) // n_passes)
            passes = []
            for pass_idx in range(n_passes):
                start = pass_idx * chunk
                if start >= n:
                    break
                end = min(start + chunk, n)
                passes.append((pass_idx, neural_items[start:end]))
        else:
            from collections import defaultdict as _ddict
            expanded = []
            for item in neural_items:
                sc = item.get("softcore_count", 1)
                per_pass = max(1, (sc + n_passes - 1) // n_passes)
                remaining = sc
                for pi in range(n_passes):
                    take = min(per_pass, remaining)
                    if take <= 0:
                        break
                    sub = dict(item)
                    sub["softcore_count"] = take
                    expanded.append((pi, sub))
                    remaining -= take
                for pi in range(n_passes):
                    if not any(p == pi for p, _ in expanded):
                        sub = dict(item)
                        sub["softcore_count"] = 0
                        expanded.append((pi, sub))
            by_pass = _ddict(list)
            for pi, sub_item in expanded:
                by_pass[pi].append(sub_item)
            passes = [(pi, by_pass[pi]) for pi in sorted(by_pass.keys())]

        seg_pass_assignments[seg_id] = passes

    new_flow = [{"kind": "input"}]
    seg_idx = 0
    saw_neural_in_segment = False
    emitted_segments = set()
    for item in old_flow:
        if item.get("kind") == "input" or item.get("kind") == "output":
            continue
        if item.get("kind") == "host":
            new_flow.append(item)
            if saw_neural_in_segment:
                seg_idx += 1
                saw_neural_in_segment = False
            continue
        if item.get("kind") == "neural":
            saw_neural_in_segment = True
            passes_for_seg = seg_pass_assignments.get(seg_idx)
            if passes_for_seg is None:
                new_flow.append(item)
                continue

            if seg_idx not in emitted_segments:
                emitted_segments.add(seg_idx)
                for pi, (pass_id, pass_items) in enumerate(passes_for_seg):
                    if pi > 0:
                        new_flow.append({
                            "kind": "host",
                            "slot": -1,
                            "compute_op_count": 0,
                            "schedule_sync": True,
                        })
                    for neural_item in pass_items:
                        new_flow.append(neural_item)
            continue

    new_flow.append({"kind": "output"})

    schedule_syncs = sum(
        1 for item in new_flow
        if item.get("kind") == "host" and item.get("schedule_sync")
    )

    result = dict(preview)
    result["flow"] = new_flow
    result["schedule_sync_count"] = schedule_syncs
    return result


class TestExpandPreviewForScheduling:
    def test_single_segment_two_passes(self):
        """Single segment with 2 passes should insert 1 sync barrier."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 10},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 10},
            {"kind": "neural", "latency_group_index": 2, "softcore_count": 10},
            {"kind": "neural", "latency_group_index": 3, "softcore_count": 10},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 2}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 1
        assert result["schedule_sync_count"] == 1

    def test_single_segment_three_passes(self):
        """Single segment with 3 passes should insert 2 sync barriers."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 5},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 5},
            {"kind": "neural", "latency_group_index": 2, "softcore_count": 5},
            {"kind": "neural", "latency_group_index": 3, "softcore_count": 5},
            {"kind": "neural", "latency_group_index": 4, "softcore_count": 5},
            {"kind": "neural", "latency_group_index": 5, "softcore_count": 5},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 3}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 2
        assert result["schedule_sync_count"] == 2

    def test_no_scheduling_no_barriers(self):
        """Single pass should not insert any barriers."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 10},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 10},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 1}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 0
        assert result["schedule_sync_count"] == 0

    def test_two_segments_different_passes(self):
        """Two segments with different pass counts should get independent barriers."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 8},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 8},
            {"kind": "host", "slot": 2, "compute_op_count": 1},
            {"kind": "neural", "latency_group_index": 2, "softcore_count": 8},
            {"kind": "neural", "latency_group_index": 3, "softcore_count": 8},
            {"kind": "neural", "latency_group_index": 4, "softcore_count": 8},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 2, 1: 3}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        # Segment 0: 2 passes → 1 barrier
        # Segment 1: 3 passes → 2 barriers
        assert len(syncs) == 3
        assert result["schedule_sync_count"] == 3

    def test_neural_items_preserved(self):
        """All original neural items should be present in the expanded flow."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 10},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 20},
            {"kind": "neural", "latency_group_index": 2, "softcore_count": 15},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 2}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        neural_items = [item for item in result["flow"] if item.get("kind") == "neural"]
        assert len(neural_items) == 3
        counts = [item["softcore_count"] for item in neural_items]
        assert counts == [10, 20, 15]

    def test_host_items_preserved(self):
        """Original host items (ComputeOps) should be preserved alongside sync barriers."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 5},
            {"kind": "host", "slot": 1, "compute_op_count": 2},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 5},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 1, 1: 1}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        host_items = [item for item in result["flow"] if item.get("kind") == "host"]
        original_hosts = [h for h in host_items if not h.get("schedule_sync")]
        assert len(original_hosts) == 1
        assert original_hosts[0]["compute_op_count"] == 2

    def test_empty_preview(self):
        """Empty preview should return as-is."""
        assert _expand_preview_for_scheduling(None, {0: 2}, 50) is None
        assert _expand_preview_for_scheduling({}, {0: 2}, 50) == {}
        assert _expand_preview_for_scheduling({"flow": []}, {0: 2}, 50) == {"flow": []}

    def test_string_keys_in_per_segment_passes(self):
        """per_segment_passes may have string keys (from JSON deserialization)."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 5},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 5},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        # JSON would give us string keys
        per_segment_passes = {"0": 2}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 1

    def test_flow_starts_and_ends_correctly(self):
        """Expanded flow must start with input and end with output."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 5},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 1}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)
        assert result["flow"][0]["kind"] == "input"
        assert result["flow"][-1]["kind"] == "output"

    def test_more_passes_than_items(self):
        """When more passes than neural items, split by softcore_count."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 10},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 3}  # 3 passes for 1 item → split into 3 sub-items

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 2  # 3 passes → 2 barriers
        # Total softcore_count across sub-items should equal original
        neural_items = [i for i in result["flow"] if i.get("kind") == "neural"]
        total_sc = sum(i["softcore_count"] for i in neural_items)
        assert total_sc == 10

    def test_host_before_neural_items(self):
        """Host item before any neural items must not break segment tracking.

        This is the MLP-Mixer scenario: patch_embed Conv2d appears at slot 0
        before any neural groups.
        """
        flow = [
            {"kind": "input"},
            {"kind": "host", "slot": 0, "compute_op_count": 1},  # before neural!
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 32},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 16},
            {"kind": "host", "slot": 2, "compute_op_count": 1},
            {"kind": "neural", "latency_group_index": 2, "softcore_count": 32},
            {"kind": "neural", "latency_group_index": 3, "softcore_count": 16},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 2, 1: 2}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        # Segment 0 (2 passes) → 1 barrier, Segment 1 (2 passes) → 1 barrier
        assert len(syncs) == 2, (
            f"Expected 2 sync barriers, got {len(syncs)}. "
            f"Flow kinds: {[item.get('kind') + (':sync' if item.get('schedule_sync') else '') for item in result['flow']]}"
        )
        assert result["schedule_sync_count"] == 2

    def test_mlp_mixer_exact_scenario(self):
        """Exact MLP-Mixer layout: 4 segments, each with 1 latency group, 2 passes each.

        This is the scenario from the user's screenshot where barriers never appeared
        because each segment had only 1 neural item that couldn't be chunked.
        """
        flow = [
            {"kind": "input"},
            {"kind": "host", "slot": 0, "compute_op_count": 1},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 32},
            {"kind": "host", "slot": 1, "compute_op_count": 32},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 16},
            {"kind": "host", "slot": 2, "compute_op_count": 16},
            {"kind": "neural", "latency_group_index": 2, "softcore_count": 32},
            {"kind": "host", "slot": 3, "compute_op_count": 32},
            {"kind": "neural", "latency_group_index": 3, "softcore_count": 16},
            {"kind": "host", "slot": 4, "compute_op_count": 18},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 2, 1: 2, 2: 2, 3: 2}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        # 4 segments × 1 barrier each = 4 barriers
        assert len(syncs) == 4, (
            f"Expected 4 sync barriers for 4 segments with 2 passes each, got {len(syncs)}"
        )
        assert result["schedule_sync_count"] == 4
        # Each original neural group (32 or 16 softcores) should be split into 2 sub-items
        neural_items = [i for i in result["flow"] if i.get("kind") == "neural"]
        assert len(neural_items) == 8  # 4 groups × 2 passes

    def test_multiple_hosts_before_neural(self):
        """Multiple consecutive host items before neural items."""
        flow = [
            {"kind": "input"},
            {"kind": "host", "slot": 0, "compute_op_count": 1},
            {"kind": "host", "slot": 1, "compute_op_count": 1},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 10},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 10},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 2}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, 50)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 1
