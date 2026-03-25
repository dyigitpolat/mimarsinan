"""Tests for _expand_preview_for_scheduling: miniview sync barrier insertion.

The function under test is defined inside server.py's setup_routes closure.
We redefine the logic here since it's a pure function with no dependencies.
The signature now takes per_segment_pass_lists (actual partitioner output)
instead of just an integer max_cores_per_pass.
"""

import pytest


def _expand_preview_for_scheduling(preview, per_segment_passes, per_segment_pass_lists):
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
        pass_lists = per_segment_pass_lists.get(seg_id, per_segment_pass_lists.get(str(seg_id)))

        if n_passes <= 1 or not pass_lists or len(pass_lists) <= 1:
            seg_pass_assignments[seg_id] = [(0, neural_items)]
            continue

        passes = []
        for pi, pass_list in enumerate(pass_lists):
            if neural_items:
                template = dict(neural_items[0])
            else:
                template = {"kind": "neural", "latency_group_index": 0, "latency_tag": 0, "softcore_count": 0, "segment_count": 1}
            template["softcore_count"] = len(pass_list)
            passes.append((pi, [template]))

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


def _make_pass_lists(per_segment_passes, segment_softcores):
    """Build per_segment_pass_lists from pass counts and total softcores.

    Creates fake pass list entries where each pass gets a proportional share.
    """
    result = {}
    for seg_id, n_passes in per_segment_passes.items():
        total = segment_softcores.get(seg_id, 10)
        if n_passes <= 1:
            result[seg_id] = [list(range(total))]
        else:
            chunk = max(1, (total + n_passes - 1) // n_passes)
            lists = []
            remaining = total
            for _ in range(n_passes):
                take = min(chunk, remaining)
                lists.append(list(range(take)))
                remaining -= take
            result[seg_id] = lists
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
        pass_lists = _make_pass_lists(per_segment_passes, {0: 40})

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

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
        pass_lists = _make_pass_lists(per_segment_passes, {0: 30})

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

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
        pass_lists = _make_pass_lists(per_segment_passes, {0: 20})

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

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
        pass_lists = _make_pass_lists(per_segment_passes, {0: 16, 1: 24})

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 3
        assert result["schedule_sync_count"] == 3

    def test_neural_items_softcore_count_preserved(self):
        """Total softcore_count should be preserved when scheduling expands items."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 10},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 20},
            {"kind": "neural", "latency_group_index": 2, "softcore_count": 15},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 2}
        pass_lists = _make_pass_lists(per_segment_passes, {0: 45})

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

        neural_items = [item for item in result["flow"] if item.get("kind") == "neural"]
        total = sum(item["softcore_count"] for item in neural_items)
        assert total == 45

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
        pass_lists = _make_pass_lists(per_segment_passes, {0: 5, 1: 5})

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

        host_items = [item for item in result["flow"] if item.get("kind") == "host"]
        original_hosts = [h for h in host_items if not h.get("schedule_sync")]
        assert len(original_hosts) == 1
        assert original_hosts[0]["compute_op_count"] == 2

    def test_empty_preview(self):
        """Empty preview should return as-is."""
        assert _expand_preview_for_scheduling(None, {0: 2}, {}) is None
        assert _expand_preview_for_scheduling({}, {0: 2}, {}) == {}
        assert _expand_preview_for_scheduling({"flow": []}, {0: 2}, {}) == {"flow": []}

    def test_string_keys_in_per_segment_passes(self):
        """per_segment_passes may have string keys (from JSON deserialization)."""
        flow = [
            {"kind": "input"},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 5},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 5},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {"0": 2}
        pass_lists = {"0": [list(range(5)), list(range(5))]}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

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
        pass_lists = _make_pass_lists(per_segment_passes, {0: 5})

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)
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
        per_segment_passes = {0: 3}
        pass_lists = {0: [list(range(4)), list(range(3)), list(range(3))]}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 2
        neural_items = [i for i in result["flow"] if i.get("kind") == "neural"]
        total_sc = sum(i["softcore_count"] for i in neural_items)
        assert total_sc == 10

    def test_host_before_neural_items(self):
        """Host item before any neural items must not break segment tracking."""
        flow = [
            {"kind": "input"},
            {"kind": "host", "slot": 0, "compute_op_count": 1},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 32},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 16},
            {"kind": "host", "slot": 2, "compute_op_count": 1},
            {"kind": "neural", "latency_group_index": 2, "softcore_count": 32},
            {"kind": "neural", "latency_group_index": 3, "softcore_count": 16},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 2, 1: 2}
        pass_lists = _make_pass_lists(per_segment_passes, {0: 48, 1: 48})

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 2, (
            f"Expected 2 sync barriers, got {len(syncs)}. "
            f"Flow kinds: {[item.get('kind') + (':sync' if item.get('schedule_sync') else '') for item in result['flow']]}"
        )
        assert result["schedule_sync_count"] == 2

    def test_mlp_mixer_exact_scenario(self):
        """Exact MLP-Mixer layout: 4 segments, each with 1 latency group, 2 passes each."""
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
        pass_lists = _make_pass_lists(per_segment_passes, {0: 32, 1: 16, 2: 32, 3: 16})

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 4, (
            f"Expected 4 sync barriers for 4 segments with 2 passes each, got {len(syncs)}"
        )
        assert result["schedule_sync_count"] == 4
        neural_items = [i for i in result["flow"] if i.get("kind") == "neural"]
        assert len(neural_items) == 8

    def test_mlp_mixer_no_zero_softcore_passes(self):
        """Regression: MLP-Mixer seg0 with 32 softcores / 100-core budget.

        The old remaining_sc capping logic would drain the budget on the
        first pass, leaving subsequent passes with 0 softcores.  The fix
        uses the partitioner's authoritative counts directly.
        """
        flow = [
            {"kind": "input"},
            {"kind": "host", "slot": 0, "compute_op_count": 1},
            {"kind": "neural", "latency_group_index": 0, "softcore_count": 32},
            {"kind": "host", "slot": 1, "compute_op_count": 32},
            {"kind": "neural", "latency_group_index": 1, "softcore_count": 16},
            {"kind": "output"},
        ]
        preview = {"flow": flow}
        per_segment_passes = {0: 2, 1: 1}
        # Partitioner says seg0 splits into 20 + 12 softcores across 2 passes.
        # Seg1 stays at 16 in a single pass.
        pass_lists = {0: [list(range(20)), list(range(12))], 1: [list(range(16))]}

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

        neural_items = [i for i in result["flow"] if i.get("kind") == "neural"]
        for ni in neural_items:
            assert ni["softcore_count"] > 0, (
                f"Pass with 0 softcores found! softcore_count={ni['softcore_count']}, "
                f"latency_group_index={ni.get('latency_group_index')}"
            )
        # Seg0 should have 2 neural items (20 + 12), seg1 should have 1 (16)
        assert len(neural_items) == 3
        assert neural_items[0]["softcore_count"] == 20
        assert neural_items[1]["softcore_count"] == 12
        assert neural_items[2]["softcore_count"] == 16

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
        pass_lists = _make_pass_lists(per_segment_passes, {0: 20})

        result = _expand_preview_for_scheduling(preview, per_segment_passes, pass_lists)

        syncs = [item for item in result["flow"] if item.get("schedule_sync")]
        assert len(syncs) == 1
