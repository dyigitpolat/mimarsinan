"""Coalescing groups must not be split across scheduled hybrid passes."""

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.schedule_partitioner import split_softcores_by_capacity


def test_coalescing_group_stays_in_one_subsegment():
    specs = [
        LayoutSoftCoreSpec(10, 4, threshold_group_id=1, latency_tag=0, name="p0"),
        LayoutSoftCoreSpec(10, 4, threshold_group_id=1, latency_tag=0, name="p1"),
        LayoutSoftCoreSpec(1, 4, threshold_group_id=1, latency_tag=0, name="acc"),
        LayoutSoftCoreSpec(8, 8, threshold_group_id=2, latency_tag=1, name="other"),
    ]
    group_ids = [0, 0, 0, None]
    hw = [LayoutHardCoreType(max_axons=32, max_neurons=32, count=4)]

    # Force a tight pool so the splitter must use multiple sub-segments.
    sub_segments = split_softcores_by_capacity(
        specs,
        hw,
        allow_coalescing=True,
        coalescing_group_ids=group_ids,
    )
    assert len(sub_segments) >= 1

    seen_groups: set[int] = set()
    for sub in sub_segments:
        gids = {group_ids[specs.index(sc)] for sc in sub if group_ids[specs.index(sc)] is not None}
        for gid in gids:
            assert gid not in seen_groups, f"coalescing group {gid} split across passes"
            seen_groups.add(gid)

    assert 0 in seen_groups
    assert all(
        sum(1 for sc in sub if group_ids[specs.index(sc)] == 0) in (0, 3)
        for sub in sub_segments
    )
