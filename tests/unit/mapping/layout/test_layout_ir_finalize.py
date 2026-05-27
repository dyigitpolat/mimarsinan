"""LayoutIRMapping finalizes latency tags and threshold groups."""

from __future__ import annotations

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec


def test_finalize_sets_latency_and_threshold_groups():
    layout = LayoutIRMapping(max_axons=8, max_neurons=8)
    node_id = 0
    layout._node_input_node_ids[node_id] = set()
    layout._node_is_neural[node_id] = True
    layout._node_id_to_softcore_idx[node_id] = 0
    layout._sc_idx_to_perceptron_index[0] = 0
    layout.layout_softcores.append(
        LayoutSoftCoreSpec(
            input_count=2,
            output_count=2,
            threshold_group_id=0,
        )
    )
    layout._finalize_softcores()

    sc = layout.layout_softcores[0]
    assert sc.latency_tag is not None
    assert sc.segment_id is not None
    assert sc.threshold_group_id == 0
    assert layout.layout_preview is not None
    assert "neural_segments" in layout.layout_preview
