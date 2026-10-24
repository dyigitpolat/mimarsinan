"""[W04] One residency basis in a build: the builder's pass planner packs each IR core under the class the hard-core
packer reads on its soft core.

The IR's layout record keys residency classes on the cores' values (``residency_key``), so two layers with equal
values share one class there. A segment's soft cores carry the source perceptron's class instead (the segment
subgraph the packer reads carries no layout record, ``legacy_convert._residency_class_of``), and so do the shape-only
walk's specs (``LayoutIRMapping`` has no values to read). Before W04 the planner read the record's values class: it
planned passes that put two layers on one core, the packer kept them apart, and the build refused its own plan
(``PlannedPassPackingRefusal``) where the shape-only plane, packing the perceptron classes, built. The census found
this on RegNet-X/Y at F/64 (RA/11_wave2/pack/plane_parity.md); RegNet-X 400MF at 32 x 32 on 256 x 256 cores at nine
cores is that divergence, reduced (seed 0, every layer's values 1.0, so the record holds one class).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
import torchvision

from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph
from mimarsinan.mapping.packing.hybrid_build_pool import _segment_specs
from mimarsinan.mapping.packing.hybrid_segment_helpers import _remap_external_sources_to_segment_inputs
from mimarsinan.mapping.packing.neural_segment_packing import neural_segment_to_soft_core_mapping
from mimarsinan.mapping.pruning.ir_segmentation import build_ir_consumed_by

from . import planes as P


def _partition(classes):
    """The equivalence relation a class list induces: which pairs share a class."""
    return {(i, j) for i, j in itertools.combinations(range(len(classes)), 2) if classes[i] == classes[j]}


def _segment_soft_cores(ir, cores):
    """The soft cores the hard-core packer packs for one segment (``_flush_neural_segment``'s conversion)."""
    consumed_by = build_ir_consumed_by(ir)
    ids = {c.id for c in cores}
    refs = [IRSource(node_id=c.id, index=i) for c in cores if any(k not in ids for k in consumed_by.get(c.id, set()))
            for i in range(c.get_output_count())]
    seg_graph, _ = _remap_external_sources_to_segment_inputs(
        nodes=cores, segment_output_refs=np.array(refs, dtype=object), weight_banks=ir.weight_banks)
    return neural_segment_to_soft_core_mapping(seg_graph, ir.weight_banks).cores


@pytest.fixture(scope="module")
def regnet():
    torch.manual_seed(0)
    model = torchvision.models.regnet_x_400mf(weights=None).eval()
    original = P.vehicle
    P.vehicle = lambda name: (model, (3, 32, 32))
    try:
        return P.representation("regnet_x_400mf")
    finally:
        P.vehicle = original


def test_the_planner_packs_the_class_the_packer_reads(regnet):
    ir = IRMapping(q_max=1.0, firing_mode="Default", max_axons=256, max_neurons=256, allow_coalescing=True,
                   hardware_bias=True).map(regnet)
    regnet.clear_ir_caches()
    merged = checked = 0
    for segment in partition_ir_graph(ir):
        if not isinstance(segment, NeuralSegment):
            continue
        specs, spec_to_core, _ = _segment_specs(segment.nodes, ir_graph=ir, hardware_bias=True)
        soft = {c.id: c for c in _segment_soft_cores(ir, segment.nodes)}
        planned = [s.residency_class_id for s in specs]
        packed = [soft[spec_to_core[id(s)].id].residency_class_id for s in specs]
        record = {ir.layout_softcores[int(spec_to_core[id(s)].layout_softcore_index)].residency_class_id for s in specs}
        merged += len(record) < len({spec_to_core[id(s)].perceptron_index for s in specs})
        assert _partition(planned) == _partition(packed), segment.label
        checked += len(specs)
    assert merged, "the vehicle must hold a segment whose layers share one values class in the IR's record"
    assert checked == sum(1 for c in ir.nodes if isinstance(c, NeuralCore))


def test_a_planned_pass_of_layers_with_equal_values_builds(regnet):
    layout, specs = P.walk(regnet, 256, 256)
    assert P.footprint(specs, 256, 256) == 287
    shape = P.shape_answer(layout, specs, 256, 256, 9, scheduled=True)
    executed = P.executed_answer(regnet, 256, 256, 9, scheduled=True)
    assert shape.ok and executed.ok, (shape, executed)
    assert shape.passes == executed.passes == 92
