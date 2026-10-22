"""Acceptance 11: on the scheduled spiking tier-0 cells at random weights, with each template's core types at a
budget forcing Q >= 2, the HCM run of the executed floor (and of the stateless step on t0_03, t0_22 and t0_53) equals
the SCM identity run and the U1-scheduled run, outputs and per-node spike counts, on 4 seeded batches.

The vehicles carry the IR latency pass the pipeline runs (``SoftCoreMappingStep``): without it every hard core's
latency read 0 and the analytical TTFS executor ran a packed segment in packing order, the t0_08 gap (0.144) of the
earlier rounds, which was never a pass cut (``test_u1_exactness.TestTheAnalyticalTtfsCoreOrder``).

Each row states what it certifies: ``CUT`` rows cut a segment into passes (a carried stage exists) and write cores
every period (W_exec > 0); ``SEGMENTS`` rows run one pass per segment, because no feasible budget cuts a segment of
that step (t0_03 stateless: every segment's tiles fit any C that holds its 262-tile unit; t0_22: units (66, 1, 1),
whose second segment's two tiles always fit), so they certify segment-level passes with pins, not a residency cut.

The tolerance is the existing certificate's: where the semantics' certification observable is counts, the
streaming twin spike-count certificate (atol 0, the ``run_spike_count_certificate_gate`` cell); exact output
equality on every cell. Random weights leave these networks silent, so each vehicle's weights are scaled by a
declared gain (``templates.activate``) until it fires; the activity is asserted, so no comparison is vacuous.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.certification.count_alignment import certify_twin_flow_counts, flow_node_counts
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.support.schedule.pass_carry import carried_outputs_by_stage
from mimarsinan.mapping.support.schedule.residency_composition import ShapeOnlyResidency, compose_shape_only_program
from mimarsinan.mapping.packing.residency_tiles import core_type_tiles
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.simulation_factory import (
    build_hybrid_mapping_for_pipeline,
    build_identity_mapping_for_pipeline,
    build_spiking_hybrid_flow,
)

from . import templates as TV

CUT, SEGMENTS = "cut", "segments"
#: (cell, core-count scale forcing Q >= 2 with every unit within the budget, weight gain, {step: what it certifies})
CELLS = [
    ("t0_03", 1.0, 32.0, {"floor": CUT, "stateless": SEGMENTS}),
    ("t0_08", 1.0, 0.0, {"floor": CUT}),
    ("t0_22", 0.7, 32.0, {"floor": SEGMENTS, "stateless": SEGMENTS}),
    ("t0_26", 1.2, 32.0, {"floor": CUT}),
    ("t0_53", 1.34, 32.0, {"floor": CUT, "stateless": CUT}),
]


def _counts_certified(ir, reference, backend, x):
    cert, report, _ = certify_twin_flow_counts(ir, reference, backend, x, backend="hcm", discipline="streaming",
                                               reference_discipline="streaming")
    assert cert.passed and cert.neuron_windows_compared > 0, (cert.summary(), report)


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("cell, scale, gain, steps", CELLS)
def test_the_executed_plan_equals_the_identity_and_u1_runs(cell, scale, gain, steps):
    vehicle = TV.template_vehicle(cell)
    if gain > 0:
        TV.activate(vehicle.ir, gain)
    pipeline, config = vehicle.pipeline(), vehicle.config
    identity = build_spiking_hybrid_flow(
        pipeline, build_identity_mapping_for_pipeline(vehicle.ir, pipeline_config=config))
    u1 = build_spiking_hybrid_flow(pipeline, build_hybrid_mapping_for_pipeline(
        vehicle.ir, TV.residency_platform(vehicle, scale, None), pipeline_config=config))
    counts_cell = DeploymentPlan.resolve(config).mode_policy().certification_observable()[0] == "counts"
    for step, kind in steps.items():
        mapping = build_hybrid_mapping_for_pipeline(
            vehicle.ir, TV.residency_platform(vehicle, scale, {"step": step}), pipeline_config=config)
        plan = mapping.schedule_residency_plan
        assert sum(p.passes for p in plan) >= 2
        passes = [s for s in mapping.stages if s.kind == "neural"]
        w_exec = sum(len(s.hard_core_mapping.cores) - len(s.schedule_resident_cores) for s in passes)
        if kind == CUT:
            assert carried_outputs_by_stage(mapping.stages) and w_exec > 0    # a residency cut, cores written
        else:
            assert not carried_outputs_by_stage(mapping.stages)               # segment-level passes only
            assert w_exec > 0 or step == "floor"                              # t0_22's floor pins every core
        executed = build_spiking_hybrid_flow(pipeline, mapping)
        outputs = []
        for batch in range(4):
            x = torch.rand(4, *vehicle.input_shape, generator=torch.Generator().manual_seed(batch))
            with torch.no_grad():
                want = identity(x)
                got, scheduled = executed(x), u1(x)
            assert torch.equal(got, scheduled)                                      # the executed plan = U1
            assert torch.equal(got, want)                                           # = the SCM identity run
            outputs.append(want)
            if counts_cell:
                _counts_certified(vehicle.ir, identity, executed, x)
                _counts_certified(vehicle.ir, identity, u1, x)
                counts = flow_node_counts(executed, x, discipline="streaming")
                assert sum(float(c.sum()) for c in counts.values()) > 0              # the vehicle fires
        stacked = torch.cat(outputs)
        assert float(stacked.std(0).max()) > 0                                       # and discriminates


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("step", ["floor", "stateless"])
def test_both_planes_execute_the_step_on_the_deep_cnn_of_t0_03(step):
    """Acceptance 9 on t0_03: the built program's pass partition and pins equal collect_noc_fragments' and the
    step's phases."""
    vehicle = TV.template_vehicle("t0_03")
    platform = TV.residency_platform(vehicle, 1.0, {"step": step})
    mapping = build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=vehicle.config)
    caps = ChipCapabilities.from_platform_constraints(platform)
    types = core_types_from_declaration(platform["cores"])
    frag = collect_noc_fragments(softcores=vehicle.specs, core_types=types, census=vehicle.census,
                                 program=vehicle.program, **caps.layout_kwargs())
    comp = compose_shape_only_program(vehicle.specs, types,
                                      ShapeOnlyResidency({"step": step}, vehicle.census, vehicle.program),
                                      allow_coalescing=caps.allow_coalescing,
                                      allow_splitting=caps.allow_neuron_splitting,
                                      max_schedule_passes=caps.max_schedule_passes)
    layer_of = {int(n.id): n.perceptron_index for n in vehicle.ir.get_neural_cores()}
    passes = [s for s in mapping.stages if s.kind == "neural"]
    built = [tuple(sorted({layer_of[int(p["ir_node_id"])] for ps in s.hard_core_mapping.soft_core_placements_per_hard_core
                           for p in ps})) for s in passes]
    shape_only = [tuple(sorted({vehicle.specs[i].perceptron_index for i, _ in pl})) for pl in frag.pass_placements]
    planned = [tuple(sorted(comp.units.layers[u] for u in range(a, b))) for a, b in comp.facts.phases]
    assert built == shape_only == planned and len(planned) == comp.step[2] >= 2
    assert [s.schedule_resident_cores for s in passes] == [p.schedule_resident_cores for p in frag.pass_programs]
    assert [(p.rotating_tiles, p.core_budget) for p in mapping.schedule_residency_plan] == [
        (comp.step[0], comp.budget)] * len(mapping.schedule_residency_plan)
    # programmed bytes, defined once: the composition's written cells are the executed passes' unpinned cores. The
    # builder fuses a position's two 1024-axon tiles into one core (``canonical_fuse_hardcores``) where the layout
    # packer keeps them apart, so the planes agree per pass in cells, bytes and tiles, not core by core.
    bits = int(vehicle.config.get("weight_bits", 8))
    written = [[core for i, core in enumerate(s.hard_core_mapping.cores) if i not in s.schedule_resident_cores]
               for s in passes]
    cells = [[CoreOccupancy.from_hard_core(core).cells_used for core in w] for w in written]
    assert [sum(c) for c in cells] == [sum(c) for c in comp.facts.written_cells]
    executed_bytes = sum(params_bytes(c, bits) for w in cells for c in w)
    assert executed_bytes == comp.facts.programmed_bytes(lambda c: params_bytes(c, bits))
    w_exec = sum(len(core_type_tiles(core.axons_per_core, core.neurons_per_core, types,
                                     getattr(core, "fused_component_axons", None))) for w in written for core in w)
    assert w_exec == comp.facts.written_core_tiles <= comp.facts.written_tiles
    print(f"t0_03 {step}: programmed bytes {executed_bytes}, W_exec {w_exec}, "
          f"W {comp.facts.written_tiles} (layer-granular; the difference is joint packing of a phase)")
