"""[F2] A lane program with fused adds (WI-5e class E) plans and executes a declared residency step.

The vehicle is the lane residual net of the lane certificates (``lanes/lane_programs.py``, a projection and an
identity shortcut, each add fused on chip) under lanes L, E and L, E, P, on the exactness review's lane-floor
geometry: cores of the widest fan-in plus two axons and three neurons, a third of the program's tiles (C =
ceil(F / 3)), neuron splitting, scheduling. Before the unit order read the layer graph every step refused ("layer order
is not a program order": the accumulator conv2 reads the projection shortcut, which the IR emits after it).

A second geometry, cores of 225 axons and three neurons at a third of the program's tiles, is where the fused adds'
operand rows decide the packing: three pieces of the first accumulator (73 record axons, 81 placed: its operand rows
are axons too) share a core by the record extent and not as placed. A planner packing the record extent plans passes
the builder cannot place; the planner packs the placed axons and counts the record's cells (``operand_axons``). There
U1 (no declared plan) builds the head's program, pass for pass and core for core.

Per class set, geometry and step (floor, stateless):
- the build executes more than one pass, its plan rows state the declared step, and its outputs equal the identity
  program's and the resident program's (one pass of unbounded cores) at atol 0 on 64 inputs, the int64 golden of
  the executed program equals its value run, and every exposed node window equals the fp64 identity program's at
  atol 0 (the executed residency certificate's budget invariance, K10, on the program);
- its record plane equals its plan plane field by field per pass and the carry per boundary (K12's census, CL3 (d),
  read at one byte per carried element as K12 reads a 4-bit value grid), and each executed pass packs and pins the
  cores of the composition's packing of it (in the hard-core packer's order);
- W15 establishes Theorem 1, Proposition 2 and H10 on the executed program, with no unknown program form.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from mimarsinan.certification.program_check.check import check_deployment
from mimarsinan.certification.program_check.declaration import CheckDeclaration
from mimarsinan.certification.program_check.report import ESTABLISHED, P2, T1
from mimarsinan.certification.value_certificate import certify_twin_flow_values, exposed_nodes
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.deployment_record.quantities.write_law import plane_write_census
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.packing.schedule_residency_plan import SCHEDULE_RESIDENCY_PASS
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENCY_PLAN, SCHEDULE_RESIDENT_CORES
from mimarsinan.mapping.support.schedule.residency_units import facts_period_census
from mimarsinan.torch_mapping.value_parity.value_program import fp64_identity_program
from unit.chip_simulation.value_run.integer_programs import residual_block_net
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform, real_inputs

from . import certified as CF

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]

CLASSES = (("L", "E"), ("L", "E", "P"))
STEPS = ("floor", "stateless")
#: The lane-floor geometry (widest fan-in plus two axons) and the one where the operand rows decide the packing.
GEOMETRIES = ("lane_floor", "axons_225")
#: U1's program at 225 x 3 on the head c9131188 (passes' hard cores), which the unit keeps.
U1_AT_225_ON_THE_HEAD = {("L", "E"): [683, 683, 704, 684, 384, 4], ("L", "E", "P"): [1049, 1049, 1049, 4]}
WEIGHT_BITS = 4
_PROGRAMS: Dict[Tuple[str, ...], Any] = {}
_BUILDS: Dict[Tuple[Any, ...], Any] = {}


def _build(ir, cores_config, caps):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_hybrid_hard_core_mapping(ir_graph=copy.deepcopy(ir), cores_config=copy.deepcopy(cores_config),
                                              strategy=MappingStrategy.resolve(ChipCapabilities(**caps)),
                                              max_schedule_passes=caps.get("max_schedule_passes", 8))


def _program(classes, geometry="lane_floor"):
    """(program, platform, the geometry's cores at a third of its tiles, the resident program on unbounded cores)."""
    if (classes, geometry) not in _PROGRAMS:
        if classes not in _PROGRAMS:
            with contextlib.redirect_stdout(io.StringIO()):
                platform = lane_platform(list(classes))
                _PROGRAMS[classes] = (build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                                         platform=platform), platform)
        program, platform = _PROGRAMS[classes]
        ir = program.ir
        a = max(int(n.get_core_matrix(ir).shape[0]) for n in ir.nodes if isinstance(n, NeuralCore))
        axons = a + 2 if geometry == "lane_floor" else 225
        flat = _build(ir, [{"max_axons": axons, "max_neurons": 3, "count": 4096}], dict(allow_neuron_splitting=True))
        F = sum(len(s.hard_core_mapping.cores) for s in flat.stages if s.kind == "neural")
        _PROGRAMS[(classes, geometry)] = (program, platform,
                                          [{"max_axons": axons, "max_neurons": 3, "count": math.ceil(F / 3)}], flat)
    return _PROGRAMS[(classes, geometry)]


def _caps(step):
    caps: Dict[str, Any] = dict(allow_neuron_splitting=True, allow_scheduling=True, max_schedule_passes=4096)
    if step is not None:
        caps["residency_plan"] = {"step": step}
    return caps


def _executed(classes, step, geometry="lane_floor"):
    if (classes, step, geometry) not in _BUILDS:
        program, _platform, cores_config, _flat = _program(classes, geometry)
        _BUILDS[(classes, step, geometry)] = _build(program.ir, cores_config, _caps(step))
    return _BUILDS[(classes, step, geometry)]


def _surface(platform, cores_config):
    """The census surface K12 reads on a 4-bit value grid: one byte per carried element (8-bit view)."""
    return {**platform, "cores": cores_config, "weight_bits": WEIGHT_BITS, "core_semantics": "mvm",
            "activation_bits": 8, "allow_neuron_splitting": True, "allow_scheduling": True,
            "max_schedule_passes": 4096}


def _neural(hybrid):
    return [s for s in hybrid.stages if s.kind == "neural"]


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("classes", CLASSES, ids=["L_E", "L_E_P"])
def test_a_lane_program_with_fused_adds_plans_and_executes_a_residency_step(classes, step, geometry):
    program, _platform, cores_config, flat = _program(classes, geometry)
    hybrid = _executed(classes, step, geometry)
    passes = _neural(hybrid)
    assert len(passes) > 1
    plan = getattr(hybrid, SCHEDULE_RESIDENCY_PLAN)
    assert {p.step_rule for p in plan} == {step} and sum(p.passes for p in plan) == len(passes)
    assert all(p.core_budget == cores_config[0]["count"] for p in plan)
    if step == "floor":
        assert sum(len(getattr(s, SCHEDULE_RESIDENT_CORES)) for s in passes) > 0   # the floor pins cores
    x = real_inputs(program, 64, seed=1000)
    with torch.no_grad():
        want = program.identity()(x)
        got = ValueHybridCoreFlow(hybrid, dtype=torch.float64)(x).reshape(want.shape)
        resident = ValueHybridCoreFlow(flat, dtype=torch.float64)(x).reshape(want.shape)
    assert torch.equal(got, want) and torch.equal(got, resident)
    assert float(want.abs().sum()) > 0
    codes, steps = run_golden(hybrid, x.numpy())
    assert np.array_equal(codes * steps, got.reshape(codes.shape).numpy())
    with torch.no_grad():
        cert, detail = certify_twin_flow_values(fp64_identity_program(program.ir, exposed=exposed_nodes(hybrid)),
                                                ValueHybridCoreFlow(hybrid, dtype=torch.float64), x[:16], atol=0.0)
    assert cert.passed and cert.neuron_windows_compared > 0, detail


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("classes", CLASSES, ids=["L_E", "L_E_P"])
def test_its_record_plane_equals_its_plan_plane(classes, step, geometry):
    program, platform, cores_config, _flat = _program(classes, geometry)
    hybrid = _executed(classes, step, geometry)
    surface = _surface(platform, cores_config)
    comp = CF.builder_composition(program.ir, surface, {"step": step})
    plan = facts_period_census(comp.facts, forced=comp.units.forced,
                               bytes_of_cells=lambda cells: params_bytes(cells, WEIGHT_BITS), tile_cells=0,
                               bytes_per_element=1, label="plan")
    written, reason = plane_write_census(mapping_program_passes(hybrid), surface, weight_bits=WEIGHT_BITS,
                                         timesteps=None, label="record")
    assert written is not None, reason
    assert CF.differences(written.census, plan) == []
    assert CF.carried(plan), "the program carries across its pass boundaries"
    for stage in _neural(hybrid):              # the two packers place the same cores (their order may differ)
        census = getattr(stage, SCHEDULE_RESIDENCY_PASS)
        assert sorted(census.planned_cores) == sorted(census.executed_cores)
        assert sorted(census.planned_cores[i] for i in census.planned_resident) == sorted(
            census.executed_cores[i] for i in census.executed_resident)
    assert len(comp.facts.phases) == len(_neural(hybrid))


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("classes", CLASSES, ids=["L_E", "L_E_P"])
def test_w15_establishes_the_lane_residency_program(classes, step):
    program, _platform, cores_config, _flat = _program(classes)
    hybrid = _executed(classes, step)
    declaration = CheckDeclaration(core_types=tuple(dict(c) for c in cores_config), semantics="value_integer",
                                   window=1, pass_transfer="collapse",
                                   accumulation={"value_census": "float64", "int64_golden": "int64"},
                                   upload={"value_census": "float32"})
    with contextlib.redirect_stdout(io.StringIO()):
        report = check_deployment(hybrid, ir_graph=program.ir, declaration=declaration, unbounded_ir=program.ir)
    assert report.established(T1) and report.established(P2), (report.theorem1, report.proposition2,
                                                                report.failures_by_condition())
    assert report.results["H10"].status == ESTABLISHED
    assert not report.unknown_forms


@pytest.mark.parametrize("classes", CLASSES, ids=["L_E", "L_E_P"])
def test_where_the_operand_rows_decide_the_packing_u1_keeps_the_heads_program(classes):
    program, _platform, _cores_config, _flat = _program(classes, "axons_225")
    hybrid = _executed(classes, None, "axons_225")
    assert [len(s.hard_core_mapping.cores) for s in _neural(hybrid)] == U1_AT_225_ON_THE_HEAD[classes]
    x = real_inputs(program, 64, seed=1000)
    with torch.no_grad():
        want = program.identity()(x)
        assert torch.equal(ValueHybridCoreFlow(hybrid, dtype=torch.float64)(x).reshape(want.shape), want)
