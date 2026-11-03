"""The seams of the lanes unit (WI-5e) with the closure's per-axon weight sign (CL-SG) and N2's operand core types,
found when the units were integrated on one tree.

Per-axon sign: the sign split re-describes a soft core by its physical rows after compaction (a mixed row becomes a
W+ and a W- row), while a lane law names each column's fused operand row by the soft core's logical row. Placement
lands every operand row on its physical row (an operand row has zero weights, so it is never mixed and takes one
row): a lane program on a per-axon pool certifies at atol 0, neuron-split fragments of a signed parent included.

Operand core types: the lane fusion, the lane grids and the lane law solver read the static perceptrons' readout
ranges only; N2's operand banks take their payload grids, consumer groups, thresholds and stamps on the per-core law
path, which a lane program does not take. Executed lanes beside an operand core type are refused by name, with that
reason, at every door, at Operand Lowering and at the law hook; before, the lane grids refused the build after
calibration with the false reason that the operand banks' perceptrons "carry cores but no readout range"."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.export.law_solver import install_readout_laws
from mimarsinan.mapping.lanes.declaration import LaneDeclarationError
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.operand_lowering import OperandDeclarationError, operand_declaration
from unit.chip_simulation.value_run.integer_programs import assert_exact, residual_block_net
from unit.chip_simulation.value_run.operand_programs import attention_program, operand_platform
from unit.config_schema.test_target_doors import PC, VALUE, _load, _refused_by_name, _verdicts
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform, real_inputs

LANES = {"classes": ["L", "E", "P"]}
LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 16}
STATIC = {"max_axons": 256, "max_neurons": 256, "count": 6}
OPERAND = {"max_axons": 240, "max_neurons": 240, "count": 4, "operand": {"payload_bits": 8}}
NAMED = "executed on-chip lanes and an operand core type"


def _pack(program, *, axons, neurons, count=999, passes=None, sign="per_axon"):
    strategy = MappingStrategy.resolve(ChipCapabilities(
        allow_neuron_splitting=True, allow_coalescing=False, allow_scheduling=passes is not None,
        max_schedule_passes=passes or 8))
    return build_hybrid_hard_core_mapping(
        ir_graph=program.ir, strategy=strategy, max_schedule_passes=passes or 8,
        cores_config=[{"max_axons": axons, "max_neurons": neurons, "count": count, "weight_sign_granularity": sign}])


@pytest.fixture(scope="module")
def residual_lanes():
    return build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform())


#: Per-axon pools for the residual block's lane program: resident, neuron-split (16 output channels on 8-neuron
#: cores: fragments of signed parents) and scheduled.
GEOMETRIES = {
    "resident": dict(axons=1024, neurons=32),
    "neuron_split": dict(axons=2048, neurons=8, count=2000),
    "scheduled": dict(axons=1024, neurons=32, count=40, passes=128),
}


class TestLanesOnAPerAxonPool:
    @pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
    def test_a_lane_program_on_a_per_axon_pool_is_exact_at_atol_zero(self, residual_lanes, geometry):
        program = residual_lanes
        mapping = _pack(program, **GEOMETRIES[geometry])
        cores = [c for s in mapping.stages if s.kind == "neural" for c in s.hard_core_mapping.cores]
        operand_cores = [c for c in cores if np.any(np.asarray(getattr(c.readout_law, "operand_row", [-1])) >= 0)]
        assert operand_cores, "the program exercises the seam: fused operand rows on packed cores"
        assert any(getattr(c, "weight_sign_granularity", None) == "per_axon" for c in cores)
        x = real_inputs(program, 32)
        with torch.no_grad():
            scm = program.identity()(x)
            hcm = ValueHybridCoreFlow(mapping, dtype=torch.float64, device="cpu")(x)
        assert_exact(program.nf()(x).detach(), scm)
        assert_exact(hcm, scm)
        codes, steps = run_golden(mapping, x.numpy())
        assert_exact(torch.as_tensor(codes * steps), hcm)

    def test_the_split_moves_some_operand_row_off_its_logical_index(self, residual_lanes):
        """The seam is real on this program: some lane core's mixed rows sit above one of its operand rows."""
        program = residual_lanes
        moved = 0
        for node in program.ir.nodes:
            law = getattr(node, "readout_law", None)
            rows = np.asarray(getattr(law, "operand_row", []))
            if not rows.size or not (rows >= 0).any():
                continue
            matrix = np.asarray(node.get_core_matrix(program.ir))
            mixed = (matrix > 0).any(axis=1) & (matrix < 0).any(axis=1)
            moved += int(sum(mixed[:r].any() for r in rows[rows >= 0].tolist()))
        assert moved > 0


def _operand_lane_platform():
    return {**operand_platform(), "onchip_arithmetic": copy.deepcopy(LANES)}


def _document(cores, **extra):
    doc = _load(VALUE)
    doc[PC] = {**{k: v for k, v in doc[PC].items() if k not in ("cores", "max_axons", "max_neurons")},
               "cores": copy.deepcopy(cores), "readout_law": dict(LAW), **copy.deepcopy(extra)}
    return doc


class TestLanesBesideAnOperandCoreType:
    def test_operand_lowering_refuses_executed_lanes_beside_an_operand_core_type_by_name(self):
        with pytest.raises(OperandDeclarationError, match=NAMED):
            operand_declaration(_operand_lane_platform())

    def test_the_off_ablation_and_the_float_path_read_as_before(self):
        off = {**operand_platform(), "onchip_arithmetic": {"classes": []}}
        assert operand_declaration(off) is not None
        float_path = {k: v for k, v in _operand_lane_platform().items() if k != "readout_law"}
        assert operand_declaration(float_path) is not None

    def test_the_door_rule_names_the_composition_once(self):
        from mimarsinan.config_schema.registry.core_type_doors import platform_lane_errors

        flat = {"cores": [dict(STATIC), dict(OPERAND)], "readout_law": dict(LAW), "onchip_arithmetic": LANES}
        errors = platform_lane_errors(flat)
        assert len(errors) == 1 and NAMED in errors[0], errors

    def test_the_door_rule_admits_either_declaration_alone(self):
        from mimarsinan.config_schema.registry.core_type_doors import platform_lane_errors

        assert platform_lane_errors({"cores": [dict(STATIC)], "readout_law": dict(LAW),
                                     "onchip_arithmetic": LANES}) == []
        assert platform_lane_errors({"cores": [dict(STATIC), dict(OPERAND)], "readout_law": dict(LAW)}) == []
        assert platform_lane_errors({"cores": [dict(STATIC), dict(OPERAND)], "onchip_arithmetic": LANES}) == []
        assert platform_lane_errors({"cores": [dict(STATIC), dict(OPERAND)], "readout_law": dict(LAW),
                                     "onchip_arithmetic": {"classes": []}}) == []

    def test_a_malformed_lane_declaration_is_left_to_its_own_rule(self):
        from mimarsinan.config_schema.registry.core_type_doors import platform_lane_errors

        assert platform_lane_errors({"cores": [dict(OPERAND)], "readout_law": dict(LAW),
                                     "onchip_arithmetic": {"classes": ["Q"]}}) == []

    def test_every_door_refuses_the_composition_by_name(self, tmp_path):
        _refused_by_name(_document([STATIC, OPERAND], onchip_arithmetic=LANES), tmp_path, NAMED)

    def test_an_operand_chip_without_lanes_passes_every_door(self, tmp_path):
        assert _verdicts(_document([STATIC, OPERAND]), tmp_path) == ([], [], None)

    def test_the_law_hook_refuses_a_lane_program_over_operand_banks_by_name(self):
        program = attention_program("softmax")
        assert getattr(program.ir, "dynamic_bank_bindings", None)
        with pytest.raises(LaneDeclarationError, match=NAMED):
            install_readout_laws(program.ir, program.model, _operand_lane_platform(), weight_quantization=True,
                                 weight_bits=8)


class TestIdentityOutputBesideAnAbsorbedRelu:
    """The lanes unit's identity output law (WI-5d) has no clamp; N2's operand floors put an absorbed ReLU into a
    bank perceptron's law as its floor. The per-core solver refuses the two on one perceptron by name rather than
    dropping the ReLU."""

    def test_an_identity_output_whose_operand_product_absorbed_a_relu_is_refused_by_name(self):
        from mimarsinan.mapping.export.law_grids import assign_core_grids
        from mimarsinan.mapping.export.law_solver import ReadoutLawSpec, solve_ir_laws
        from mimarsinan.mapping.export.law_stamp import readout_quantizers_by_perceptron
        from mimarsinan.mapping.export.operand_laws import operand_floors, operand_ranges, output_groups_of
        from mimarsinan.mapping.export.output_law import OutputLawError

        program = attention_program("relu")
        model_repr = program.model.get_mapper_repr()
        floors = operand_floors(model_repr)
        assert floors, "the ReLU attention program's bank perceptrons carry the absorbed ReLU as floor 0"
        ranges = {**{i: float(e[2].range) for i, e in readout_quantizers_by_perceptron(model_repr).items()},
                  **operand_ranges(model_repr)}
        spec = ReadoutLawSpec.from_declaration(program.platform["readout_law"],
                                               activation_bits=program.platform["activation_bits"], weight_bits=8)
        grids, _steps = assign_core_grids(program.ir, ranges, spec.activation_bits,
                                          groups=output_groups_of(model_repr))
        assert solve_ir_laws(program.ir, ranges, spec, grids=grids, floors=floors)
        with pytest.raises(OutputLawError, match="absorbed a ReLU"):
            solve_ir_laws(program.ir, ranges, spec, grids=grids, floors=floors,
                          identity_outputs=frozenset({next(iter(floors))}))
