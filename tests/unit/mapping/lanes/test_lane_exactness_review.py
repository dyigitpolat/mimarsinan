"""The lane exactness review's findings as certificates (11_wave2/lanes/review_exactness.md): NF = SCM = HCM = int64
golden at atol 0, or a refusal by name, for every reproducer.

M1 a fused add whose accumulator reads a host value; M2 a neuron split of a fused-add core; M3 the identity output
law beside an entry offset; M4 pruning beside an offset grid; m1 a law bias on an always-on row; m2 an operand of two
grids in one channel; m3 a core type wider than 16 bits; m4 the fold's 2^53 bound with the operand term."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.lanes.ir.fusion import OPERAND_ROWS
from unit.chip_simulation.value_run.integer_programs import PRUNE, assert_exact, lenet5, residual_block_net
from unit.mapping.lanes.lane_programs import build_lane_program, cores, host_ops, lane_platform, real_inputs


def _certify(program, x, **geometry):
    """NF = SCM = HCM = golden at atol 0 on ``x``; the packed mapping."""
    with torch.no_grad():
        scm = program.identity()(x)
        assert_exact(program.nf()(x), scm)
        mapping = program.packed_mapping(**geometry)
        hcm = ValueHybridCoreFlow(mapping, dtype=torch.float64)(x)
    assert_exact(hcm, scm)
    codes, steps = run_golden(mapping, x.numpy())
    assert np.array_equal(codes * steps, hcm.reshape(codes.shape).numpy())
    return mapping


class _PoolMainPath(nn.Module):
    """R1: ``conv(maxpool(h)) + conv1x1s2(h)``; the max pool stays on the host (class C undeclared)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(11)
        self.stem, self.bn, self.relu = nn.Conv2d(3, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(8, 8, 3, 1, 1)
        self.short = nn.Conv2d(8, 8, 1, 2)
        self.relu2, self.gap, self.fc = nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Linear(8, 10)

    def forward(self, x):
        h = self.relu(self.bn(self.stem(x)))
        y = self.conv(self.pool(h)) + self.short(h)
        return self.fc(torch.flatten(self.gap(self.relu2(y)), 1))


class _EarlierOperand(nn.Module):
    """R1e: ``conv(maxpool3s1(h)) + h``, the operand from the segment before the host pool."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(13)
        self.stem, self.bn, self.relu = nn.Conv2d(3, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU()
        self.pool = nn.MaxPool2d(3, 1, 1)
        self.conv = nn.Conv2d(8, 8, 3, 1, 1)
        self.relu2, self.gap, self.fc = nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Linear(8, 10)

    def forward(self, x):
        h = self.relu(self.bn(self.stem(x)))
        y = self.conv(self.pool(h)) + h
        return self.fc(torch.flatten(self.gap(self.relu2(y)), 1))


def _commensurate(flow, entries, grids, perceptrons, quantizers, lanes):
    """R1d / R1e: the stem lane at step 0.125 (codes 0..15) and the conv's entry grid at the same step with half its
    code range (codes -7..7), so the stem's values sit on the entry grid and clamp there."""
    stem = [q for p, q in zip(perceptrons, quantizers) if str(getattr(p, "name", "")).startswith("stem")][0]
    conv = [g for p, g in zip(entries, grids) if str(getattr(p, "name", "")).startswith("conv")][0]
    stem.calibrate(1.875)
    conv.calibrate(0.875)


class TestM1HostFedAccumulator:
    def test_a_host_fed_perceptron_takes_no_add_and_the_other_input_holds_it(self):
        program = build_lane_program(_PoolMainPath().eval(), (3, 8, 8), num_classes=10, platform=lane_platform())
        assert host_ops(program) == ["MaxPool2d"]
        fused = [c for c in cores(program) if getattr(c, OPERAND_ROWS, None) is not None]
        assert fused and all(c.name.startswith("short") for c in fused), "the shortcut accumulates the add"
        assert not any(getattr(c, "boundary_grid", None) is not None and c.boundary_grid.armed for c in fused)

    @pytest.mark.parametrize("hook", [None, _commensurate], ids=["calibrated", "commensurate"])
    def test_r1_and_r1d_certify(self, hook):
        program = build_lane_program(_PoolMainPath().eval(), (3, 8, 8), num_classes=10, platform=lane_platform(),
                                     state_hook=hook)
        _certify(program, real_inputs(program, 32), axons=256, neurons=64)

    @pytest.mark.parametrize("hook", [None, _commensurate], ids=["calibrated", "commensurate"])
    def test_r1e_keeps_the_add_on_the_host_and_certifies(self, hook):
        program = build_lane_program(_EarlierOperand().eval(), (3, 8, 8), num_classes=10, platform=lane_platform(),
                                     state_hook=hook)
        assert "_operator.add" in host_ops(program) and "MaxPool2d" in host_ops(program)
        _certify(program, real_inputs(program, 32), axons=256, neurons=64)

    def test_the_fusion_refuses_a_host_fed_accumulator_by_name(self):
        from types import SimpleNamespace

        from mimarsinan.mapping.ir import IRSource
        from mimarsinan.mapping.lanes.ir.fusion import _Fuser
        from mimarsinan.mapping.lanes.ir.refusal import LaneFusionError

        fuser = _Fuser.__new__(_Fuser)
        fuser.cores = {7: SimpleNamespace(name="acc", boundary_grid=SimpleNamespace(armed=True)),
                       8: SimpleNamespace(name="core", boundary_grid=None)}
        op = SimpleNamespace(name="add")
        fuser._refuse_host_fed(np.array([IRSource(node_id=8, index=0)], dtype=object), op)
        with pytest.raises(LaneFusionError, match="reads host values on an entry grid"):
            fuser._refuse_host_fed(np.array([IRSource(node_id=7, index=0)], dtype=object), op)


class TestM2NeuronSplit:
    @pytest.mark.parametrize("geometry", [dict(axons=256, neurons=8, count=99999),
                                          dict(axons=256, neurons=12, count=99999),
                                          dict(axons=256, neurons=8, count=12, passes=4096)],
                             ids=["resident_8", "resident_12", "scheduled_8"])
    def test_r3b_a_split_fused_add_core_certifies(self, geometry):
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform())
        mapping = _certify(program, real_inputs(program, 16), **geometry)
        split = sum(1 for s in mapping.stages if s.kind == "neural" for hc in s.hard_core_mapping.cores
                    for p in hc.matrix_placements if p.neurons < 16 and p.axons > 100)
        assert split > 0, "the geometry splits the 16-column fused-add cores"

    def test_a_fragment_turns_its_siblings_operand_rows_off(self):
        from mimarsinan.mapping.packing.softcore.fragments import sibling_operand_rows
        from mimarsinan.mapping.ir.lane_law import LaneLaw

        law = LaneLaw.of(LaneLaw.blank(4), operand_row=np.array([10, 11, -1, 13]))
        assert sibling_operand_rows(law, 0, 2) == {13} and sibling_operand_rows(law, 2, 4) == {10, 11}
        assert sibling_operand_rows(None, 0, 2) == set()


class TestM3IdentityOutputBesideAnOffset:
    def test_r2_the_identity_law_writes_m0_acc_plus_b_unclamped(self, tmp_path):
        from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
        from unit.mapping.lanes.lane_programs import write_state

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 4)).eval()
        with torch.no_grad():
            model[0].weight.abs_()  # every weight positive: |sum W (k - o/step)| reaches 22.5 sum W
        platform = lane_platform(None, activation_bits=5, weight_scale_rule="fractional", bias_resolution="law")
        platform["readout_law"]["output_law"] = "identity"

        def declare(flow, entries, grids, perceptrons, quantizers, lanes):
            fx = {id(p): n for n, p in perceptron_fx_names(flow.get_mapper_repr()).items()}
            document = {fx[id(entries[0])]: {"entry": {"scale": 6.0, "offset": 3.0, "rounding": "floor",
                                                       "codes": [-15, 15]}}}
            platform["quantizer_state"] = write_state(tmp_path / "state.json", document)

        program = build_lane_program(model, (8,), num_classes=4, platform=platform, hardware_bias=True,
                                     state_hook=declare)
        core = [n for n in program.ir.nodes if isinstance(n, NeuralCore)][0]
        law, grid = core.readout_law, core.boundary_grid
        x = torch.full((2, 8), -20.0, dtype=torch.float64)  # every entry code -15
        x[1] = 1.0
        mapping = _certify(program, x, axons=64, neurons=8)
        codes, _steps = run_golden(mapping, x.numpy())
        weight = np.asarray(core.get_core_matrix(program.ir), dtype=np.int64)
        acc = grid.entry_codes(x.numpy()) @ weight
        expected = law.m0.astype(np.int64) * acc + law.bias.astype(np.int64)  # shift 0, no clamp
        assert np.array_equal(codes, expected), "no identity output clamps"
        assert int(np.abs(codes).max()) < int(law.hi[0])


    def test_a_narrow_adc_spans_what_the_folded_bias_leaves_unsaturated(self):
        """The ADC reach of a symmetric law reads the folded bias: with B = 0 every accumulator beyond 7 saturates
        the clamp of 7 (an ADC step of 1 spans it); the fold's B = -20 moves the unsaturated window to [-13, 27],
        which a 7-level ADC spans only at step 4."""
        from mimarsinan.mapping.export.entry_offsets import refit_folded_law
        from mimarsinan.mapping.ir.core_law import CoreLaw

        one = np.ones(1)
        law = CoreLaw(input_step=one, m0=np.array([1]), shift=np.array([0]), bias=np.array([-20]),
                      lo=np.array([-7]), hi=np.array([7]), output_step=one, adc_shift=np.array([0]),
                      adc_levels=np.array([7]))
        refit = refit_folded_law(law, [100], identity=False, reach_hi=7, core="c")
        assert refit.adc_shift.tolist() == [2] and refit.lo.tolist() == [-7] and refit.hi.tolist() == [7]
        assert refit_folded_law(law, [100], identity=False, reach_hi=None, core="c") is law


class TestM4PruningBesideAnOffsetGrid:
    @pytest.mark.parametrize("classes", [None, ("L", "E", "P")], ids=["no_lanes", "lanes"])
    def test_r6_a_pruned_program_reading_an_offset_entry_grid_certifies(self, tmp_path, classes):
        from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
        from unit.mapping.lanes.lane_programs import write_state

        platform = lane_platform(classes, activation_bits=5)

        def declare(flow, entries, grids, perceptrons, quantizers, lanes):
            fx = {id(p): n for n, p in perceptron_fx_names(flow.get_mapper_repr()).items()}
            document = {fx[id(entries[1])]: {"entry": {"scale": 3.0, "offset": 0.5, "rounding": "floor",
                                                       "codes": [0, 15]}}}
            platform["quantizer_state"] = write_state(tmp_path / "state.json", document)

        program = build_lane_program(lenet5(), (1, 28, 28), num_classes=10, prune=PRUNE, platform=platform,
                                     state_hook=declare)
        _certify(program, real_inputs(program, 16), axons=1024, neurons=512)

    def test_a_pruned_program_beside_an_offset_lane_whose_zero_is_no_code_is_refused(self):
        from types import SimpleNamespace

        from mimarsinan.mapping.lanes.law.lane_grids import LaneGrid, PrunedOffsetGridError, refuse_pruned_offset_lanes

        pruned = SimpleNamespace(pre_pruning_row_mask=[False, True], pre_pruning_col_mask=[False])
        plain = SimpleNamespace()
        owners = {3: LaneGrid(range=7.0, offset=0.25, unsigned=True)}  # step 1 at 3 bits (codes 0..7): 0 is no code
        refuse_pruned_offset_lanes([plain], owners, 3)  # not pruned: nothing to refuse
        refuse_pruned_offset_lanes([pruned], {3: LaneGrid(range=7.0, offset=2.0, unsigned=True)}, 3)  # 0 is code 2
        with pytest.raises(PrunedOffsetGridError, match="value 0 is no code"):
            refuse_pruned_offset_lanes([pruned], owners, 3)


class TestMinorRefusals:
    def test_m1_a_law_bias_on_an_always_on_row_is_refused_by_name(self):
        from mimarsinan.mapping.support.grid_declarations import GridDeclarationError

        platform = lane_platform(weight_scale_rule="fractional", bias_resolution="law")
        with pytest.raises(GridDeclarationError, match="always-on axon row"):
            build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=platform,
                               hardware_bias=False)

    def test_m2_an_operand_of_two_grids_in_one_channel_is_refused_by_name(self):
        class SpatialConcatOperand(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(41)
                self.stem, self.relu = nn.Conv2d(3, 8, 3, 1, 1), nn.ReLU()
                self.conv = nn.Conv2d(8, 8, 3, 1, 1)
                self.a, self.ra = nn.Conv2d(8, 8, (2, 1), stride=(2, 1)), nn.ReLU()
                self.b = nn.Conv2d(8, 8, (2, 1), stride=(2, 1))
                self.relu2, self.gap, self.fc = nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Linear(8, 10)

            def forward(self, x):
                h = self.relu(self.stem(x))
                operand = torch.cat([self.ra(self.a(h)), self.b(h)], dim=2)
                return self.fc(torch.flatten(self.gap(self.relu2(self.conv(h) + operand)), 1))

        with pytest.raises(ToolchainRefusal, match="disagrees with another core's law"):
            build_lane_program(SpatialConcatOperand().eval(), (3, 8, 8), num_classes=10, platform=lane_platform())

    def test_m3_a_core_type_wider_than_16_bits_keeps_its_codes(self):
        from mimarsinan.mapping.export.chip_quantize import code_dtype

        assert code_dtype(4) == np.int8 and code_dtype(16) == np.int16 and code_dtype(20) == np.int32
        torch.manual_seed(43)
        wide = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
        platform = lane_platform(cores=[{"max_axons": 64, "max_neurons": 32, "count": 8, "weight_bits": 20}],
                                 layer_core_types=[{"layers": ["*"], "weight_bits": 20}])
        program = build_lane_program(wide, (8,), num_classes=4, weight_bits=4, platform=platform)
        widest = max(int(np.abs(np.asarray(c.get_core_matrix(program.ir), dtype=np.int64)).max())
                     for c in cores(program))
        assert widest > (1 << 15), "20-bit codes beyond int16 are stored whole"
        x = real_inputs(program, 8)
        _certify(program, x, axons=64, neurons=32, widths=[20])

    def test_m4_the_fold_bound_counts_the_operand_term(self):
        from mimarsinan.mapping.export.entry_offsets import refit_folded_law
        from mimarsinan.mapping.ir.lane_law import LaneLaw
        from mimarsinan.models.nn.activations.integer_law import FP64_EXACT_BOUND, IntegerLawRangeError

        law = dataclasses.replace(LaneLaw.of(LaneLaw.blank(1), operand_row=np.array([0])), m0=np.array([1]),
                                  bias=np.array([FP64_EXACT_BOUND - 10]), adc_shift=np.array([0]),
                                  adc_levels=np.array([1]))
        refit_folded_law(law, [1], identity=False, reach_hi=None, core="c")  # 1 * (1 + 1) + bias < 2^53
        with pytest.raises(IntegerLawRangeError, match="2\\^53"):
            refit_folded_law(law, [1], identity=False, reach_hi=None, core="c", extra=100)


class TestPipelineCertificates:
    def test_m6_a_lane_program_keeps_its_r_edge_armed(self):
        from types import SimpleNamespace

        from mimarsinan.mapping.lanes.ir.fusion import LANE_REPORT
        from mimarsinan.pipelining.core.gates.value_gates import LaneParityDisarmedError, run_model_value_parity_gate

        pipeline = SimpleNamespace(config={"value_parity_samples": 0})
        run_model_value_parity_gate(pipeline, None, SimpleNamespace())  # a lane-free program may disarm it
        with pytest.raises(LaneParityDisarmedError, match="value_parity_samples >= 1"):
            run_model_value_parity_gate(pipeline, None, SimpleNamespace(**{LANE_REPORT: {"adds": 1}}))

    def test_f9_a_lane_free_scheduled_segment_of_chained_cores_compares_its_carries(self):
        """The C-edge's exposure is unconditional: a lane-free value program whose scheduled segment chains cores
        (two convolutions and a linear head with no host op between) has its pass carries compared, where the plain
        identity reference refuses them as backend-only nodes."""
        from mimarsinan.certification.value_certificate import certify_twin_flow_values, exposed_nodes
        from mimarsinan.torch_mapping.value_parity.value_program import fp64_identity_program

        class Chain(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(5)
                self.a, self.b = nn.Conv2d(2, 4, 3, padding=1), nn.Conv2d(4, 4, 3, padding=1)
                self.fc = nn.Linear(4 * 6 * 6, 4)

            def forward(self, x):
                return self.fc(torch.flatten(self.b(self.a(x)), 1))

        program = build_lane_program(Chain().eval(), (2, 6, 6), num_classes=4, platform=lane_platform(None))
        assert host_ops(program) == [] and not any(getattr(c, OPERAND_ROWS, None) is not None for c in cores(program))
        mapping = program.packed_mapping(axons=160, neurons=16, count=3, passes=64)
        assert sum(s.kind == "neural" for s in mapping.stages) > 1, "the segment runs in several passes"
        backend = ValueHybridCoreFlow(mapping, dtype=torch.float64)
        x = real_inputs(program, 8)
        plain, detail = certify_twin_flow_values(fp64_identity_program(program.ir), backend, x, atol=0.0)
        assert not plain.passed and "backend-only=[]" not in detail
        cert, detail = certify_twin_flow_values(fp64_identity_program(program.ir, exposed=exposed_nodes(mapping)),
                                                backend, x, atol=0.0)
        assert cert.passed and detail.endswith("backend-only=[]"), detail
