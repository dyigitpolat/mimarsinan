"""WI-5a to WI-5d on the integer value path: entry offsets folded into the law, the fractional weight scale, the bias
at the law's resolution and the identity output law, each certified NF = SCM = HCM = int64 golden at atol 0."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.certification.value_certificate import VALUE_TWIN_FP64_ATOL
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph, verify_ir_graph_quantized
from mimarsinan.mapping.export.entry_offsets import (
    core_entry_offset,
    fold_entry_offset,
    host_row_weight_sums,
    offset_multipliers,
    unfold_entry_offset,
)
from mimarsinan.mapping.export.law_grids import CoreGrid
from mimarsinan.mapping.export.law_stamp import core_channels
from mimarsinan.mapping.export.law_solver import (
    ReadoutLawDeclarationError,
    ReadoutLawSpec,
    core_bias_and_data_weights,
)
from mimarsinan.mapping.export.output_law import OutputLawError, identity_output_multiplier, solve_identity_law
from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.mapping.support.boundary_grids import widest_boundary_step
from mimarsinan.mapping.support.grid_declarations import ParameterGrids
from mimarsinan.models.nn.activations.integer_law import IntegerLawRangeError
from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from unit.chip_simulation.value_run.integer_programs import (
    assert_exact,
    conv1d_net,
    lenet5,
    mlp,
    residual_block_net,
    resnet18_quarter,
)
from unit.mapping.declared_programs import INPUT_OFFSET, INPUT_SCALE, build_declared_program, real_inputs

CASES = {
    "mlp": (mlp, (8,), 4, dict(axons=256, neurons=5)),
    "lenet5": (lenet5, (1, 28, 28), 10, dict(axons=1024, neurons=512)),
    "residual_block": (residual_block_net, (3, 16, 16), 10, dict(axons=256, neurons=32)),
    "conv1d": (conv1d_net, (2, 16), 5, dict(axons=128, neurons=16)),
}


@pytest.fixture(scope="module")
def programs(tmp_path_factory):
    built = {}
    for name, (factory, shape, classes, geometry) in CASES.items():
        path = tmp_path_factory.mktemp(name) / "state.json"
        program, platform, declaration = build_declared_program(factory(), shape, num_classes=classes,
                                                               state_path=path)
        built[name] = (program, platform, geometry)
    return built


def _cores(program):
    return [n for n in program.ir.nodes if isinstance(n, NeuralCore)]


class TestCertifiedAtAtolZero:
    @pytest.mark.parametrize("name", list(CASES))
    def test_nf_scm_hcm_and_the_golden_agree(self, programs, name):
        program, _platform, geometry = programs[name]
        x = real_inputs(program)
        with torch.no_grad():
            nf = program.nf()(x)
            scm = program.identity()(x)
            packed_mapping = program.packed_mapping(**geometry)
            from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
            hcm = ValueHybridCoreFlow(packed_mapping, dtype=torch.float64)(x)
        assert_exact(nf, scm)
        assert_exact(scm, hcm)
        codes, steps = run_golden(packed_mapping, x.numpy())
        assert_exact(torch.as_tensor(codes * steps), hcm)

    def test_the_multi_pass_program_agrees_too(self, programs):
        program, _platform, _geometry = programs["lenet5"]
        x = real_inputs(program, batch=8)
        mapping = program.packed_mapping(axons=1024, neurons=256, count=3, passes=128)
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
        with torch.no_grad():
            assert_exact(ValueHybridCoreFlow(mapping, dtype=torch.float64)(x), program.identity()(x))
        codes, steps = run_golden(mapping, x.numpy())
        assert_exact(torch.as_tensor(codes * steps), program.identity()(x))


class TestEntryOffsets:
    def test_the_input_cores_carry_the_declared_offset_grid(self, programs):
        program, *_ = programs["residual_block"]
        first = [c for c in _cores(program) if c.perceptron_index == 0]
        assert first and all(core_entry_offset(c) == INPUT_OFFSET for c in first)
        grid = first[0].boundary_grid
        assert isinstance(grid, DeclaredBoundaryGrid) and grid.rounding == "floor" and grid.code_range == (0, 15)
        assert all(core_entry_offset(c) == 0.0 for c in _cores(program) if c.perceptron_index != 0)

    def test_zero_padding_gives_border_positions_their_own_bias(self, programs):
        program, *_ = programs["residual_block"]
        stem = [c for c in _cores(program) if c.perceptron_index == 0]
        sums = {host_row_weight_sums(c, program.ir).tobytes() for c in stem}
        laws = {c.readout_law.digest() for c in stem}
        assert len(sums) > 1 and len(laws) == len(sums)

    def test_the_fold_is_the_real_accumulator_on_the_shifted_codes(self, programs):
        program, *_ = programs["mlp"]
        core = next(c for c in _cores(program) if c.perceptron_index == 0)
        law = core.readout_law
        sums = host_row_weight_sums(core, program.ir)
        unfolded = unfold_entry_offset(law, core, program.ir)
        multipliers = offset_multipliers(unfolded, float(core.threshold), INPUT_OFFSET)
        assert np.array_equal(law.bias, unfolded.bias - multipliers * sums)
        assert fold_entry_offset(unfolded, core, program.ir).digest() == law.digest()
        # M_o 2^-s is offset / (theta out_step) to within one half unit at shift s.
        target = np.ldexp(INPUT_OFFSET / (float(core.threshold) * law.output_step), law.shift)
        assert np.all(np.abs(multipliers - target) <= 0.5)

    def test_the_offset_function_is_the_float_function_of_the_snapped_input(self, programs):
        program, *_ = programs["mlp"]
        model = program.nf()
        first = next(iter(model.get_perceptrons()))
        x = real_inputs(program, batch=64)
        grid = DeclaredBoundaryGrid(scale=INPUT_SCALE, bits=5, offset=INPUT_OFFSET, rounding="floor",
                                    code_min=0, code_max=15)
        v = grid.snap(x)
        transformer = PerceptronTransformer()
        w = transformer.get_effective_weight(first).double()
        b = transformer.get_effective_bias(first).double()
        real = v @ w.T + b
        with torch.no_grad():
            deployed = first(x)
        step = float(first.scaler.stamped.law["output_step"][0])
        levels = 15
        expected = torch.clamp(torch.round(real / step), -levels, levels) * step
        differ = (deployed - expected).abs() > 1e-9
        assert float(differ.double().mean()) < 0.01
        assert float((deployed - expected).abs().max()) <= step + 1e-9


class TestIdentityOutputLaw:
    def test_output_cores_write_exact_integers(self, programs):
        program, platform, geometry = programs["lenet5"]
        mult_bits = platform["readout_law"]["mult_bits"]
        output_index = max(c.perceptron_index for c in _cores(program))
        outputs = [c for c in _cores(program) if c.perceptron_index == output_index]
        for core in outputs:
            law = core.readout_law
            assert np.all(law.shift == 0) and np.all(law.m0 == identity_output_multiplier(mult_bits))
            assert np.all(law.hi >= 1 << 20)
        x = real_inputs(program, batch=8)
        codes, steps = run_golden(program.packed_mapping(**geometry), x.numpy())
        assert np.abs(codes).max() > 1 << 20

    def test_logits_are_the_float_logits_to_within_the_bias_resolution(self, programs):
        program, *_ = programs["mlp"]
        model = program.nf()
        last = list(model.get_perceptrons())[-1]
        captured = {}
        def capture(_module, args):
            captured.setdefault("x", args[0])  # the first call; the law's integer rerun calls it again

        handle = last.register_forward_pre_hook(capture)
        x = real_inputs(program, batch=16)
        with torch.no_grad():
            logits = model(x)
        handle.remove()
        transformer = PerceptronTransformer()
        w = transformer.get_effective_weight(last).double()
        b = transformer.get_effective_bias(last).double()
        snapped = last.input_activation(captured["x"]) if not isinstance(last.input_activation, nn.Identity) else captured["x"]
        real = snapped.double() @ w.T + b
        # The law reads the integer weights (fp32 weights times theta); the float reference reads the fp32 weights.
        assert float((logits - real).abs().max()) < 1e-5

    def test_the_identity_law_refuses_a_bound_beyond_fp64(self):
        grid = CoreGrid(input_step=0.1, input_levels=15, output_step=0.2, output_range=3.0)
        spec = ReadoutLawSpec(adc_bits=64, mult_bits=24, activation_bits=5, weight_bits=4)
        law = solve_identity_law(core="c", theta=5.0, bias=np.array([1.0, -2.0]), acc_bounds=[1 << 20, 3],
                                 grid=grid, spec=spec)
        assert law.m0.tolist() == [1 << 23] * 2 and law.shift.tolist() == [0, 0]
        assert law.output_step.tolist() == [0.1 / (5.0 * (1 << 23))] * 2
        assert law.bias.tolist() == [round((1 << 23) * 1.0 / 0.1), round(-(1 << 23) * 2.0 / 0.1)]
        with pytest.raises(IntegerLawRangeError, match="2\\^53"):
            solve_identity_law(core="c", theta=5.0, bias=np.zeros(1), acc_bounds=[1 << 31], grid=grid, spec=spec)

    def test_an_output_read_by_another_core_refuses_by_name(self, tmp_path):
        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b = nn.Linear(8, 6), nn.Linear(6, 3)

            def forward(self, x):
                y = self.a(x)
                return torch.cat([y, self.b(y)], dim=1)

        torch.manual_seed(0)
        with pytest.raises(OutputLawError, match="'a'") as info:
            build_declared_program(TwoHeads().eval(), (8,), num_classes=9, state_path=tmp_path / "s.json",
                                   declare_ranges=False)
        assert isinstance(info.value, ToolchainRefusal)

    def test_an_identity_law_with_no_output_core_refuses_by_name(self, tmp_path):
        model = nn.Sequential(nn.Linear(8, 6), nn.ReLU()).eval()
        with pytest.raises(OutputLawError, match="no core writes"):
            build_declared_program(model, (8,), num_classes=6, state_path=tmp_path / "s.json", declare_ranges=False)

    def test_requantize_is_the_absent_meaning(self, tmp_path):
        program, *_ = build_declared_program(mlp(), (8,), num_classes=4, state_path=tmp_path / "s.json",
                                             output_law=None)
        last = max(c.perceptron_index for c in _cores(program))
        assert all(np.all(c.readout_law.hi == 15) for c in _cores(program) if c.perceptron_index == last)


class TestFractionalScaleAndLawBias:
    def _perceptron(self, scale=0.37):
        torch.manual_seed(4)
        perceptron = Perceptron(5, 6)
        with torch.no_grad():
            perceptron.layer.weight.mul_(scale)
            perceptron.layer.bias.copy_(torch.tensor([3.0, -2.5, 0.01, 0.0, 1.7]))
        perceptron.eval()
        return perceptron

    def test_napq_fractional_scale_maps_the_peak_onto_q_max(self):
        perceptron = self._perceptron()
        peak = float(perceptron.layer.weight.abs().max())
        NormalizationAwarePerceptronQuantization(4, "cpu", 1.0, fractional_scale=True,
                                                 float_bias=True).transform(perceptron)
        scale = float(perceptron.parameter_scale)
        assert scale == pytest.approx(7.0 / peak, rel=1e-6) and scale != math.floor(scale)
        codes = perceptron.layer.weight.detach() * perceptron.parameter_scale
        assert torch.allclose(codes, torch.round(codes), atol=1e-4)
        assert float(torch.round(codes).abs().max()) == 7.0

    def test_napq_float_bias_leaves_the_bias_untouched(self):
        perceptron = self._perceptron()
        bias = perceptron.layer.bias.detach().clone()
        NormalizationAwarePerceptronQuantization(4, "cpu", 1.0, fractional_scale=True,
                                                 float_bias=True).transform(perceptron)
        assert torch.equal(perceptron.layer.bias.detach(), bias)

    def test_the_default_projection_is_unchanged(self):
        a, b = self._perceptron(), self._perceptron()
        NormalizationAwarePerceptronQuantization(4, "cpu", 1.0).transform(a)
        NormalizationAwarePerceptronQuantization(4, "cpu", 1.0, fractional_scale=False,
                                                 float_bias=False).transform(b)
        assert float(a.parameter_scale) == math.floor(float(a.parameter_scale))
        for pa, pb in zip(a.state_dict().values(), b.state_dict().values()):
            assert torch.equal(pa, pb)

    def _core(self, threshold_scale, bias):
        core = NeuralCore(id=0, name="c", input_sources=np.array([IRSource(-2, i) for i in range(4)]),
                          core_matrix=np.array([[0.5, -0.25], [0.125, 0.75], [-0.5, 0.25], [1.0, -1.0]]),
                          hardware_bias=np.asarray(bias, dtype=np.float64))
        core.parameter_scale = torch.tensor(threshold_scale)
        return core

    def _graph(self, core):
        from mimarsinan.mapping.ir import IRGraph
        return IRGraph(nodes=[core], output_sources=np.array([IRSource(0, 0), IRSource(0, 1)]))

    def test_a_fractional_trained_scale_deploys_under_the_declaration(self):
        core = self._core(7.0 / 1.0 * 0.9137, [0.3, -1.25])
        ir = self._graph(core)
        quantize_ir_graph(ir, 4, weight_quantization=True,
                          grids=ParameterGrids(fractional_scale=True, law_bias=True))
        assert core.threshold == pytest.approx(7.0 * 0.9137)
        assert core.hardware_bias.dtype == np.float64
        assert np.allclose(core.hardware_bias, np.array([0.3, -1.25]) * core.threshold)
        verify_ir_graph_quantized(ir, 4, grids=ParameterGrids(fractional_scale=True, law_bias=True))
        with pytest.raises(AssertionError):
            verify_ir_graph_quantized(ir, 4)

    def test_without_the_declaration_a_fractional_scale_still_refuses(self):
        core = self._core(6.3951, [0.3, -1.25])
        with pytest.raises(ValueError, match="integer lattice"):
            quantize_ir_graph(self._graph(core), 4, weight_quantization=True)

    def test_the_float_path_adds_the_float_bias(self, tmp_path):
        """No readout law: the bias stays the trained float (no grid, no rounding) on fractional-scale cores read
        through the declared offset entry grid, and the program certifies at the float path's classes."""
        factory, shape, classes, geometry = CASES["residual_block"]
        program, platform, _ = build_declared_program(factory(), shape, num_classes=classes,
                                                      state_path=tmp_path / "s.json", readout_law=False,
                                                      declare_ranges=False)
        assert "readout_law" not in platform
        cores = _cores(program)
        assert not any(getattr(c, "readout_law", None) is not None for c in cores)
        assert any(isinstance(c.boundary_grid, DeclaredBoundaryGrid) and c.boundary_grid.offset == INPUT_OFFSET
                   for c in cores)
        assert any(float(c.threshold) != math.floor(float(c.threshold)) for c in cores)
        biased = [c for c in cores if c.hardware_bias is not None]
        assert biased and all(np.asarray(c.hardware_bias).dtype == np.float64 for c in biased)
        assert any(not np.array_equal(c.hardware_bias, np.round(c.hardware_bias)) for c in biased)
        order, _, _ = program.model.get_mapper_repr()._ensure_exec_graph()
        trained = {int(n.perceptron_index): n.perceptron.layer.bias.detach().double().numpy() for n in order
                   if getattr(n, "perceptron", None) is not None and getattr(n, "perceptron_index", None) is not None}
        for core in biased:
            want = trained[int(core.perceptron_index)][core_channels(core)]
            assert np.allclose(np.asarray(core.hardware_bias) / float(core.threshold), want, rtol=1e-12, atol=0.0)
        x = real_inputs(program, batch=8)
        with torch.no_grad():
            scm = program.identity()(x)
            hcm = program.packed(**geometry)(x)
            nf = program.nf()(x)
        assert float((hcm - scm).abs().max()) <= VALUE_TWIN_FP64_ATOL
        assert float((nf - scm).abs().max()) <= widest_boundary_step(program.ir)
        assert torch.equal(nf.argmax(1), scm.argmax(1))

    def test_a_real_bias_reaches_the_solver_only_under_the_declaration(self):
        core = self._core(6.3951, [0.3, -1.25])
        ir = self._graph(core)
        quantize_ir_graph(ir, 4, weight_quantization=True, grids=ParameterGrids(True, True))
        with pytest.raises(ReadoutLawDeclarationError, match="weight grid"):
            core_bias_and_data_weights(core, ir)
        bias, _weights = core_bias_and_data_weights(core, ir, real_bias=True)
        assert np.array_equal(bias, core.hardware_bias)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_resnet18_quarter_with_every_declaration_certifies(tmp_path):
    program, _platform, _declaration = build_declared_program(resnet18_quarter(), (3, 32, 32), num_classes=10,
                                                              state_path=tmp_path / "s.json")
    x = real_inputs(program, batch=4)
    from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
    with torch.no_grad():
        scm = program.identity()(x)
        assert_exact(program.nf()(x), scm)
        mapping = program.packed_mapping(axons=256, neurons=256, coalescing=True, count=99999)
        assert_exact(ValueHybridCoreFlow(mapping, dtype=torch.float64)(x), scm)
    codes, steps = run_golden(mapping, x.numpy())
    assert_exact(torch.as_tensor(codes * steps), scm)
