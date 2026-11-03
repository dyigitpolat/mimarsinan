"""E283: the value executor equals the int64 golden at atol 0 on law-carrying programs; f_adc and the 2^53 bound."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import mimarsinan.chip_simulation.value_run.int64_golden as golden_module
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import evaluate, run_golden
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.integer_law import (
    FP64_EXACT_BOUND,
    IntegerLawRangeError,
    accumulator_bounds,
    adc_int64,
    apply_law_int64,
    grid_levels,
    law_magnitude_bound,
)
from unit.chip_simulation.value_run.integer_programs import (
    INSTANCES,
    assert_exact,
    build_integer_program,
    integer_inputs,
    resnet18_quarter,
)

@pytest.fixture(scope="module", params=sorted(INSTANCES))
def instance(request):
    build, packed = INSTANCES[request.param]
    return request.param, build(), packed


def _executor(hybrid, x_int):
    with torch.no_grad():
        return ValueHybridCoreFlow(hybrid, dtype=torch.float64, device="cpu")(x_int.double())


def _golden_values(hybrid, x_int):
    codes, steps = run_golden(hybrid, x_int)
    return torch.as_tensor(codes * steps)


class TestExecutorEqualsGolden:
    def test_identity_program(self, instance):
        _name, program, _packed = instance
        x_int = integer_inputs(program, 64)
        hybrid = program.identity().hybrid_mapping
        assert_exact(_executor(hybrid, x_int), _golden_values(hybrid, x_int))

    def test_packed_program(self, instance):
        name, program, packed = instance
        x_int = integer_inputs(program, 64)
        hybrid = program.packed_mapping(**packed)
        if name == "pruned_multi_pass":
            assert sum(1 for s in hybrid.stages if s.kind == "neural") > 10
        assert_exact(_executor(hybrid, x_int), _golden_values(hybrid, x_int))

    def test_golden_codes_are_int64(self, instance):
        _name, program, _packed = instance
        codes = evaluate(program.identity().hybrid_mapping, integer_inputs(program, 4))
        assert codes.dtype == np.int64


class TestRealValuedInputs:
    """The golden reads host values as the executor does: the entry snap rounds them onto the boundary grid."""

    @pytest.mark.parametrize("name", ["lenet5", "residual_block"])
    def test_real_valued_inputs_equal_the_fp64_executor(self, name):
        build, packed = INSTANCES[name]
        program = build()
        x = torch.randn(16, *program.input_shape, generator=torch.Generator().manual_seed(4),
                        dtype=torch.float64) * 2.0
        for hybrid in (program.identity().hybrid_mapping, program.packed_mapping(**packed)):
            assert_exact(_executor(hybrid, x), _golden_values(hybrid, x))
        assert np.array_equal(evaluate(program.identity().hybrid_mapping, x.numpy()), run_golden(
            program.identity().hybrid_mapping, x)[0])


class TestAdcIdentity:
    def test_adc_at_or_above_the_accumulator_width_leaves_every_accumulator_unchanged(self, instance, monkeypatch):
        _name, program, _packed = instance
        seen = {"calls": 0}
        law_int64 = golden_module.apply_law_int64

        def checked(acc, **law):
            assert np.array_equal(adc_int64(acc, law["adc_shift"], law["adc_levels"]), acc)
            seen["calls"] += 1
            return law_int64(acc, **law)

        monkeypatch.setattr(golden_module, "apply_law_int64", checked)
        run_golden(program.identity().hybrid_mapping, integer_inputs(program, 16))
        assert seen["calls"] > 0


def _q_min_stem():
    """A 3x3 single-channel stem at 4 weight bits: weights beyond q_max project onto q_min = -8."""
    model = nn.Sequential(nn.Conv2d(1, 1, 3, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.fill_(-10.0)
    program = build_integer_program(model, (1, 5, 5), num_classes=9, weight_bits=4, boundary_scales=[1.0],
                                    readout_ranges=[72.0], readout_law={"adc_bits": 32, "range": "calibrated",
                                                                        "mult_bits": 16})
    return program, torch.full((1, 1, 5, 5), 3.0, dtype=torch.float64), -8


def _q_min_linear():
    """65 rows at the 8-bit q_min code -128: the accumulator of a saturated input is 65 x 128 x 127."""
    model = nn.Sequential(nn.Linear(65, 1, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.fill_(-200.0)
    program = build_integer_program(model, (65,), num_classes=1, boundary_scales=[1.0], readout_ranges=[8320.0],
                                    parameter_scales=[1.0])
    return program, torch.full((1, 65), 5.0, dtype=torch.float64), -128


def _data_rows(core):
    return np.array([not (s.is_always_on() or s.is_off()) for s in core.input_sources.flatten()])


class TestAdcIdentityAtQMin:
    """An ADC declared at or above A is the identity on every reachable accumulator, q_min codes included."""

    @pytest.mark.parametrize("build", [_q_min_stem, _q_min_linear])
    def test_the_worst_case_accumulator_passes_the_identity_adc_unchanged(self, build):
        program, x, q_min = build()
        core = next(n for n in program.ir.nodes if isinstance(n, NeuralCore))
        weights = np.asarray(core.get_core_matrix(program.ir)).astype(np.int64)
        law = core.readout_law
        assert int(weights[_data_rows(core)].max()) == q_min  # every data weight sits at the q_min code
        acc = (weights[_data_rows(core)] * grid_levels(8)).sum(axis=0, keepdims=True)  # every input at +L
        assert np.array_equal(adc_int64(acc, law.adc_shift, law.adc_levels), acc)
        assert np.array_equal(adc_int64(-acc, law.adc_shift, law.adc_levels), -acc)
        declared = apply_law_int64(acc, m0=law.m0, bias=law.bias, shift=law.shift, lo=law.lo, hi=law.hi)
        codes, steps = run_golden(program.identity().hybrid_mapping, x)
        with torch.no_grad():
            nf, scm = program.nf()(x), program.identity()(x)
        want = torch.as_tensor(np.broadcast_to(declared * law.output_step, codes.shape).copy())
        assert_exact(torch.as_tensor(codes * steps), want)
        assert_exact(nf, want)
        assert_exact(scm, want)

    def test_every_installed_identity_adc_holds_each_column_s_worst_case_accumulator(self, instance):
        _name, program, _packed = instance
        for core in program.ir.nodes:
            if not isinstance(core, NeuralCore):
                continue
            law = core.readout_law
            weights = np.asarray(core.get_core_matrix(program.ir)).astype(np.int64)[_data_rows(core)]
            worst = (np.abs(weights) * grid_levels(8)).sum(axis=0, keepdims=True)
            assert np.array_equal(adc_int64(worst, law.adc_shift, law.adc_levels), worst), core.name


class TestFp64Bound:
    def test_every_installed_law_lies_below_two_to_the_fifty_three(self, instance):
        _name, program, _packed = instance
        for core in program.ir.nodes:
            if not isinstance(core, NeuralCore):
                continue
            law = core.readout_law
            weights = np.asarray(core.get_core_matrix(program.ir))[_data_rows(core)]
            bound = law_magnitude_bound(accumulator_bounds(weights, grid_levels(8)), law.m0, law.bias, law.adc_shift)
            assert bound < FP64_EXACT_BOUND

    def test_a_witness_above_the_bound_refuses_by_name(self):
        wide = nn.Sequential(nn.Linear(2048, 4)).eval()
        with pytest.raises(IntegerLawRangeError, match="2\\^53"):
            build_integer_program(wide, (2048,), num_classes=4, activation_bits=16, weight_bits=16,
                                  readout_law={"adc_bits": 64, "range": "calibrated", "mult_bits": 24})


@pytest.mark.slow
@pytest.mark.timeout(1200)
class TestResNet18Quarter:
    def test_multi_pass_resnet18_quarter_equals_the_golden(self):
        program = build_integer_program(resnet18_quarter(), (3, 32, 32), num_classes=10)
        x_int = integer_inputs(program, 64)
        hybrid = program.packed_mapping(axons=1280, neurons=256, count=8, passes=1024)
        assert sum(1 for s in hybrid.stages if s.kind == "neural") > 100
        assert_exact(_executor(hybrid, x_int), _golden_values(hybrid, x_int))
        identity = program.identity().hybrid_mapping
        assert_exact(_executor(identity, x_int), _golden_values(identity, x_int))
