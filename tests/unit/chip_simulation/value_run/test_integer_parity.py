"""E284: NF = SCM = HCM at atol 0 on law-carrying programs; the tie rule is observable and the integer
accumulator, not the fp32 perceptron output, is what the law reads."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import mimarsinan.models.nn.activations.integer_law as law_module
from unit.chip_simulation.value_run.integer_programs import (
    INSTANCES,
    LAW,
    assert_exact,
    build_integer_program,
    integer_inputs,
    lenet5,
)


@pytest.fixture(scope="module", params=sorted(INSTANCES))
def instance(request):
    build, packed = INSTANCES[request.param]
    return request.param, build(), packed


def _run(flow, x):
    with torch.no_grad():
        return flow(x)


class TestNfScmHcm:
    def test_nf_equals_scm_equals_hcm_at_atol_zero(self, instance):
        _name, program, packed = instance
        x = integer_inputs(program, 64).double()
        scm = _run(program.identity(), x)
        assert_exact(_run(program.nf(), x), scm)
        assert_exact(_run(program.packed(**packed), x), scm)


def _tie_program(weight: float, theta: float, boundary_scale: float, readout_range: float):
    model = nn.Sequential(nn.Linear(1, 1, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.fill_(weight)
    return build_integer_program(
        model, (1,), num_classes=1, boundary_scales=[boundary_scale], readout_ranges=[readout_range],
        parameter_scales=[theta],
    )


class TestRoundingWitnesses:
    def test_the_half_away_tie_flips_by_one_lsb_under_half_to_even(self, monkeypatch):
        # Δin = 1, θ = 4, Δout = 1/2, W = 1: y/Δout = q/2, so every odd input code sits on an rnd_s tie.
        program = _tie_program(weight=0.25, theta=4.0, boundary_scale=127.0, readout_range=63.5)
        law = next(n for n in program.ir.nodes if hasattr(n, "readout_law")).readout_law
        assert float(law.input_step[0]) == 1.0 and float(law.output_step[0]) == 0.5
        q = torch.arange(-9, 10, dtype=torch.float64).reshape(-1, 1)
        away = _run(program.identity(), q) / 0.5
        assert away.flatten().tolist() == [float(law_module.round_half_away(torch.tensor(v / 2.0)))
                                           for v in range(-9, 10)]

        monkeypatch.setattr(law_module, "round_half_away", torch.round)
        even = _run(program.identity(), q) / 0.5
        moved = (away - even).abs()
        assert float(moved.max()) == 1.0 and int((moved > 0).sum()) > 0
        assert bool((moved[q.remainder(2) == 0] == 0).all()), "only ties move"

    def test_a_near_tie_reads_the_integer_accumulator_and_the_fp32_output_form_flips(self):
        # W = 7/3 (its fp32 weight lies below 7/3), θ = 3, Δin = 3/2, Δout = 1: the accumulator 7q sits exactly
        # on a tie (3.5 q) for odd q, while the fp32 perceptron output lies just below it.
        program = _tie_program(weight=7.0 / 3.0, theta=3.0, boundary_scale=1.5 * 127, readout_range=127.0)
        core = next(n for n in program.ir.nodes if hasattr(n, "readout_law"))
        assert int(core.readout_law.m0[0]) == 1 << 15 and int(core.readout_law.shift[0]) == 16
        q = torch.tensor([[9.0], [17.0], [-9.0], [-17.0]], dtype=torch.float64)
        x = q * 1.5
        hcm = _run(program.packed(axons=8, neurons=8), x)
        assert_exact(_run(program.nf(), x), hcm)
        assert hcm.flatten().tolist() == [32.0, 60.0, -32.0, -60.0]
        weight32 = torch.tensor(7.0 / 3.0, dtype=torch.float32)
        fp32_output = (weight32 * x.float()).double()
        assert bool((fp32_output.abs() < (3.5 * q).abs()).all())
        fp32_form = torch.clamp(law_module.round_half_away(fp32_output), -127, 127)
        assert fp32_form.flatten().tolist() == [31.0, 59.0, -31.0, -59.0]


class TestAdcInsideTheCertifiedProgram:
    def test_a_narrow_adc_changes_the_function_and_stays_exact(self):
        wide = build_integer_program(lenet5(), (1, 28, 28), num_classes=10)
        narrow = build_integer_program(lenet5(), (1, 28, 28), num_classes=10,
                                       readout_law={**LAW, "adc_bits": 6})
        assert any(int(n.readout_law.adc_shift.max()) > 0 for n in narrow.ir.nodes if hasattr(n, "readout_law"))
        x = integer_inputs(wide, 32).double()
        scm_wide, scm_narrow = _run(wide.identity(), x), _run(narrow.identity(), x)
        assert not torch.equal(scm_wide, scm_narrow), "adc_bits must be function-bearing"
        assert_exact(_run(narrow.nf(), x), scm_narrow)
        assert_exact(_run(narrow.packed(axons=1024, neurons=512), x), scm_narrow)
