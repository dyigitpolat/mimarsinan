"""[U-PFOLD r2, the merge cut with U-PCH3] Per-neuron grids over a fold per position: the output axis from the reader.

U-PCH3's per-neuron grids (``weight_scale_granularity: per_neuron``: each output neuron projected on its own scale
``s_c``, ``theta_c = s_c``) read a weight, bias or code tensor per output neuron (``neuron_scales.row_peaks`` and
``scale_view``). Over a fold per position the one reader hands out weight rows ``(rows, out, in)`` and bias rows
``(positions, out)``, so the output neuron is no literal axis: every per-neuron helper takes the axis the reader
declares for the view (``position_fold.output_axes``), in Weight Quantization's projection and in Quantization
Verification's checks.

- A per-position perceptron whose positions are unequal to its outputs projects with no refusal.
- One whose positions equal its outputs, with a distinct peak per neuron and per position, takes each neuron's own peak
  (a read on the positions' axis takes the positions' peaks instead).
- On the token-mixing form, the generated tree 481's form and the convolution seam, with per-neuron grids: the NF and
  SCM parity is bit-exact (count family, both LIF semantics) and the walk equals the deployed read element by element
  in both families. The timing family reads seeds at which the older defect of ruling (h) of 2026-11-12 20:51 does
  not strike: with Weight Quantization the timing walk and the deployed program read some seeds of these forms apart
  under either grid, with a fold per input as well as per position, on the first revision's tree too, and per-neuron
  grids move which seeds it strikes (UF/report_r2.md).

Red on the first merge cut (FU/upfold/merge): the unequal forms refuse with an unnamed ValueError, the equal forms
take the positions' peaks, and Quantization Verification reads the rows' axis as the neurons'.
"""

from __future__ import annotations

import pytest
import torch

import unit.spiking.host_gather_forms as hg
import unit.spiking.position_fold_forms as pf

import mimarsinan.transformations.normalization_aware_perceptron_quantization as napq_module
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
    assert_effective_parameters_on_chip_grid,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.perceptron.position_fold import effective_parameters

BITS = hg.WQ_BITS
Q_MAX = 2 ** (BITS - 1) - 1
NAPQ = napq_module.NormalizationAwarePerceptronQuantization


class _PerNeuron(NAPQ):
    """The row's projection with ``weight_scale_granularity: per_neuron`` (U-PCH3's option), in the forms' sequences."""

    def __init__(self, *args, **kwargs):
        kwargs["per_neuron"] = True
        super().__init__(*args, **kwargs)


@pytest.fixture
def per_neuron_grids(monkeypatch):
    monkeypatch.setattr(napq_module, "NormalizationAwarePerceptronQuantization", _PerNeuron)
    monkeypatch.setattr(hg, "NormalizationAwarePerceptronQuantization", _PerNeuron)


def _reader(form: str, gauges=None, seed: int = 0):
    _model, flow, shape = pf.converted(form, seed)
    perceptrons = list(flow.get_perceptrons())
    drawn = pf.draw_gauges(flow, perceptrons, shape, seed, "one") if gauges is None else gauges(perceptrons)
    for p, g in zip(perceptrons, drawn):
        p.set_activation_scale(g)
    compute_per_source_scales(flow.get_mapper_repr())
    reader = perceptrons[-1]
    assert reader.per_input_scales.dim() == 2, "the reader's fold is held per position"
    return flow, reader


def _peaks_per_neuron(reader) -> torch.Tensor:
    """Each neuron's largest effective weight over every position and input, by explicit loops."""
    weights = effective_parameters(reader, PerceptronTransformer())[0].detach()  # (positions, out, in)
    return torch.tensor([max(float(weights[p, c].abs().max()) for p in range(int(weights.shape[0])))
                         for c in range(int(weights.shape[1]))])


@pytest.mark.parametrize("form", ("token_mixing_3", "conv_seam"))
def test_a_per_neuron_grid_over_positions_unequal_to_its_outputs_projects_without_refusal(form):
    _flow, reader = _reader(form)
    out, positions = int(reader.layer.weight.shape[0]), int(reader.per_input_scales.shape[0])
    assert out != positions
    with torch.no_grad():
        reader.layer.bias.zero_()
    peaks = _peaks_per_neuron(reader)
    NAPQ(BITS, "cpu", rate=1.0, per_neuron=True).transform(reader)
    scale = reader.parameter_scale.detach()
    assert scale.shape == (out,)
    torch.testing.assert_close(scale, torch.clamp(torch.floor(Q_MAX / peaks), min=1.0))
    codes = effective_parameters(reader, PerceptronTransformer())[0].detach() * scale.reshape(1, -1, 1)
    torch.testing.assert_close(codes, codes.round(), rtol=0, atol=1e-4)
    assert float(codes.abs().max()) <= Q_MAX + 1e-4
    assert_effective_parameters_on_chip_grid(reader, Q_MAX)


def _apart(perceptrons):
    """The token-mixing form's gauges: the encoding at one, the two producers at 1.0 and 0.5 (each position reads one
    producer, so the positions' folds differ), the reader at one."""
    return [torch.tensor(v) for v in (1.0, 1.0, 0.5, 1.0)]


def test_a_per_neuron_grid_takes_each_neuron_s_own_peak_where_positions_equal_outputs():
    """Two positions and two outputs, a peak per neuron (1 and 1/4 of the first's) apart from the peak per position
    (1 and 1/2): each neuron's scale is ``floor(q_max / its own peak)``; the positions' peaks would give another."""
    _flow, reader = _reader("token_mixing", gauges=_apart)
    with torch.no_grad():
        reader.layer.weight.copy_(torch.tensor([[0.8, -0.4], [0.05, -0.2]]))
        reader.layer.bias.zero_()
    peaks = _peaks_per_neuron(reader)
    weights = effective_parameters(reader, PerceptronTransformer())[0].detach()
    per_position = weights.abs().amax(dim=(1, 2))
    assert int(peaks.numel()) == int(per_position.numel()) == 2
    expected = torch.clamp(torch.floor(Q_MAX / peaks), min=1.0)
    assert not torch.equal(expected, torch.clamp(torch.floor(Q_MAX / per_position), min=1.0)), "the fixture's peaks"
    NAPQ(BITS, "cpu", rate=1.0, per_neuron=True).transform(reader)
    torch.testing.assert_close(reader.parameter_scale.detach(), expected)
    assert_effective_parameters_on_chip_grid(reader, Q_MAX)


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam_wide"))
def test_quantization_verification_reads_each_neuron_on_its_own_scale(form):
    """A program off one neuron's grid at one position only is refused; on every neuron's grid it passes."""
    _flow, reader = _reader(form)
    NAPQ(BITS, "cpu", rate=1.0, per_neuron=True).transform(reader)
    assert_effective_parameters_on_chip_grid(reader, Q_MAX)
    from mimarsinan.transformations.perceptron.position_fold import POSITION_WEIGHT, position_rows

    held, folds = getattr(reader, POSITION_WEIGHT), position_rows(reader)[0]
    slot = int(torch.nonzero(folds[-1] != 1.0)[0]) if bool((folds[-1] != 1.0).any()) else 0
    with torch.no_grad():  # one code of neuron 0 on the last weight row moved by 0.37 of a step
        held[-1, 0, slot] += 0.37 * float(reader.activation_scale.reshape(-1)[0]) / float(folds[-1, slot]) / float(
            reader.parameter_scale[0])
    with pytest.raises(AssertionError, match="weights off the chip grid"):
        assert_effective_parameters_on_chip_grid(reader, Q_MAX)


COUNT_CASES = [(form, seed) for form in ("token_mixing", "tree481", "conv_seam") for seed in (0, 1)]


@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
@pytest.mark.parametrize("form,seed", COUNT_CASES)
def test_the_count_family_under_per_neuron_grids_is_exact(form, seed, semantics, per_neuron_grids):
    """The NF and SCM parity is bit-exact (every core's window counts, atol 0) and the count walk equals the deployed
    read element by element, every core of the reader holding one threshold per neuron."""
    program = pf.count_program(form, seed, semantics, wq=True)
    reader = program.perceptrons[-1]
    assert reader.per_input_scales.dim() == 2 and reader.parameter_scale.dim() == 1
    index = next(n.perceptron_index for n in program.repr_._exec_order if getattr(n, "perceptron", None) is reader)
    thresholds = [n.threshold for n in program.ir.get_neural_cores() if n.perceptron_index == index]
    assert thresholds and all(torch.as_tensor(t).dim() == 1 for t in thresholds)
    mismatches, compared, _ = hg.compare(program, semantics, n=16, seed=seed)
    assert compared > 0 and all(v == 0 for v in mismatches.values()), mismatches
    pf.assert_equal_reads(pf.count_read(program, semantics, seed))


#: Seeds at which the older defect of ruling (h) does not strike the timing family under per-neuron grids (docstring).
TIMING_CASES = [("token_mixing", 0), ("token_mixing", 1), ("tree481", 0), ("tree481", 13), ("conv_seam", 12)]


@pytest.mark.parametrize("form,seed", TIMING_CASES)
def test_the_timing_family_under_per_neuron_grids_reads_as_the_deployed_program(form, seed, per_neuron_grids):
    read = pf.timing_read(form, seed, "one", wq=True)
    reader = read["perceptrons"][-1]
    assert reader.per_input_scales.dim() == 2 and reader.parameter_scale.dim() == 1
    pf.assert_equal_reads(read)


@pytest.mark.parametrize("shifted", (False, True))
@pytest.mark.parametrize("route", ("register", "rows"))
@pytest.mark.parametrize("form", ("token_mixing_3", "conv_seam"))
def test_the_two_scale_routes_take_each_neuron_s_own_grid_over_positions(form, route, shifted):
    """The two-scale projection (the on-chip bias register, or bias rows on their own grid): each neuron's weight scale
    from its own peak over every position, every position's bias on its neuron's grid; a recorded input shift (its
    compensation re-derived on the projected rows, each position's own) sizes the bias grid after the weights."""
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias

    _flow, reader = _reader(form)
    if shifted:
        apply_negative_shift_bias(reader, 0.3)
    peaks = _peaks_per_neuron(reader)
    register = route == "register"
    NAPQ(BITS, "cpu", rate=1.0, two_scale=True, register_bias=register, per_neuron=True).transform(reader)
    torch.testing.assert_close(reader.parameter_scale.detach(), torch.clamp(torch.floor(Q_MAX / peaks), min=1.0))
    assert_effective_parameters_on_chip_grid(reader, Q_MAX, register_bias=register)


@pytest.mark.parametrize("form", ("token_mixing_3", "conv_seam"))
def test_a_float_bias_grid_takes_each_neuron_s_own_peak_over_positions(form):
    _flow, reader = _reader(form)
    peaks = _peaks_per_neuron(reader)
    NAPQ(BITS, "cpu", rate=1.0, float_bias=True, per_neuron=True).transform(reader)
    torch.testing.assert_close(reader.parameter_scale.detach(), torch.clamp(torch.floor(Q_MAX / peaks), min=1.0))
    codes = effective_parameters(reader, PerceptronTransformer())[0].detach() * reader.parameter_scale.reshape(1, -1, 1)
    torch.testing.assert_close(codes, codes.round(), rtol=0, atol=1e-4)


def test_the_layer_grid_check_and_the_grid_report_read_each_neuron_on_its_own_scale():
    """Where positions equal outputs with the peaks apart (the own-peak fixture), each neuron's codes stay inside the
    register on its own scale; the report lists every neuron's scale and the codes' range."""
    from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
        assert_weights_within_layer_grid, quantization_grid_report)

    _flow, reader = _reader("token_mixing", gauges=_apart)
    with torch.no_grad():
        reader.layer.weight.copy_(torch.tensor([[0.8, -0.4], [0.05, -0.2]]))
        reader.layer.bias.zero_()
    NAPQ(BITS, "cpu", rate=1.0, per_neuron=True).transform(reader)
    assert_weights_within_layer_grid(reader, Q_MAX)
    (row,) = quantization_grid_report([reader], Q_MAX)
    scale = reader.parameter_scale.detach()
    codes = torch.round(effective_parameters(reader, PerceptronTransformer())[0].detach() * scale.reshape(1, -1, 1))
    assert row["parameter_scale"] == scale.tolist()
    assert (row["int_min"], row["int_max"]) == (int(codes.min()), int(codes.max()))
    assert -Q_MAX - 1 <= row["int_min"] and row["int_max"] <= Q_MAX
