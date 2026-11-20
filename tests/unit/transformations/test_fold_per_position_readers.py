"""[U-PFOLD] Every other reader of the stamp by its rule (the design review's J-2, correction 8).

Each reader takes a per-position fold through the one per-position reader:

- the bias writers from the effective weight (the negative-shift bake, the projection's compensation of a recorded
  shift, the affine fold into a consumer): the correction per position from that position's effective weight; a
  perceptron whose correction differs per position holds one bias per position (each position's core its own bias);
  where it is equal across positions, one bias and today's bytes;
- the per-output-channel bounds and scans (the saturation clip, the predicted zero fraction, the canonicalization's
  dominator scan, the event-serial split's grid): over the effective weight with the position axis reduced explicitly
  (a bound the minimum over positions, a scan the maximum), never flattened across it;
- the advisories: per output channel, reduced over positions;
- the readout law stamp: refuses by its own name, as for any fold other than one.

Each on the token-mixing form (positions equal to the outputs, and unequal) and on the convolution seam. Red on
U-TGAUGE4's tree by the fold's refusal at the stamp.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import unit.spiking.host_gather_forms as hg
import unit.spiking.position_fold_forms as pf

from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

FORMS = ("token_mixing", "token_mixing_3", "conv_seam", "conv_seam_wide")


def _reader(form: str, seed: int = 0):
    _model, flow, shape = pf.converted(form, seed)
    perceptrons = list(flow.get_perceptrons())
    for p, g in zip(perceptrons, pf.draw_gauges(flow, perceptrons, shape, seed, "one")):
        p.set_activation_scale(g)
    compute_per_source_scales(flow.get_mapper_repr())
    reader = perceptrons[-1]
    assert reader.per_input_scales.dim() == 2, "the reader's fold is held per position"
    return flow, reader


def _rows(reader):
    """Independent of the unit's reader: each position's effective weight ``fold_p W / theta`` (no normalization)."""
    stamp = reader.per_input_scales
    return stamp.unsqueeze(1) * reader.layer.weight.detach().unsqueeze(0) / reader.activation_scale.detach()


def _read_slots(flow, reader) -> torch.Tensor:
    """``(positions, 1, in)``: 1 where a position's slot reads an element of its input, 0 for a padding slot (a
    convolution's own patch gather over the seam's ``(C, 6, 3)`` input; every slot of a linear)."""
    node = next(n for n in flow.get_mapper_repr()._exec_order if getattr(n, "perceptron", None) is reader)
    if not hasattr(node, "patch_gather"):
        return torch.ones(reader.per_input_scales.shape).unsqueeze(1)
    table = node.patch_gather((int(node.in_channels), 6, 3)).slot_source_index
    return torch.as_tensor(table >= 0, dtype=torch.float32).unsqueeze(1)


def _correct_last_position(reader, channel: int, correction: float) -> None:
    """``B_p <- B_p - c`` on the last position's ``channel`` only (``position_bias.add_bias_corrections``)."""
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
    from mimarsinan.transformations.perceptron.position_bias import add_bias_corrections

    corrections = torch.zeros(int(reader.per_input_scales.shape[0]), int(reader.layer.weight.shape[0]))
    corrections[-1, channel] = correction
    add_bias_corrections(reader, corrections, lambda f: PerceptronTransformer().write_shared_bias(reader, f))


def _bias_rows(reader):
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
    from mimarsinan.transformations.perceptron.position_fold import effective_bias_rows

    return effective_bias_rows(reader, PerceptronTransformer().shared_effective_bias(reader))


@pytest.mark.parametrize("form", FORMS)
def test_the_shift_bake_corrects_each_position_by_its_own_effective_weight(form):
    """``B_p' = B - W~_p s`` per position (a scalar shift): the corrections differ per position, so the perceptron
    holds one bias per position; on U-TGAUGE4's tree the stamp refused."""
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias

    flow, reader = _reader(form)
    before = _bias_rows(reader).detach().clone()
    rows = _rows(reader)
    apply_negative_shift_bias(reader, 0.3)
    want = before - (rows * 0.3 * _read_slots(flow, reader)).sum(dim=-1)  # a padding slot reads no lifted element
    torch.testing.assert_close(_bias_rows(reader), want, rtol=1e-6, atol=1e-6)
    assert bool((want != want[:1]).any()), "the corrections differ per position"


@pytest.mark.parametrize("form", FORMS)
def test_the_bias_row_drive_asks_every_position_s_own_bias(form):
    """The row-drive stamp's question at Weight Quantization entry (does a nonzero effective bias ride the always-on
    row) read on each position's own bias: none on a zero bias, one once the bake gives positions their own."""
    from mimarsinan.mapping.support.bias_drive import _rides_a_row
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias

    _flow, reader = _reader(form)
    from mimarsinan.transformations.perceptron.position_fold import bias_offsets
    with torch.no_grad():
        reader.layer.bias.zero_()
    assert not _rides_a_row(reader), "a zero bias on every position rides no row"
    _correct_last_position(reader, 0, -1.0)
    assert _bias_rows(reader)[0].abs().max() == 0 and _rides_a_row(reader), "the last position's bias rides a row"
    apply_negative_shift_bias(reader, 0.3)
    assert bias_offsets(reader) is not None and _rides_a_row(reader)


def test_an_equal_correction_keeps_one_bias_and_today_s_bytes():
    """Where every position's correction is equal (a shift on no axon), the bias holds one row as before."""
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias

    _flow, reader = _reader("token_mixing")
    from mimarsinan.transformations.perceptron.position_fold import bias_offsets
    before = reader.layer.bias.detach().clone()
    apply_negative_shift_bias(reader, 0.0)
    assert bias_offsets(reader) is None and torch.equal(reader.layer.bias.detach(), before)


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam"))
def test_the_projection_compensates_each_position_s_projected_weights(form):
    """A recorded shift is compensated on the weights the projection installs, per position:
    ``B_p' = B_pre - Q(W~_p) s`` within half a weight step."""
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias
    from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
        NormalizationAwarePerceptronQuantization)

    flow, reader = _reader(form)
    from mimarsinan.transformations.perceptron.position_fold import effective_weight_rows
    unbaked = _bias_rows(reader).detach().clone()
    apply_negative_shift_bias(reader, 0.25)
    NormalizationAwarePerceptronQuantization(4, "cpu", rate=1.0, two_scale=False, float_bias=True).transform(reader)
    weights, index = effective_weight_rows(reader)
    want = unbaked - (weights[index] * 0.25 * _read_slots(flow, reader)).sum(dim=-1)
    torch.testing.assert_close(_bias_rows(reader), want, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("form", FORMS)
def test_the_saturation_clip_bound_is_the_minimum_over_positions(form):
    """A channel constant-OFF at every position keeps it after the clip: its bias rises to the minimum over positions
    of ``-max_x w~_p x``, never past any position's bound; the predicted zero fraction reads every position."""
    from mimarsinan.transformations.perceptron.bias_saturation import (
        clip_off_saturated_effective_bias, predicted_weight_zero_fraction)

    _flow, reader = _reader(form)
    with torch.no_grad():
        reader.layer.bias[0] = -50.0
    _correct_last_position(reader, 0, -5.0)  # the last position's bias above the others' (its own offset)
    rows = _rows(reader)
    bound = -rows.clamp(min=0).sum(dim=-1)  # (positions, out): -max over [0, 1] inputs
    assert clip_off_saturated_effective_bias(reader) >= 1
    biases = _bias_rows(reader)
    assert bool((biases[:, 0] <= bound[:, 0] + 1e-6).all()), "OFF at every position"
    torch.testing.assert_close((biases[:, 0] - bound[:, 0]).max(), torch.tensor(0.0), rtol=0, atol=1e-5)
    w = rows.abs()
    step = max(float(w.max()), float(biases.abs().max())) / 7
    assert predicted_weight_zero_fraction(reader, 4) == pytest.approx(float((w < step / 2).float().mean()))


@pytest.mark.parametrize("form", FORMS)
def test_the_advisories_read_each_output_channel_over_every_position(form):
    """The scale-spread advisory's channel norms are per output channel (the largest over positions), never one per
    position; the bias-dominance advisory reads the largest weight over every position."""
    from mimarsinan.advisories.rules_graph_scale import _weight_channel_norms, rule_bias_grid_dominance

    flow, reader = _reader(form)
    norms = _weight_channel_norms(reader)
    rows = _rows(reader)
    assert len(norms) == int(reader.layer.weight.shape[0])
    torch.testing.assert_close(torch.tensor(norms, dtype=rows.dtype), rows.norm(dim=-1).amax(dim=0), rtol=1e-5, atol=0)
    with torch.no_grad():
        reader.layer.bias.fill_(float(rows.abs().max()) * 50.0)
    plan = SimpleNamespace(weight_quantization=True, config={"weight_bits": 4}, spiking_mode="lif",
                           is_synchronized_ttfs=False)
    advisories = rule_bias_grid_dominance(flow.get_mapper_repr(), plan, None)
    assert advisories and str(reader.name) in advisories[0].title


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam"))
def test_the_event_serial_split_reads_the_grid_over_every_position(form):
    """The split's grid (``max|w~|`` and ``max|b~|`` over every position) is read, not refused, under a per-event
    soma with a register: a split of one row passes; a bias held at one position past the weights' reach needs more
    rows and is refused by the split's own name."""
    from mimarsinan.chip_simulation.soma_law import SomaLaw
    from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import (
        BiasRowSplitEventSerialError, refuse_split_under_event_serial_soma)

    law = SomaLaw(firing_mode="Novena", thresholding_mode="<=", firing_granularity="per_event",
                  membrane_arithmetic="saturating_signed", membrane_bits=8)
    flow, reader = _reader(form)
    with torch.no_grad():
        for p in flow.get_perceptrons():  # every bias within the weights' reach: one row each
            p.layer.bias.zero_()
    refuse_split_under_event_serial_soma(flow.get_mapper_repr(), 4, law)
    _correct_last_position(reader, 0, -10.0 * float(_rows(reader).abs().max()))
    with pytest.raises(BiasRowSplitEventSerialError, match="needs [0-9]+ always-on rows"):
        refuse_split_under_event_serial_soma(flow.get_mapper_repr(), 4, law)


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam"))
def test_the_affine_fold_into_a_consumer_is_per_position(form):
    """``W~_p[:, j] *= a_j`` and ``B_p += W~_p c`` per position."""
    from mimarsinan.tuning.lif_affine_fold import fold_affine_into_consumer

    flow, reader = _reader(form)
    from mimarsinan.transformations.perceptron.position_fold import effective_weight_rows
    fan_in = int(reader.layer.weight.shape[1])
    a = torch.linspace(0.8, 1.2, fan_in)
    c = torch.linspace(-0.1, 0.2, fan_in)
    rows = _rows(reader).detach().clone()
    before = _bias_rows(reader).detach().clone()
    fold_affine_into_consumer(reader, a, c)
    weights, index = effective_weight_rows(reader)
    torch.testing.assert_close(weights[index], rows * a, rtol=1e-5, atol=1e-6)
    want = before + (rows * c * _read_slots(flow, reader)).sum(dim=-1)  # a padding slot reads no producer's value
    torch.testing.assert_close(_bias_rows(reader), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam"))
def test_the_readout_law_stamp_refuses_by_its_own_name(form):
    """A per-position fold is a fold other than one: the readout law stamp refuses it by its own name, as before."""
    from mimarsinan.mapping.export.law_stamp import ReadoutLawPerceptronFormError, perceptron_layer_weight

    _flow, reader = _reader(form)
    with pytest.raises(ReadoutLawPerceptronFormError):
        perceptron_layer_weight(reader, 0)


@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
def test_the_pipeline_s_shift_bake_reads_as_the_deployed_program(semantics):
    """The pipeline's negative-shift bake on the token-mixing form (a signed host LayerNorm joined with a core over it,
    read by a token-mixing Linear): each position's core holds its own bias, and the count walk equals the deployed
    read."""
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
    from mimarsinan.mapping.support.negative_shift import apply_negative_value_shifts, transfer_negative_shifts_to_ir
    from mimarsinan.mapping.support.bias_compensation import propagate_negative_shifts_to_hybrid

    def shift(flow, _perceptrons):
        (ln,) = [n for n in flow.get_mapper_repr().execution_order()
                 if isinstance(n, ComputeOpMapper) and isinstance(getattr(n, "module", None), torch.nn.LayerNorm)]
        compute_per_source_scales(flow.get_mapper_repr())  # the stamp the bake reads (the pipeline's order)
        shifted = apply_negative_value_shifts(flow, {ln: torch.tensor([-0.6])})  # a token seam's scalar minimum
        assert list(shifted) == [ln]

    program = _signed_program(semantics, shift)
    transfer_negative_shifts_to_ir(program.flow, program.ir)
    propagate_negative_shifts_to_hybrid(program.ir, program.hybrid)
    reader = program.perceptrons[-1]
    assert reader.per_input_scales.dim() == 2 and getattr(reader, "position_bias", None) is not None
    pf.assert_equal_reads(pf.count_read(program, semantics, 0))


class _Signed(torch.nn.Module):
    shape = (2, 3)

    def __init__(self, gen):
        super().__init__()
        self.e, self.ln, self.c, self.r = pf._lin(3, 2), torch.nn.LayerNorm(2), pf._lin(2, 2), pf._lin(2, 2)
        pf._init(gen, self.modules())

    def forward(self, x):
        s = self.ln(self.e(x))
        return self.r(torch.cat([s, self.c(s)], dim=-1).transpose(1, 2)).flatten(1)


_SIGNED = _Signed


def _signed_program(semantics, prepare):
    pf.FORMS["signed_token_mixing"] = _Signed
    try:
        return pf.count_program("signed_token_mixing", 0, semantics, prepare=prepare)
    finally:
        del pf.FORMS["signed_token_mixing"]


@pytest.mark.parametrize("form", ("token_mixing_3", "conv_seam"))
def test_the_starved_grid_scan_removes_a_channel_on_every_position(form):
    """The canonicalization's dominator scan reads the largest weight over every position and removes an output
    channel on every position alike (an index on the first axis of a per-position weight is a position, not a
    channel), its bias with it; a decision-invariant removal (a constant-OFF channel) is kept."""
    from mimarsinan.transformations.perceptron.bias_canonicalization import _decisions, _remove_nuisance_grid_dominators
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

    flow, reader = _reader(form)
    from mimarsinan.transformations.perceptron.position_fold import effective_weight_rows
    with torch.no_grad():
        reader.layer.bias[1] = -float(_rows(reader).abs().max()) * 400.0
    torch.manual_seed(0)
    batches = [torch.rand(8, *pf.shape_of(form))]
    report = {"removed": 0, "removal_restored": 0}
    _remove_nuisance_grid_dominators(reader, flow, batches, "cpu", 4, _decisions(flow, batches, "cpu"),
                                     PerceptronTransformer(), report)
    weights, _ = effective_weight_rows(reader)
    assert report["removed"] == 1
    assert bool((weights[:, 1] == 0).all()) and bool((weights[:, 0] != 0).any())
    assert bool((_bias_rows(reader)[:, 1] == 0).all())


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam"))
def test_a_timing_core_reads_its_position_s_own_bias(form):
    """The timing family over a bias held per position (a shift baked into the reader): the walk's timing activation
    rests at each position's own bias, as each position's core does, and the walk equals the deployed read."""
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias

    def bake(_flow, perceptrons):
        assert perceptrons[-1].per_input_scales.dim() == 2, "the reader's fold is held per position"
        apply_negative_shift_bias(perceptrons[-1], 0.05)

    read = pf.timing_read(form, 0, prepare=bake)
    reader = read["perceptrons"][-1]
    assert getattr(reader, "position_bias", None) is not None
    pf.assert_equal_reads(read)
    index = [getattr(n, "perceptron_index", None) for n in read["flow"].get_mapper_repr()._exec_order
             if getattr(n, "perceptron", None) is reader][0]
    fan_in, held = int(reader.layer.weight.shape[1]), _bias_rows(reader).double()
    cores = [c for c in read["ir"].get_neural_cores() if getattr(c, "perceptron_index", None) == index
             and not getattr(c, "_depth_relay", False)]
    assert len(cores) == int(held.shape[0])
    for core in cores:  # the always-on rows past the data axons sum to the position's own bias
        rows = torch.as_tensor(np.asarray(core.get_core_matrix(read["ir"]), dtype=np.float64)[fan_in:])
        torch.testing.assert_close(rows.sum(dim=0), held[core.perceptron_output_column], rtol=1e-5, atol=1e-5)


def test_a_refused_change_restores_each_position_s_state():
    """The canonicalization's decision check restores a refused change whole: the shared bias and every position's
    offset as they were."""
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias
    from mimarsinan.transformations.perceptron.bias_canonicalization import _apply_decision_verified

    flow, reader = _reader("token_mixing")
    apply_negative_shift_bias(reader, 0.3)
    from mimarsinan.transformations.perceptron.position_fold import bias_offsets
    offsets, bias = bias_offsets(reader).detach().clone(), reader.layer.bias.detach().clone()
    torch.manual_seed(0)
    batches = [torch.rand(8, *pf.shape_of("token_mixing"))]
    never = [torch.full((8,), -1)]  # decisions no model gives: the change is refused

    def change():
        _correct_last_position(reader, 1, 0.75)

    assert not _apply_decision_verified(reader, flow, batches, "cpu", never, change)
    assert torch.equal(bias_offsets(reader), offsets) and torch.equal(reader.layer.bias.detach(), bias)


def test_the_dominator_scan_reads_a_bias_held_at_a_later_position():
    """A bias that dominates the grid at one position only (its own offset) is found by the scan, which reads every
    position; the channel, constant-OFF at every position (no positive weight), is removed on every position."""
    from mimarsinan.transformations.perceptron.bias_canonicalization import _decisions, _remove_nuisance_grid_dominators
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

    flow, reader = _reader("token_mixing_3")
    from mimarsinan.transformations.perceptron.position_fold import effective_weight_rows
    with torch.no_grad():
        reader.layer.weight[1] = -reader.layer.weight[1].abs()
        reader.layer.bias[1] = -0.5 * float(_rows(reader).abs().max()) * float(reader.activation_scale.max())
    _correct_last_position(reader, 1, float(_rows(reader).abs().max()) * 400.0)
    torch.manual_seed(0)
    batches = [torch.rand(8, *pf.shape_of("token_mixing_3"))]
    report = {"removed": 0, "removal_restored": 0}
    _remove_nuisance_grid_dominators(reader, flow, batches, "cpu", 4, _decisions(flow, batches, "cpu"),
                                     PerceptronTransformer(), report)
    assert report["removed"] == 1
    assert bool((effective_weight_rows(reader)[0][:, 1] == 0).all()) and bool((_bias_rows(reader)[:, 1] == 0).all())
