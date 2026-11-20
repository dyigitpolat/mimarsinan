"""[U-PFOLD r2] The output axis declared by the one reader (the fold unit's M-1, ruling (e) of 2026-11-12 20:51).

The one reader of a perceptron's fold (``position_fold`` in its two forms, through ``PerceptronTransformer``) declares,
for every view it hands out, which axis holds the output neuron: a weight view and a bias view, in the per-input form
(``(out, in, ...)`` and ``(out,)``) and in the per-position form (``(rows, out, in)`` and ``(positions, out)``):
``OutputAxes``, ``PER_INPUT``, ``PER_POSITION`` and ``output_axes``. A bias transform of the reader in both forms
receives the bias view it declares (every position's, where a fold is held per position). Every reader of the unit
that works per output neuron takes the axis from the declaration, never from a literal index (``along_outputs``,
``output_rows``, ``outputs_first``, ``position_bias.weight_charges``).

Each site is read against an independent reference on forms whose positions equal their outputs (the token-mixing
form, two and two; the wide convolution seam, eighteen and eighteen), with each neuron's and each position's values
drawn apart, so a read on the wrong axis gives another number, and on forms whose positions are unequal to their
outputs (three outputs over two positions), where it gives another shape. Also here: the event-serial twin over a
linear whose fold is held per position (m-6: refused by name, never a silent difference), and a cache written before
the channel-axis declaration (m-7: its load declares the axis on its spiking activations in either restore order).

Red on the first revision's tree: the declaration is absent; a bias transform of the reader in both forms received
the shared bias where a fold per position holds no offsets; the event-serial twin refused a fold per position by a
rank message; a perceptron restored after its owner kept its activations at the last axis.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import unit.spiking.host_gather_forms as hg
import unit.spiking.position_fold_forms as pf

import mimarsinan.transformations.perceptron.position_fold as PF
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

#: (form, positions equal to the outputs): the reader's outputs and positions.
EQUAL = ("token_mixing", "conv_seam_wide")
UNEQUAL = ("token_mixing_3", "tree481", "conv_seam")
FORMS = EQUAL + UNEQUAL


def _flow(form: str, seed: int = 0, gauges: str = "drawn"):
    """The converted form with drawn gauges (one per output channel where a perceptron has several), its fold stamped."""
    _model, flow, shape = pf.converted(form, seed)
    perceptrons = list(flow.get_perceptrons())
    for p, g in zip(perceptrons, pf.draw_gauges(flow, perceptrons, shape, seed, gauges)):
        p.set_activation_scale(g)
    compute_per_source_scales(flow.get_mapper_repr())
    return flow, perceptrons


def _reader(form: str, seed: int = 0, gauges: str = "drawn"):
    flow, perceptrons = _flow(form, seed, gauges)
    reader = perceptrons[-1]
    assert reader.per_input_scales.dim() == 2, "the reader's fold is held per position"
    return flow, reader


def _act(p) -> torch.Tensor:
    act = torch.as_tensor(p.activation_scale).detach().reshape(-1)
    return act.expand(int(p.layer.weight.shape[0])) if act.numel() == 1 else act


def _reference_weights(p) -> torch.Tensor:
    """``(positions, out, in)`` by explicit loops, independent of the unit's reader: position ``p`` and neuron ``c``
    read ``fold_p * W_c / act_c`` (the forms carry no normalization); the per-input form is one position."""
    stamp = p.per_input_scales.detach()
    stamp = stamp.reshape(1, -1) if stamp.dim() == 1 else stamp
    weight, act = p.layer.weight.detach(), _act(p)
    out = torch.empty(int(stamp.shape[0]), int(weight.shape[0]), int(weight.shape[1]))
    for position in range(int(stamp.shape[0])):
        for neuron in range(int(weight.shape[0])):
            out[position, neuron] = stamp[position] * weight[neuron] / act[neuron]
    return out


def _distinct_per_neuron(n: int) -> torch.Tensor:
    return torch.linspace(0.5, 2.0, n) + 0.01 * torch.arange(n) ** 2


def _offsets_on_last_position(reader, channel: int = 0, correction: float = -0.75) -> None:
    from mimarsinan.transformations.perceptron.position_bias import add_bias_corrections

    corrections = torch.zeros(int(reader.per_input_scales.shape[0]), int(reader.layer.weight.shape[0]))
    corrections[-1, channel] = correction
    add_bias_corrections(reader, corrections, lambda f: PerceptronTransformer().write_shared_bias(reader, f))


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_the_two_forms_are_declared_with_their_row_axis():
    assert PF.PER_INPUT == PF.OutputAxes(weight=0, bias=0)
    assert PF.PER_POSITION == PF.OutputAxes(weight=1, bias=1)
    assert PF.ROW_AXIS == 0


@pytest.mark.parametrize("form", FORMS)
def test_the_reader_declares_the_output_axis_of_its_weight_view_in_both_forms(form):
    """Each neuron's slice on the declared axis is that neuron's effective weight at every position: on the reader (a
    fold per position, the per-position form) and on each producer (one fold per input, the per-input form)."""
    _flow_, perceptrons = _flow(form)
    assert PF.holds_positions(perceptrons[-1])
    for p in perceptrons:
        if getattr(p, "per_input_scales", None) is None:
            continue
        both = PerceptronTransformer(positions=True)
        rows = PF.holds_positions(p)
        axes = PF.output_axes(p, both.positions)
        assert axes == (PF.PER_POSITION if rows else PF.PER_INPUT)
        assert PF.output_axes(p, False).weight == PF.PER_INPUT.weight
        view = PF.effective_parameters(p, both)[0] if rows else both.get_effective_weight(p)
        reference = _reference_weights(p)
        got = PF.outputs_first(view, axes.weight)  # (out, positions, in), or (out, in)
        want = reference.movedim(1, 0) if rows else reference[0]
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("form", FORMS)
def test_the_reader_declares_the_output_axis_of_its_bias_view_in_both_forms(form):
    """Each neuron's slice on the declared axis is that neuron's bias at every position, its offset where held."""
    _flow_, reader = _reader(form)
    _offsets_on_last_position(reader, channel=1, correction=-0.75)
    both = PerceptronTransformer(positions=True)
    axes = PF.output_axes(reader, both.positions)
    view = both.get_effective_bias(reader)
    shared = PerceptronTransformer().shared_effective_bias(reader).detach()
    positions, out = int(reader.per_input_scales.shape[0]), int(reader.layer.weight.shape[0])
    want = torch.empty(out, positions)
    for neuron in range(out):
        for position in range(positions):
            want[neuron, position] = shared[neuron] + (0.75 if (neuron == 1 and position == positions - 1) else 0.0)
    torch.testing.assert_close(PF.outputs_first(view, axes.bias), want, rtol=1e-6, atol=1e-6)
    # The per-input reader hands out no bias view of it, but a bias transform still receives every position's.
    assert PF.output_axes(reader, False) == PF.OutputAxes(weight=0, bias=1)


@pytest.mark.parametrize("form", FORMS)
def test_a_bias_transform_of_the_reader_in_both_forms_receives_the_declared_view(form):
    """Without any offset, the reader in both forms hands every position's bias to a bias transform (the view it
    declares), so a transform per output neuron on the declared axis lands on each neuron at every position."""
    _flow_, reader = _reader(form)
    both = PerceptronTransformer(positions=True)
    assert PF.bias_offsets(reader) is None
    seen: list = []
    out = int(reader.layer.weight.shape[0])
    factor = _distinct_per_neuron(out)
    before = PerceptronTransformer().shared_effective_bias(reader).detach().clone()

    def per_neuron(t):
        seen.append(tuple(t.shape))
        return t * PF.along_outputs(factor.to(t.dtype), PF.output_axes(reader, True).bias, t.dim())

    both.apply_effective_bias_transform(reader, per_neuron)
    assert seen == [(int(reader.per_input_scales.shape[0]), out)]
    assert PF.bias_offsets(reader) is None, "one transform alike at every position holds no offset"
    torch.testing.assert_close(PerceptronTransformer().shared_effective_bias(reader).detach(), before * factor,
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("form", FORMS)
def test_a_weight_transform_per_output_neuron_lands_on_each_neuron_at_every_position(form):
    _flow_, reader = _reader(form)
    both = PerceptronTransformer(positions=True)
    axes = PF.output_axes(reader, True)
    out = int(reader.layer.weight.shape[0])
    factor = _distinct_per_neuron(out)
    before = PF.effective_parameters(reader, both)[0].detach().clone()
    both.apply_effective_weight_transform(
        reader, lambda t: t * PF.along_outputs(factor.to(t.dtype), axes.weight, t.dim()))
    after = PF.effective_parameters(reader, both)[0]
    for neuron in range(out):
        torch.testing.assert_close(after[:, neuron], before[:, neuron] * factor[neuron], rtol=1e-5, atol=1e-6)


def test_the_view_helpers_follow_the_axis_they_are_given():
    view = torch.arange(24.0).reshape(2, 3, 4)
    assert PF.along_outputs(torch.tensor([1.0, 2.0, 3.0]), 1, 3).shape == (1, 3, 1)
    assert PF.along_outputs(torch.tensor(2.0), 1, 3).shape == ()
    torch.testing.assert_close(PF.output_rows(view, 1), view.permute(1, 0, 2).reshape(3, 8))
    torch.testing.assert_close(PF.outputs_first(view, 1), view.permute(1, 0, 2))


# ---------------------------------------------------------------------------
# The unit's readers per output neuron, each on its declared axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", EQUAL)
def test_normalization_fusion_scales_each_projected_row_per_output_neuron(form):
    from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
        NormalizationAwarePerceptronQuantization as NAPQ)

    _flow_, reader = _reader(form)
    NAPQ(4, "cpu").transform(reader)
    held = getattr(reader, PF.POSITION_WEIGHT).detach().clone()
    out = int(reader.layer.weight.shape[0])
    factor = _distinct_per_neuron(out)
    PF.fuse_output_scale(reader, factor)
    after = getattr(reader, PF.POSITION_WEIGHT)
    for neuron in range(out):
        torch.testing.assert_close(after[:, neuron], held[:, neuron] * factor[neuron], rtol=1e-6, atol=0)


@pytest.mark.parametrize("form", EQUAL)
def test_a_pruned_entry_stays_zero_on_every_row_and_only_there(form):
    from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
        NormalizationAwarePerceptronQuantization as NAPQ)

    _flow_, reader = _reader(form)
    mask = torch.zeros_like(reader.layer.weight, dtype=torch.bool)
    mask[0, 1] = True  # one entry of neuron 0: no row and no other neuron holds it
    with torch.no_grad():
        reader.layer.weight[0, 1] = 0.0
    reader.layer.register_buffer("prune_mask", mask)
    NAPQ(4, "cpu", rate=0.5).transform(reader)
    held = getattr(reader, PF.POSITION_WEIGHT)
    assert bool((held[:, 0, 1] == 0).all())
    others = held.clone()
    others[:, 0, 1] = 1.0
    assert bool((others[:, :, 1] != 0).any(dim=0).all()), "the other neurons keep that input on some row"


@pytest.mark.parametrize("form", EQUAL + ("token_mixing_3",))
def test_an_output_channel_is_zeroed_on_its_own_neuron_at_every_position(form):
    from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
        NormalizationAwarePerceptronQuantization as NAPQ)
    from mimarsinan.transformations.perceptron.position_bias import zero_output_channel

    _flow_, reader = _reader(form)
    NAPQ(4, "cpu", rate=0.5).transform(reader)
    raw, held = reader.layer.weight.detach().clone(), getattr(reader, PF.POSITION_WEIGHT).detach().clone()
    zero_output_channel(reader, 1)
    assert bool((reader.layer.weight[1] == 0).all()) and bool((getattr(reader, PF.POSITION_WEIGHT)[:, 1] == 0).all())
    keep = [c for c in range(int(raw.shape[0])) if c != 1]
    torch.testing.assert_close(reader.layer.weight[keep], raw[keep], rtol=0, atol=0)
    torch.testing.assert_close(getattr(reader, PF.POSITION_WEIGHT)[:, keep], held[:, keep], rtol=0, atol=0)


@pytest.mark.parametrize("form", FORMS)
def test_one_correction_per_output_neuron_lands_on_that_neuron_at_every_position(form):
    from mimarsinan.transformations.perceptron.position_bias import add_bias_corrections

    _flow_, reader = _reader(form)
    out = int(reader.layer.weight.shape[0])
    corrections = _distinct_per_neuron(out)
    before = PerceptronTransformer().shared_effective_bias(reader).detach().clone()
    add_bias_corrections(reader, corrections, lambda f: PerceptronTransformer().write_shared_bias(reader, f))
    assert PF.bias_offsets(reader) is None
    torch.testing.assert_close(PerceptronTransformer().shared_effective_bias(reader).detach(), before - corrections,
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("form", FORMS)
def test_each_position_s_charge_reads_its_own_weights_per_output_neuron(form):
    from mimarsinan.transformations.perceptron.position_bias import weight_charges

    _flow_, reader = _reader(form)
    weights = PF.effective_parameters(reader, PerceptronTransformer())[0]
    shift = torch.linspace(0.1, 0.9, int(reader.per_input_scales.shape[1]))
    slots = PF.shift_rows(reader, shift)
    reference = _reference_weights(reader)
    want = torch.empty(int(reference.shape[0]), int(reference.shape[1]))
    for position in range(int(reference.shape[0])):
        for neuron in range(int(reference.shape[1])):
            want[position, neuron] = (reference[position, neuron] * slots[position, 0]).sum()
    torch.testing.assert_close(PF.outputs_first(weight_charges(weights, slots), PF.PER_POSITION.bias),
                               want.t(), rtol=1e-5, atol=1e-6)


def _project_with_offsets(reader) -> None:
    from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
        NormalizationAwarePerceptronQuantization as NAPQ)

    NAPQ(4, "cpu", rate=0.5).transform(reader)
    _offsets_on_last_position(reader, channel=1, correction=-0.5)


@pytest.mark.parametrize("form", ("token_mixing", "token_mixing_3", "tree481"))
def test_a_linear_over_positions_computes_each_position_at_its_row_and_offset(form):
    """The network function: position ``p``, neuron ``c`` is ``W_p,c x_p + b_c + offset_p,c act_c``."""
    from mimarsinan.transformations.perceptron.position_forward import positional_preactivation

    _flow_, reader = _reader(form)
    _project_with_offsets(reader)
    held, act = getattr(reader, PF.POSITION_WEIGHT).detach(), _act(reader)
    offsets, bias = PF.bias_offsets(reader).detach(), reader.layer.bias.detach()
    positions, out, fan_in = (int(d) for d in held.shape)
    x = torch.rand(3, positions, fan_in, generator=torch.Generator().manual_seed(5))
    want = torch.empty(3, positions, out)
    for position in range(positions):
        for neuron in range(out):
            want[:, position, neuron] = (x[:, position] * held[position, neuron]).sum(-1) + bias[neuron] + \
                offsets[position, neuron] * act[neuron]
    with torch.no_grad():
        torch.testing.assert_close(positional_preactivation(reader, x), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("form", ("conv_seam_wide", "conv_seam"))
def test_a_convolution_computes_each_output_position_at_its_row_and_offset(form):
    from mimarsinan.transformations.perceptron.position_forward import add_position_offsets, conv_positions

    flow, reader = _reader(form)
    _project_with_offsets(reader)
    held, act = getattr(reader, PF.POSITION_WEIGHT).detach(), _act(reader)
    _rows, row_of = PF.position_rows(reader)
    offsets, bias = PF.bias_offsets(reader).detach(), reader.layer.bias.detach()
    node = next(n for n in flow.get_mapper_repr()._exec_order if getattr(n, "perceptron", None) is reader)
    x = torch.rand(2, int(node.in_channels), 6, 3, generator=torch.Generator().manual_seed(7))
    patches = torch.nn.functional.unfold(x, (3, 3), padding=(1, 1))  # (B, in, positions)
    out = int(held.shape[1])
    want = torch.empty(2, out, int(patches.shape[2]))
    for position in range(int(patches.shape[2])):
        for neuron in range(out):
            want[:, neuron, position] = (patches[:, :, position] * held[int(row_of[position]), neuron]).sum(-1) + \
                bias[neuron] + offsets[position, neuron] * act[neuron]
    with torch.no_grad():
        got = add_position_offsets(reader, conv_positions(reader, x, (3, 3), (1, 1), (1, 1), (1, 1)))
    torch.testing.assert_close(got.reshape(2, out, -1), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam_wide"))
def test_the_timing_activation_rests_each_position_at_its_own_bias_per_output_neuron(form):
    """``position_bias_totals`` lays every position's bias out as the perceptron's output sample, so the timing
    activation's view of it holds neuron ``c`` at position ``p`` where the output does."""
    from mimarsinan.models.nn.activations.ttfs_spiking import _channel_broadcast_view
    from mimarsinan.transformations.perceptron.position_forward import position_bias_totals

    flow, reader = _reader(form)
    _project_with_offsets(reader)
    act, offsets = _act(reader), PF.bias_offsets(reader).detach()
    bias = torch.linspace(-0.3, 0.3, int(reader.layer.weight.shape[0]))
    totals = position_bias_totals(reader, bias)
    positions, out = (int(d) for d in offsets.shape)
    channel_first = PF.reads_channels_first(reader)
    sample = (out, 6, 3) if channel_first else (positions, out)
    view = _channel_broadcast_view(totals, torch.zeros(1, *sample), int(reader.output_channel_axis))
    flat = view.reshape(1, out, positions) if channel_first else view.reshape(1, positions, out).transpose(1, 2)
    for neuron in range(out):
        for position in range(positions):
            want = bias[neuron] + offsets[position, neuron] * act[neuron]
            assert float(flat[0, neuron, position]) == pytest.approx(float(want), rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("form", ("token_mixing", "token_mixing_3", "tree481"))
def test_each_position_s_cores_hold_its_own_weights_neuron_by_neuron(form):
    """A linear over positions maps each position to cores whose columns are the output neurons, each holding that
    position's effective weight of that neuron."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping

    flow, reader = _reader(form)
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=256, max_neurons=256).map(repr_)
    reference = _reference_weights(reader)
    index = next(n.perceptron_index for n in repr_._exec_order if getattr(n, "perceptron", None) is reader)
    checked = 0
    for core in ir.get_neural_cores():
        if core.perceptron_index != index:
            continue
        position = int(core.perceptron_output_column)
        matrix = torch.as_tensor(core.get_core_matrix(ir), dtype=torch.float32)  # (axons, neurons)
        for neuron in range(int(reference.shape[1])):
            torch.testing.assert_close(matrix[: reference.shape[2], neuron], reference[position, neuron],
                                       rtol=1e-5, atol=1e-6)
        checked += 1
    assert checked == int(reader.per_input_scales.shape[0])


@pytest.mark.parametrize("form", ("conv_seam_wide", "conv_seam"))
def test_each_patch_fold_s_bank_holds_its_row_neuron_by_neuron(form):
    from mimarsinan.mapping.ir_mapping_class import IRMapping

    flow, reader = _reader(form)
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=256, max_neurons=256).map(repr_)
    weights, row_of = PF.effective_weight_rows(reader)
    index = next(n.perceptron_index for n in repr_._exec_order if getattr(n, "perceptron", None) is reader)
    checked = 0
    for core in (c for c in ir.get_neural_cores() if c.perceptron_index == index):
        row = weights[int(row_of[int(core.perceptron_output_column)])].detach()
        matrix = torch.as_tensor(core.get_core_matrix(ir), dtype=torch.float32)
        lo, hi = core.perceptron_output_slice
        for neuron in range(int(lo), int(hi)):
            torch.testing.assert_close(matrix[: row.shape[1], neuron - int(lo)], row[neuron], rtol=1e-5, atol=1e-6)
        checked += 1
    assert checked == int(reader.per_input_scales.shape[0])


@pytest.mark.parametrize("form", FORMS)
def test_the_advisory_norm_per_output_neuron_is_its_largest_over_positions(form):
    from mimarsinan.advisories.rules_graph_scale import _weight_channel_norms

    _flow_, reader = _reader(form)
    reference = _reference_weights(reader)
    want = [max(float(reference[p, c].norm()) for p in range(int(reference.shape[0])))
            for c in range(int(reference.shape[1]))]
    assert _weight_channel_norms(reader) == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize("form", FORMS)
def test_the_off_bound_per_output_neuron_is_its_least_over_positions(form):
    from mimarsinan.transformations.perceptron.bias_saturation import UNIT_INPUT_RANGE, _off_bound

    _flow_, reader = _reader(form)
    _offsets_on_last_position(reader, channel=0, correction=0.4)
    reference, offsets = _reference_weights(reader), PF.bias_offsets(reader).detach()
    want = []
    for neuron in range(int(reference.shape[1])):
        want.append(min(-float(reference[p, neuron].clamp(min=0).sum()) - float(offsets[p, neuron])
                        for p in range(int(reference.shape[0]))))
    got = _off_bound(reader, PerceptronTransformer(), UNIT_INPUT_RANGE)
    assert got.tolist() == pytest.approx(want, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("form", FORMS)
def test_the_twins_read_each_core_s_weights_and_bias_neuron_by_neuron(form):
    from mimarsinan.spiking.segment_policy_lif_fold import twin_parameters

    _flow_, reader = _reader(form)
    _offsets_on_last_position(reader, channel=1, correction=-0.25)
    weight, bias = twin_parameters(reader)
    reference = _reference_weights(reader)
    shared = PerceptronTransformer().shared_effective_bias(reader).detach()
    offsets = PF.bias_offsets(reader).detach()
    for neuron in range(int(reference.shape[1])):
        torch.testing.assert_close(weight[:, neuron], reference[:, neuron], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(bias[:, neuron], shared[neuron] + offsets[:, neuron], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("form", ("token_mixing_3", "conv_seam"))
def test_the_starved_grid_scan_reads_each_neuron_s_largest_bias_over_positions(form, monkeypatch):
    """The canonicalization's scan per output neuron: the neuron whose bias at some position dominates the grid is
    the one removed, at every position."""
    import mimarsinan.transformations.perceptron.bias_canonicalization as canon

    flow, reader = _reader(form)
    out = int(reader.layer.weight.shape[0])
    big = float(_reference_weights(reader).abs().max()) * 40.0
    _offsets_on_last_position(reader, channel=out - 1, correction=-big)  # its bias dominates at the last position only
    removed: list = []

    def accept(perceptron, model, batches, device, reference, apply):
        apply()
        removed.append([c for c in range(out) if bool((perceptron.layer.weight[c] == 0).all())])
        return True

    monkeypatch.setattr(canon, "_apply_decision_verified", accept)
    canon._remove_nuisance_grid_dominators(reader, flow, [], "cpu", 4, None, PerceptronTransformer(),
                                           {"removed": 0, "removal_restored": 0})
    assert removed and removed[0] == [out - 1]
    assert float(PF.effective_bias_rows(reader, PerceptronTransformer().shared_effective_bias(reader))[:, out - 1]
                 .abs().max()) == 0.0


# ---------------------------------------------------------------------------
# m-6: the event-serial twin over a linear whose fold is held per position
# ---------------------------------------------------------------------------


class _FlatTokenMixing(nn.Module):
    """A flat encoding read by two producers whose outputs are joined and read per producer: the reader over the
    joined rows is the only hop over positions, so the event-serial twin reaches it."""

    shape = (4,)

    def __init__(self, gen, width=2):
        super().__init__()
        self.e, self.a, self.b, self.r = pf._lin(4, 4), pf._lin(4, 2), pf._lin(4, 2), pf._lin(2, width)
        pf._init(gen, self.modules())

    def forward(self, x):
        e = self.e(x)
        return self.r(torch.cat([self.a(e), self.b(e)], dim=-1).reshape(-1, 2, 2)).flatten(1)


@pytest.mark.parametrize("width", (2, 3))
@pytest.mark.parametrize("gauges", ("one", "same"))
def test_the_event_serial_twin_over_a_linear_over_positions_is_refused_by_name(width, gauges, monkeypatch):
    """Under a per-event soma law the event-serial twin over the reader refuses by name: a fold per position (one gauge
    per producer) by its own rule, since the mapper's unfold holds one core for every position; one fold per input
    (every producer at one gauge) by the twin's slot order, as before this unit. Never a silent difference."""
    from mimarsinan.chip_simulation.soma_law import SomaLaw
    from mimarsinan.models.spiking.serial import SerialFoldUnsupportedError

    monkeypatch.setitem(pf.FORMS, "flat_token_mixing", lambda gen: _FlatTokenMixing(gen, width=width))
    if gauges == "same":
        monkeypatch.setattr(pf, "draw_gauges", lambda flow, ps, shape, seed, mode, scale=1.0: [torch.tensor(1.0)
                                                                                             for _ in ps])
    program = pf.count_program("flat_token_mixing", 0, "streamed_lif")
    assert program.perceptrons[-1].per_input_scales.dim() == (2 if gauges == "one" else 1)
    law = SomaLaw(firing_mode="Novena", thresholding_mode="<=", firing_granularity="per_event",
                  membrane_arithmetic="unbounded", membrane_bits=0)
    match = "held per core position" if gauges == "one" else "canonical slot order"
    with pytest.raises(SerialFoldUnsupportedError, match=match):
        pf.count_read(program, "streamed_lif", 0, soma_law=law)


# ---------------------------------------------------------------------------
# m-7: a cache written before the channel-axis declaration
# ---------------------------------------------------------------------------


def _old_cache_conv_flow():
    """The convolution seam with per-channel LIF activations whose own ``channel_axis`` was never written (a cache of
    the export before this unit)."""
    from mimarsinan.models.nn.activations import LIFActivation

    flow, perceptrons = _flow("conv_seam_wide", 0, "drawn")
    for p in perceptrons:
        lif = LIFActivation(T=hg.T, activation_scale=p.activation_scale)
        p.base_activation = lif
        p.activation = lif
    for p in perceptrons:
        for m in p.modules():
            if hasattr(type(m), "channel_axis"):
                m.__dict__.pop("channel_axis", None)
    return flow, perceptrons


def _declared(p) -> set:
    return {int(m.channel_axis) for m in p.modules() if hasattr(type(m), "channel_axis")}


def test_a_perceptron_restored_after_its_owner_declares_its_axis_on_its_activations():
    """A copy that reaches the perceptron first and its owner (the convolution mapper, which re-declares the axis)
    inside it restores the owner before the perceptron: the load declares the axis all the same."""
    flow, perceptrons = _old_cache_conv_flow()
    reader = perceptrons[-1]
    node = next(n for n in flow.get_mapper_repr()._exec_order if getattr(n, "perceptron", None) is reader)
    reader.__dict__["_restored_with"] = node  # the owner reached from inside the perceptron
    assert _declared(reader) == {-1}
    copied = copy.deepcopy(reader)
    assert int(copied.output_channel_axis) == 1
    assert _declared(copied) == {1}
    x = torch.rand(2, int(reader.layer.weight.shape[0]), 6, 3)
    lif = copied.activation
    assert int(torch.as_tensor(lif.activation_scale).numel()) == int(x.shape[1]) > 1
    assert lif._safe_scale(x).shape == (int(x.shape[1]), 1, 1)


def test_a_whole_program_cache_load_declares_every_activation_s_axis():
    import io

    flow, _perceptrons = _old_cache_conv_flow()
    buffer = io.BytesIO()
    torch.save(flow, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    for p in loaded.get_perceptrons():
        assert _declared(p) == {int(p.output_channel_axis)}


class _NormalizedConvSeam(nn.Module):
    """The convolution seam whose reader carries a batch normalization (each output channel its own fold factor)."""

    shape = (2, 3, 3)

    def __init__(self, gen):
        super().__init__()
        self.enc = pf._conv(2, 3, 1, 0)
        self.a, self.b = pf._conv(3, 2, 1, 0), pf._conv(3, 2, 1, 0)
        self.conv, self.bn, self.act = nn.Conv2d(2, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU()
        pf._init(gen, self.modules())
        with torch.no_grad():
            self.bn.weight.copy_(torch.tensor([0.5, 1.5, 2.5, 0.75]))
            self.bn.running_var.copy_(torch.tensor([0.25, 1.0, 4.0, 0.5]))
            self.bn.running_mean.copy_(torch.tensor([0.1, -0.2, 0.0, 0.3]))

    def forward(self, x):
        e = self.enc(x)
        return self.act(self.bn(self.conv(torch.cat([self.a(e), self.b(e)], dim=2)))).flatten(1)


def test_a_normalized_reader_s_weight_view_folds_each_neuron_s_factor_on_its_axis(monkeypatch):
    """With a normalization, neuron ``c`` at position ``p`` reads ``fold_p W_c u_c / act_c``: the factor per output
    neuron lies on the declared axis (two scales apart per neuron and a fold apart per position)."""
    from mimarsinan.models.nn.layers import norm_affine_params

    monkeypatch.setitem(pf.FORMS, "normalized_seam", _NormalizedConvSeam)
    _flow_, perceptrons = _flow("normalized_seam")
    reader = next(p for p in perceptrons if p.per_input_scales is not None and p.per_input_scales.dim() == 2)
    assert not isinstance(reader.normalization, nn.Identity), "the reader carries the normalization"
    u = norm_affine_params(reader.normalization)[0].detach().reshape(-1)
    reference = _reference_weights(reader)
    for neuron in range(int(reference.shape[1])):
        reference[:, neuron] *= u[neuron]
    weights = PF.effective_parameters(reader, PerceptronTransformer())[0]
    torch.testing.assert_close(PF.outputs_first(weights, PF.output_axes(reader, True).weight),
                               reference.movedim(1, 0), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("form", ("conv_seam_wide", "conv_seam"))
def test_a_shift_per_input_element_charges_each_position_s_neurons_on_their_axis(form):
    """A shift of one value per element of the per-sample input (a convolution's input grid, more elements than the
    reader's inputs): each position's slots read their own elements (a padding slot none), its charge per neuron."""
    from mimarsinan.transformations.perceptron.position_bias import weight_charges

    flow, reader = _reader(form)
    node = next(n for n in flow.get_mapper_repr()._exec_order if getattr(n, "perceptron", None) is reader)
    shift = torch.linspace(0.05, 0.95, int(node.in_channels) * 6 * 3)
    slots = PF.shift_rows(reader, shift)
    table = node.patch_gather((int(node.in_channels), 6, 3)).slot_source_index
    reference = _reference_weights(reader)
    positions, out = int(reference.shape[0]), int(reference.shape[1])
    want = torch.zeros(out, positions)
    for position in range(positions):
        read = torch.tensor([float(shift[s]) if s >= 0 else 0.0 for s in table[position]])
        for neuron in range(out):
            want[neuron, position] = (reference[position, neuron] * read).sum()
    weights = PF.effective_parameters(reader, PerceptronTransformer())[0]
    torch.testing.assert_close(PF.outputs_first(weight_charges(weights, slots), PF.PER_POSITION.bias), want,
                               rtol=1e-5, atol=1e-6)


def test_the_starved_grid_scan_removes_a_per_input_reader_s_neuron_on_its_own_axis(monkeypatch):
    """One fold per input (every producer at one gauge): the dominating neuron's weights are zeroed, not an input
    column, and every other neuron keeps its weights."""
    import mimarsinan.transformations.perceptron.bias_canonicalization as canon

    monkeypatch.setattr(pf, "draw_gauges", lambda flow, ps, shape, seed, mode, scale=1.0: [torch.tensor(1.0)
                                                                                         for _ in ps])
    flow, perceptrons = _flow("token_mixing_3", 0, "one")
    reader = perceptrons[-1]
    assert reader.per_input_scales.dim() == 1, "one fold per input"
    out = int(reader.layer.weight.shape[0])
    with torch.no_grad():
        reader.layer.bias[out - 1] = float(reader.layer.weight.abs().max()) * 40.0
    before = reader.layer.weight.detach().clone()

    def accept(perceptron, model, batches, device, reference, apply):
        apply()
        return True

    monkeypatch.setattr(canon, "_apply_decision_verified", accept)
    canon._remove_nuisance_grid_dominators(reader, flow, [], "cpu", 4, None, PerceptronTransformer(),
                                           {"removed": 0, "removal_restored": 0})
    assert bool((reader.layer.weight[out - 1] == 0).all())
    torch.testing.assert_close(reader.layer.weight[: out - 1], before[: out - 1], rtol=0, atol=0)
