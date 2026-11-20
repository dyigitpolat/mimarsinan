"""[U-PFOLD] Weight quantization over a fold per position (the design review's J-3, correction 9).

One raw weight cannot carry several folds' rounding: each position's effective weight ``fold_p W / theta`` rounds to
its own codes on the perceptron's one grid. The projection therefore holds each weight row's projected raw weight beside
the raw weight (a linear one per position, a convolution one per distinct patch fold), and the raw weight stays the
trained latent every re-projection starts from. Every reader that must equal the deployed program reads those rows:
the mapper, both walks, the network function the NF and SCM parity reads, the QAT and endpoint re-projections (through
the transform trainer) and the grid checks. The parity of the network function and the mapped program is bit-exact,
with no budget (the owner's rule).

Red on U-TGAUGE4's tree by the fold's refusal (each form refused at its stamp before any projection).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import unit.spiking.host_gather_forms as hg
import unit.spiking.position_fold_forms as pf

from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
    assert_effective_parameters_on_chip_grid,
    assert_weights_within_layer_grid,
    quantization_grid_report,
)

SEMANTICS = ("streamed_lif", "windowed_lif")
Q_MAX = 2 ** (hg.WQ_BITS - 1) - 1
WQ_FORMS = ("token_mixing", "token_mixing_3", "tree481", "conv_seam", "conv_seam_tall", "shared_rows")


def _program(form: str, seed: int, semantics: str = "streamed_lif") -> hg.Program:
    latents: dict = {}

    def keep(_flow, perceptrons):
        latents.update({id(p): p.layer.weight.detach().clone() for p in perceptrons})

    program = pf.count_program(form, seed, semantics, wq=True, prepare=keep)
    program.latents = latents  # type: ignore[attr-defined]
    return program


@pytest.mark.parametrize("seed", (0, 1))
@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("form", WQ_FORMS)
def test_the_network_function_and_the_mapped_program_are_count_exact(form, semantics, seed):
    """The NF and SCM parity on the 4-bit program (every core's window counts, NF against the deployed flow, atol 0,
    no budget) and the walk against the deployed read element by element."""
    program = _program(form, seed, semantics)
    mismatches, compared, _ = hg.compare(program, semantics, n=16, seed=seed)
    assert compared > 0
    assert mismatches == {pi: 0 for pi in mismatches}, mismatches
    pf.assert_equal_reads(pf.count_read(program, semantics, seed))


@pytest.mark.parametrize("form", WQ_FORMS)
def test_each_weight_row_is_held_on_the_perceptron_s_one_grid(form):
    """Each weight row's effective weight is integer on the perceptron's one grid (``parameter_scale``, sized by the
    largest magnitude over every row and the bias), within its register; the raw weight is the latent it was projected
    from, unchanged; the grid checks of Quantization Verification hold per row."""

    program = _program(form, 0)
    reader = program.perceptrons[-1]
    assert reader.per_input_scales.dim() == 2, "the reader's fold is held per position"
    from mimarsinan.transformations.perceptron.position_fold import POSITION_WEIGHT, effective_weight_rows
    assert getattr(reader, POSITION_WEIGHT, None) is not None
    weights, index = effective_weight_rows(reader)
    codes = weights * reader.parameter_scale
    assert torch.allclose(codes, torch.round(codes), atol=1e-3, rtol=0)
    assert float(codes.abs().max()) <= Q_MAX + 0.5
    assert int(weights.shape[0]) == (int(torch.unique(reader.per_input_scales, dim=0).shape[0])
                                     if form.startswith("conv") else int(index.numel()))
    assert torch.equal(reader.layer.weight.detach(), program.latents[id(reader)]), "the raw weight is the latent"
    for p in program.perceptrons:
        assert_effective_parameters_on_chip_grid(p, Q_MAX)
        assert_weights_within_layer_grid(p, Q_MAX)
    report = {row["name"]: row for row in quantization_grid_report(program.perceptrons, Q_MAX)}
    assert report[str(reader.name)]["n_weights"] == int(weights.numel())


def _tuner(bits: int = hg.WQ_BITS):
    from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
        NormalizationAwarePerceptronQuantizationTuner as Tuner)

    tuner = Tuner.__new__(Tuner)
    tuner.quantization_bits = bits
    tuner.two_scale_projection = False
    tuner.fractional_scale, tuner.float_bias = False, False
    tuner.bias_rows_floor = None
    tuner.register_bias = False
    tuner.pipeline = SimpleNamespace(config={"device": "cpu"})
    return tuner


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam"))
def test_re_projections_through_the_transform_trainer_hold_each_row(form):
    """The QAT and endpoint re-projections (``_update_and_transform_model``): every step projects a copy of the float
    master, and the deployed perceptron takes each row's projected weight with it; the master never absorbs a
    projection, and an unchanged master re-projects to the same rows (no drift)."""
    from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer

    program = pf.count_program(form, 0)
    main = program.perceptrons[-1]
    assert main.per_input_scales.dim() == 2, "the reader's fold is held per position"
    from mimarsinan.transformations.perceptron.position_fold import POSITION_WEIGHT, effective_weight_rows

    aux = copy.deepcopy(main)
    temp = copy.deepcopy(aux)
    trainer = PerceptronTransformTrainer.__new__(PerceptronTransformTrainer)
    trainer._perceptron_slots = [(main, aux, temp)]
    trainer.perceptron_transformation = _tuner()._mixed_transform(1.0)
    gen = torch.Generator().manual_seed(3)
    for _ in range(3):
        with torch.no_grad():
            aux.layer.weight.add_(0.02 * torch.randn(aux.layer.weight.shape, generator=gen))
        master = aux.layer.weight.detach().clone()
        trainer._update_and_transform_model()
        assert torch.equal(aux.layer.weight.detach(), master), "the master absorbs no projection"
        held = getattr(main, POSITION_WEIGHT, None)
        assert held is not None, "the deployed perceptron takes each row's projected weight"
        codes = effective_weight_rows(main)[0] * main.parameter_scale
        assert torch.allclose(codes, torch.round(codes), atol=1e-3, rtol=0)
        fresh = copy.deepcopy(aux)
        _tuner()._mixed_transform(1.0)(fresh)
        assert torch.equal(held, getattr(fresh, POSITION_WEIGHT)), "the deployed rows are this step's projection"
    before = getattr(main, POSITION_WEIGHT).detach().clone()
    trainer._update_and_transform_model()
    assert torch.equal(getattr(main, POSITION_WEIGHT).detach(), before), "no drift"


def test_the_network_function_trains_through_the_projected_rows():
    """The per-position forward reads each row's projected weight forward and passes the gradient to the latent raw
    weight (the transform trainer's straight-through surrogate)."""

    program = _program("tree481", 0)
    from mimarsinan.transformations.perceptron.position_fold import POSITION_WEIGHT
    reader = program.perceptrons[-1]
    held = getattr(reader, POSITION_WEIGHT)
    positions, fan_in = int(reader.per_input_scales.shape[0]), int(reader.layer.weight.shape[1])
    x = torch.rand(4, positions, fan_in)
    captured: dict = {}
    handle = reader.activation.register_forward_pre_hook(lambda _m, args: captured.update(z=args[0]))
    try:
        reader(x)
    finally:
        handle.remove()
    want = torch.stack([x[:, p] @ held[p].t() for p in range(positions)], dim=1) + reader.layer.bias
    torch.testing.assert_close(captured["z"].detach(), want.detach(), rtol=1e-6, atol=1e-6)
    captured["z"].sum().backward()
    torch.testing.assert_close(reader.layer.weight.grad, x.sum(dim=(0, 1)).expand_as(reader.layer.weight).clone())


def test_a_per_input_perceptron_keeps_today_s_projection():
    """Guard (green before and after): a perceptron whose fold is held per input projects into its raw weight as
    before, and holds no rows."""
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

    p = Perceptron(3, 4, normalization=nn.Identity())
    p.per_input_scales = torch.tensor([0.5, 1.0, 2.0, 1.5])
    before = p.layer.weight.detach().clone()
    transform = _tuner()._get_new_perceptron_transform(1.0)
    with torch.no_grad():
        transform(p)
    assert getattr(p, "position_weight", None) is None
    assert not torch.equal(p.layer.weight.detach(), before)
    assert_effective_parameters_on_chip_grid(p, Q_MAX)


def test_normalization_fusion_carries_each_projected_row():
    """Normalization Fusion scales the raw weight per output channel by the normalization's factor: each projected
    row of a fold per position is scaled alike, so every position's effective weight is unchanged by the fusion."""
    from mimarsinan.models.nn.layers import FrozenStatsNormalization
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
    from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron

    torch.manual_seed(0)
    norm = nn.BatchNorm1d(3).eval()
    with torch.no_grad():
        norm.running_mean.uniform_(-0.2, 0.2)
        norm.running_var.uniform_(0.5, 1.5)
        norm.weight.uniform_(0.5, 1.5)
    p = Perceptron(3, 4, normalization=FrozenStatsNormalization(norm))
    p.set_activation_scale(torch.tensor(1.3))
    p.per_input_scales = torch.tensor([[0.5, 1.0, 2.0, 1.5], [1.5, 0.7, 0.9, 1.1]])
    _tuner()._get_new_perceptron_transform(1.0)(p)
    from mimarsinan.transformations.perceptron.position_fold import POSITION_WEIGHT, effective_weight_rows
    assert getattr(p, POSITION_WEIGHT, None) is not None
    before = effective_weight_rows(p)[0].detach().clone()
    fuse_into_perceptron(p, device="cpu")
    torch.testing.assert_close(effective_weight_rows(p)[0], before, rtol=1e-6, atol=1e-6)


def _stamped(form: str):
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

    _model, flow, shape = pf.converted(form, 0)
    perceptrons = list(flow.get_perceptrons())
    for p, g in zip(perceptrons, pf.draw_gauges(flow, perceptrons, shape, 0, "one")):
        p.set_activation_scale(g)
    compute_per_source_scales(flow.get_mapper_repr())
    reader = perceptrons[-1]
    assert reader.per_input_scales.dim() == 2, "the reader's fold is held per position"
    return flow, reader


def test_a_pruned_entry_stays_zero_on_every_projected_row():
    """A pruned weight entry is zero on every projected row, whatever the latent holds between commits (the float-bias
    projection writes no bias, so nothing commits the mask before the rows are projected)."""
    from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
        NormalizationAwarePerceptronQuantization)

    _flow, reader = _stamped("token_mixing")
    from mimarsinan.transformations.perceptron.position_fold import POSITION_WEIGHT
    mask = torch.zeros_like(reader.layer.weight, dtype=torch.bool)
    mask[0, 0] = True
    reader.layer.register_buffer("prune_mask", mask)
    with torch.no_grad():
        reader.layer.weight[0, 0] = 0.7
    NormalizationAwarePerceptronQuantization(hg.WQ_BITS, "cpu", rate=1.0, float_bias=True).transform(reader)
    assert bool((getattr(reader, POSITION_WEIGHT)[:, 0, 0] == 0).all())


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam"))
def test_a_projection_puts_every_position_s_own_bias_on_the_grid(form):
    """With a shift baked per position, the projection quantizes every position's own bias on the perceptron's grid,
    each within half a step of its compensated value ``B_p - Q(W~_p) s`` (the bias transform on every position's)."""
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

    _flow, reader = _stamped(form)
    from mimarsinan.transformations.perceptron.position_fold import bias_offsets, effective_weight_rows, shift_rows
    unbaked = PerceptronTransformer(positions=True).get_effective_bias(reader).detach().clone()
    apply_negative_shift_bias(reader, 0.25)
    assert bias_offsets(reader) is not None
    _tuner()._get_new_perceptron_transform(1.0)(reader)
    assert_effective_parameters_on_chip_grid(reader, Q_MAX)
    weights, index = effective_weight_rows(reader)
    want = unbaked - (weights[index] * shift_rows(reader, torch.tensor(0.25))).sum(dim=-1)
    got = PerceptronTransformer(positions=True).get_effective_bias(reader)
    assert float((got - want).abs().max()) <= 0.5 / float(reader.bias_scale) + 1e-5


def test_a_channel_removed_after_the_projection_is_zero_on_every_row():
    """An output channel's removal zeroes it on the raw weight and on every projected row."""
    program = _program("token_mixing_3", 0)
    reader = program.perceptrons[-1]
    from mimarsinan.transformations.perceptron.position_bias import zero_output_channel
    from mimarsinan.transformations.perceptron.position_fold import effective_weight_rows
    zero_output_channel(reader, 1)
    weights, _index = effective_weight_rows(reader)
    assert bool((weights[:, 1] == 0).all()) and bool((reader.layer.weight[1] == 0).all())
    assert bool((weights[:, 0] != 0).any())


def test_the_affine_fold_scales_every_projected_row():
    """LIF Adaptation's affine fold into a projected consumer scales each input column of every projected row (as of
    the raw weight), so every position's effective weight takes the producer's gain."""
    program = _program("token_mixing", 0)
    reader = program.perceptrons[-1]
    from mimarsinan.transformations.perceptron.position_fold import effective_weight_rows
    from mimarsinan.tuning.lif_affine_fold import fold_affine_into_consumer
    before = effective_weight_rows(reader)[0].detach().clone()
    gains = torch.linspace(0.5, 1.5, int(reader.layer.weight.shape[1]))
    fold_affine_into_consumer(reader, gains, torch.zeros_like(gains))
    torch.testing.assert_close(effective_weight_rows(reader)[0], before * gains, rtol=1e-6, atol=1e-7)


def test_the_network_function_reads_each_position_s_bias_offset():
    """Before any projection, a bias held per position (a shift baked per position) enters the network function at
    each position: ``W x_p + b + offset_p theta``."""
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias

    _flow, reader = _stamped("token_mixing")
    from mimarsinan.transformations.perceptron.position_fold import POSITION_WEIGHT, bias_offsets
    apply_negative_shift_bias(reader, 0.3)
    offsets = bias_offsets(reader)
    assert offsets is not None and getattr(reader, POSITION_WEIGHT, None) is None
    positions, fan_in = int(reader.per_input_scales.shape[0]), int(reader.layer.weight.shape[1])
    x = torch.rand(4, positions, fan_in)
    captured: dict = {}
    handle = reader.activation.register_forward_pre_hook(lambda _m, args: captured.update(z=args[0]))
    try:
        with torch.no_grad():
            reader(x)
    finally:
        handle.remove()
    want = x @ reader.layer.weight.detach().t() + reader.layer.bias.detach() + offsets * reader.activation_scale
    torch.testing.assert_close(captured["z"], want, rtol=1e-6, atol=1e-6)


def test_quantization_verification_reads_every_weight_row():
    """The grid checks read every weight row: a row other than the first off the grid, or past the register, fails
    them."""
    program = _program("token_mixing", 0)
    reader = program.perceptrons[-1]
    from mimarsinan.transformations.perceptron.position_fold import POSITION_WEIGHT
    with torch.no_grad():
        getattr(reader, POSITION_WEIGHT)[-1].mul_(1.13)
    with pytest.raises(AssertionError):
        assert_effective_parameters_on_chip_grid(reader, Q_MAX)
    with torch.no_grad():
        getattr(reader, POSITION_WEIGHT)[-1].mul_(8.0)  # a row past the register
    with pytest.raises(AssertionError, match="outside"):
        assert_weights_within_layer_grid(reader, Q_MAX)
