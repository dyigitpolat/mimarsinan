"""[U-RSHIFT] One test per site of the received-shift class, on converter-built forms (hand-built where the converter builds
none): every reader's correction is its own response to the shift as each axon receives it.

The read (``received_shift_forms``) is the reader's own identity at the effective level: on its input plus the field each
axon receives, placed by the form's own construction, the reader equals the trained reader at every core position within
float dust of max(1, |value|). Each red names today's error in gauges and raw units. The sites: Soft Core Mapping's
shift (T1, T2, T4), the seam's lift through a join (T3) and through a padded average pool (T5), the seam's fallback (T6),
the opt-in signed-seam install (T7), a host bias carrier and a LayerNorm reading a join (T8, T9), Weight Quantization's
compensation (T10), the affine fold (T11, and its pin: no compensation at a later projection), an operand product
(T12), the host-side encoder (T13), the stamp writer (T14), two shifted sources (T17), the policy called twice (T18),
and a host attention module's operands (C7).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.transformations.perceptron import position_fold
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

from unit.mapping.received_shift import received_shift_forms as rf

REFUSED = (NotImplementedError, ToolchainRefusal)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(4000 + seed)


def _sources(repr_) -> list:
    return [o for o in rf.ops(repr_) if rf.own_shift(o) is not None]


def _assert_exact(node, d, before, what: str) -> dict:
    read = rf.reader_error(node, d, before)
    assert read["max"] <= read["tol"], (
        f"{what}: {node.name} departs from the trained reader by {read['max']:.4g} of its gauge ({read['raw']:.4g} raw): "
        f"border {read['border']:.4g}, inner {read['inner']:.4g} (min over positions {read['inner_min']:.4g})")
    return read


def _held_per_position(node) -> bool:
    return position_fold.holds_positions(node.perceptron)


# ---------------------------------------------------------------------------------------------------------------------
# T1: Soft Core Mapping's shift into a zero-padded reader (the padding record's 16 forms)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pre", ("none", "avg", "max", "proj"))
@pytest.mark.parametrize("k", (3, 5))
@pytest.mark.parametrize("s", (1, 2))
def test_t1_soft_core_mappings_shift_into_a_padded_reader(pre, k, s):
    model = rf.PadRead(_gen(0), pre, k, s)
    flow, repr_, x = rf.build(model, 0)
    reader = rf.conv_readers(repr_)[-1]
    before = rf.biases([reader])
    rf.sequence(flow, x, "policy")
    (op,) = _sources(repr_)
    d = rf.field_full(rf.input_shape(flow, reader, model), rf.own_shift(op))
    _assert_exact(reader, d, before[id(reader.perceptron)], f"T1 {pre} k{k} s{s}")


# ---------------------------------------------------------------------------------------------------------------------
# T2, T3, T4: the structural join (a shifted and an unshifted host value), read by a 1x1 and a 3x3 zero-padded core
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sequence", ("policy", "seam"))
@pytest.mark.parametrize("k", (1, 3))
def test_t2_t3_t4_a_join_of_a_shifted_and_an_unshifted_value(sequence, k):
    """T2 (the policy, 1x1), T3 (the seam's lift through the join, 1x1), T4 (3x3, both sequences: the seam leaves it to
    the policy). Only the normalization's axons receive the shift; a 1x1 reader holds one bias, a padded one a bias per
    position."""
    model = rf.Join(_gen(0), k)
    flow, repr_, x = rf.build(model, 0)
    reader = rf.conv_readers(repr_)[-1]
    before = rf.biases([reader])
    rf.sequence(flow, x, sequence)
    bn = rf.op_named(repr_, "bn")
    d = rf.field_parts(rf.input_shape(flow, reader, model), [(4, rf.own_shift(bn)), (4, 0.0)])
    _assert_exact(reader, d, before[id(reader.perceptron)], f"T{2 if sequence == 'policy' else 3}/T4 join k{k}")
    assert _held_per_position(reader) == (k == 3), "a 1x1 reader of a join holds one bias, a padded one each position's"


# ---------------------------------------------------------------------------------------------------------------------
# T5: the seam's lift through a zero-padded average pool that counts its padding
# ---------------------------------------------------------------------------------------------------------------------
def test_t5_the_seams_lift_through_a_padded_average_pool():
    model = rf.PoolLift(_gen(0))
    flow, repr_, x = rf.build(model, 0)
    reader = next(n for n in rf.conv_readers(repr_) if n.name.startswith("reader"))
    before = rf.biases([reader])
    printed = rf.sequence(flow, x, "seam")
    bn = rf.op_named(repr_, "bn")
    assert bn.output_value_offset is not None and float(bn.output_value_offset) > 0, f"fixture: the lift ({printed!r})"
    shape = rf.input_shape(flow, reader, model)
    share = nn.AvgPool2d(3, 1, 1)(torch.ones(1, *shape, dtype=torch.float64))[0]  # 4/9, 6/9 and 9/9
    _assert_exact(reader, share * rf.own_shift(bn), before[id(reader.perceptron)], "T5 padded average pool")


# ---------------------------------------------------------------------------------------------------------------------
# T6: the seam's fallback, two readers of different padding; T7: the opt-in signed-seam install
# ---------------------------------------------------------------------------------------------------------------------
def test_t6_the_seams_fallback_for_two_readers_of_different_padding():
    model = rf.TwoPads(_gen(0))
    flow, repr_, x = rf.build(model, 0)
    readers = rf.conv_readers(repr_)[-2:]
    before = rf.biases(readers)
    rf.sequence(flow, x, "seam")
    bn = rf.op_named(repr_, "bn")
    assert bn.output_value_offset is None and getattr(bn, "_negative_shift", None) is not None, "fixture: the fallback"
    for reader in readers:
        d = rf.field_full(rf.input_shape(flow, reader, model), rf.own_shift(bn))
        _assert_exact(reader, d, before[id(reader.perceptron)], "T6 two paddings")


def test_t7_the_opt_in_signed_seam_install():
    model = rf.OneRead(_gen(0), 3)
    flow, repr_, x = rf.build(model, 0)
    reader = rf.conv_readers(repr_)[-1]
    before = rf.biases([reader])
    rf.sequence(flow, x, "install")
    bn = rf.op_named(repr_, "bn")
    assert bn.output_value_offset is not None, "fixture: the install lifts the normalization"
    d = rf.field_full(rf.input_shape(flow, reader, model), rf.own_shift(bn))
    _assert_exact(reader, d, before[id(reader.perceptron)], "T7 install")


# ---------------------------------------------------------------------------------------------------------------------
# T8, T9: a host bias carrier and a LayerNorm reading the join under the lift
# ---------------------------------------------------------------------------------------------------------------------
def _host_join(kind: str):
    model = rf.HostJoin(_gen(0), kind)
    flow, repr_, x = rf.build(model, 0)
    host = next(o for o in rf.ops(repr_) if isinstance(o.module, (nn.Linear, nn.LayerNorm)))
    trained = copy.deepcopy(host.module)
    with torch.no_grad():
        e = model.enc(x[:8])
        j = torch.cat([model.bn(e), model.mp(e)], dim=1).flatten(1).double()
    return model, flow, repr_, x, host, trained, j


def test_t8_a_host_linear_reading_the_join_takes_one_bias_for_what_it_receives():
    model, flow, repr_, x, host, trained, j = _host_join("linear")
    rf.sequence(flow, x, "seam")
    bn = rf.op_named(repr_, "bn")
    assert bn.output_value_offset is not None, "fixture: the lift is taken"
    half = j.shape[1] // 2
    d = torch.zeros_like(j[:1])
    d[:, :half] = float(bn.output_value_offset)  # value units: only the normalization's half receives the lift
    got = host.module.double()(j + d)
    want = trained.double()(j)
    err = float((got - want).abs().max())
    assert err <= rf.RELATIVE * max(1.0, float(want.abs().max())), f"T8: the host Linear departs by {err:.4g}"


def test_t9_a_layernorm_reading_a_varying_field_is_refused_by_name():
    model, flow, repr_, x, host, trained, j = _host_join("layernorm")
    try:
        rf.sequence(flow, x, "seam")
    except REFUSED as exc:
        assert "ComputeOp" in str(exc) or "LayerNorm" in str(exc), str(exc)
        assert rf.op_named(repr_, "bn").output_value_offset is None, "the lift is not taken"
        return
    bn = rf.op_named(repr_, "bn")
    half = j.shape[1] // 2
    d = torch.zeros_like(j[:1])
    if bn.output_value_offset is not None:
        d[:, :half] = float(bn.output_value_offset)
    err = float((host.module.double()(j + d) - trained.double()(j)).abs().max())
    pytest.fail(f"T9: a LayerNorm reading a join of a lifted and an unlifted value was treated as absorbing the lift: "
                f"its output departs by {err:.4g}")


# ---------------------------------------------------------------------------------------------------------------------
# T10: Weight Quantization's compensation of a recorded field (4 bits, the law route's float bias)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k,s", ((3, 1), (5, 2)))
def test_t10_weight_quantizations_compensation_follows_each_positions_field(k, s):
    model = rf.PadRead(_gen(0), "none", k, s)
    flow, repr_, x = rf.build(model, 0)
    reader = rf.conv_readers(repr_)[-1]
    before = rf.biases([reader])
    rf.sequence(flow, x, "policy")
    (op,) = _sources(repr_)
    d = rf.field_full(rf.input_shape(flow, reader, model), rf.own_shift(op))
    baked = rf.reader_error(reader, d, before[id(reader.perceptron)])
    w0, b0 = rf.effective_rows(reader.perceptron)
    rf.project(list(flow.get_perceptrons()))
    w1, b1 = rf.effective_rows(reader.perceptron)
    rows = rf.slot_rows(reader, d).unsqueeze(1)
    compensation = (b1 - b0) + ((w1 - w0) * rows).sum(-1)  # the bias's move against the weights' move on the field
    comp = float(compensation.abs().max())
    after = rf.reader_error(reader, d, before[id(reader.perceptron)])
    assert comp <= 2 * after["tol"], (
        f"T10 k{k} s{s}: the projection's compensation departs from each position's field by {comp:.4g} of the gauge "
        f"(the bake itself by {baked['max']:.4g})")
    assert after["max"] <= 2 * after["tol"], f"T10 k{k} s{s}: after the projection the reader departs by {after['max']:.4g}"


class _WholeKernel(nn.Module):
    """A 5x5 zero-padded convolution over a 5x5 map: its input holds as many elements as its kernel holds slots, so a
    field per element and one value per input have one length (the record keeps the input's own shape)."""

    def __init__(self, gen):
        super().__init__()
        self.shape = (2, 5, 5)
        self.enc = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ReLU())
        self.bn = nn.BatchNorm2d(3)
        self.reader = nn.Sequential(nn.Conv2d(3, 4, 5, padding=2), nn.ReLU())
        rf.init_(self, gen)

    def forward(self, x):
        return self.reader(self.bn(self.enc(x))).flatten(1)


def test_t10_a_field_per_element_as_long_as_the_fan_in_is_compensated_per_element():
    model = _WholeKernel(_gen(0))
    flow, repr_, x = rf.build(model, 0)
    reader = rf.conv_readers(repr_)[-1]
    before = rf.biases([reader])
    rf.sequence(flow, x, "policy")
    (op,) = _sources(repr_)
    d = rf.field_full(rf.input_shape(flow, reader, model), rf.own_shift(op))
    assert d.numel() == int(reader.perceptron.input_features), "fixture: the input's elements equal the kernel's slots"
    w0, b0 = rf.effective_rows(reader.perceptron)
    rf.project([reader.perceptron])
    w1, b1 = rf.effective_rows(reader.perceptron)
    compensation = (b1 - b0) + ((w1 - w0) * rf.slot_rows(reader, d).unsqueeze(1)).sum(-1)
    after = rf.reader_error(reader, d, before[id(reader.perceptron)])
    comp = float(compensation.abs().max())
    assert comp <= 2 * after["tol"] and after["max"] <= 2 * after["tol"], (
        f"T10 whole kernel: the compensation departs from each position's field by {comp:.4g}, the reader by "
        f"{after['max']:.4g} of its gauge")


def test_t10_a_record_per_element_is_read_per_element_at_a_projection():
    """A field that varies inside a channel, over a kernel as large as the map (the record's length is the fan-in): each
    position's compensation reads the record by its own patch, not slot for element."""
    from mimarsinan.mapping.support.received_shift import correct_for_received_shift, received_for

    model = _WholeKernel(_gen(0))
    flow, repr_, _x = rf.build(model, 0)
    reader = rf.conv_readers(repr_)[-1]
    field = torch.linspace(0.1, 1.0, 75, dtype=torch.float64).reshape(1, 3, 5, 5)
    received = received_for(reader, {"ramp": field}, {"ramp": torch.tensor(0.0)})
    assert correct_for_received_shift(reader.perceptron, received) == "positions"
    w0, b0 = rf.effective_rows(reader.perceptron)
    rf.project([reader.perceptron])
    w1, b1 = rf.effective_rows(reader.perceptron)
    compensation = (b1 - b0) + ((w1 - w0) * rf.slot_rows(reader, field[0]).unsqueeze(1)).sum(-1)
    assert float(compensation.abs().max()) <= 2e-5 * max(1.0, float(b1.abs().max())), float(compensation.abs().max())


def test_t14_positions_whose_corrections_coincide_keep_their_own_bias_through_the_stamping_walk_and_a_projection():
    """Field rows that differ only on slots whose weights are zero give every position one correction today; the
    positions still hold their own bias (the rows decide), so the stamping walk keeps the stamp per position and a later
    projection re-derives each position's compensation (green before and after: today's bake is exact here)."""
    model = rf.PadRead(_gen(0), "none", 3, 1)
    flow, repr_, x = rf.build(model, 0)
    reader = rf.conv_readers(repr_)[-1]
    with torch.no_grad():
        weight = reader.perceptron.layer.weight
        mask = torch.zeros(3, 3)
        mask[1, 1] = 1.0
        weight.mul_(mask.reshape(1, 1, 9).expand(weight.shape[0], 3, 9).reshape(weight.shape))
    before = rf.biases([reader])
    rf.sequence(flow, x, "policy")
    compute_per_source_scales(repr_, arm_wire_value_ops=True)
    (op,) = _sources(repr_)
    d = rf.field_full(rf.input_shape(flow, reader, model), rf.own_shift(op))
    rf.project([reader.perceptron])
    read = rf.reader_error(reader, d, before[id(reader.perceptron)])
    assert read["max"] <= 2 * read["tol"], read


# ---------------------------------------------------------------------------------------------------------------------
# T11 and the affine fold's pin
# ---------------------------------------------------------------------------------------------------------------------
def _fold(repr_, producer, consumer, a, c, k2):
    """The affine fold of ``producer``'s output offset ``c`` (per channel) into ``consumer``: the step's graph-level
    fold where the tree defines it, else the per-reader fold the tree has (its step voids a padded consumer)."""
    from mimarsinan.tuning import lif_affine_fold

    if hasattr(lif_affine_fold, "fold_affine_into_consumers"):
        field = c.view(-1, 1, 1).expand(-1, 10, 10).reshape(-1)
        lif_affine_fold.fold_affine_into_consumers(repr_, producer, [consumer.perceptron], a.repeat_interleave(k2), field)
    else:
        lif_affine_fold.fold_affine_into_consumer(consumer.perceptron, a.repeat_interleave(k2), c.repeat_interleave(k2))


def _core_chain():
    model = rf.CoreChain(_gen(0))
    flow, repr_ = rf.convert(model)
    for p in flow.get_perceptrons():
        p.set_activation_scale(torch.tensor(1.0))
    compute_per_source_scales(repr_)
    a_node, b_node = rf.conv_readers(repr_)[-2:]
    gen = torch.Generator().manual_seed(71)
    a = 0.5 + torch.rand(3, generator=gen)
    c = 0.3 * torch.randn(3, generator=gen)
    return flow, repr_, a_node, b_node, a, c


def test_t11_the_affine_fold_into_a_padded_consumer():
    flow, repr_, a_node, b_node, a, c = _core_chain()
    before = rf.biases([b_node])[id(b_node.perceptron)]
    w_before = PerceptronTransformer().get_effective_weight(b_node.perceptron).detach().double()
    k2 = int(b_node.kernel_size[0] * b_node.kernel_size[1])
    _fold(repr_, a_node, b_node, a, c, k2)
    field = c.double().view(3, 1, 1).expand(3, 10, 10)
    want = (w_before.unsqueeze(0) * rf.slot_rows(b_node, field).unsqueeze(1)).sum(-1) + before.reshape(1, -1)
    _w, b_now = rf.effective_rows(b_node.perceptron)
    err = float((b_now - want).abs().max())
    assert err <= rf.RELATIVE * max(1.0, float(want.abs().max())), (
        f"T11: the folded consumer's bias departs from its own response to the offset it receives by {err:.4g} "
        f"(a padding slot receives none of it)")


def test_c5_an_affine_folded_readers_bias_takes_no_compensation_at_a_later_projection():
    """The fold writes through the one writer at sign +1 and records nothing, so a later projection re-derives no
    compensation of it: the law route's float bias is left exactly as folded (green before and after the unit)."""
    flow, repr_, a_node, b_node, a, c = _core_chain()
    k2 = int(b_node.kernel_size[0] * b_node.kernel_size[1])
    _fold(repr_, a_node, b_node, a, c, k2)
    assert getattr(b_node.perceptron, "input_shift", None) is None, "the fold records no received shift"
    _w, folded = rf.effective_rows(b_node.perceptron)
    rf.project([b_node.perceptron])
    _w, projected = rf.effective_rows(b_node.perceptron)
    assert torch.equal(folded, projected), float((folded - projected).abs().max())


# ---------------------------------------------------------------------------------------------------------------------
# T12: an operand product reading a shifted value (a hand-built mapper graph)
# ---------------------------------------------------------------------------------------------------------------------
class _Product(Mapper):
    """A stand-in operand product between a shifted host op and a perceptron: it declares its operand banks and moves
    its input as is, the shape the structural walk reads as a structural node."""

    def __init__(self, source):
        super().__init__(source)
        self.operand_perceptrons = nn.ModuleList()

    def _forward_impl(self, x):
        return x


def _product_graph():
    from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
    from mimarsinan.mapping.mappers.structural import InputMapper
    from mimarsinan.mapping.model_representation import ModelRepresentation
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

    torch.manual_seed(3)
    inp = InputMapper((4,))
    op = ComputeOpMapper(inp, nn.Tanh(), input_shape=(4,), output_shape=(4,), name="signed")
    reader = Perceptron(3, 4, normalization=nn.Identity())
    repr_ = ModelRepresentation(PerceptronMapper(_Product(op), reader))

    class _Model:
        def get_mapper_repr(self):
            return repr_
    return _Model(), op, reader


def test_t12_an_operand_product_reading_a_shifted_value_is_refused_by_name():
    from mimarsinan.mapping.support.negative_shift import apply_negative_value_shifts

    model, op, reader = _product_graph()
    bias = reader.layer.bias.detach().clone()
    try:
        apply_negative_value_shifts(model, {op: torch.tensor([-0.5])})
    except REFUSED as exc:
        assert "operand product" in str(exc), str(exc)
        assert torch.equal(reader.layer.bias.detach(), bias), "a refusal before any bake"
        return
    moved = float((reader.layer.bias.detach() - bias).abs().max())
    pytest.fail(f"T12: the shift was carried through an operand product as if it passed unchanged: the reader past it "
                f"was baked by {moved:.4g}")


# ---------------------------------------------------------------------------------------------------------------------
# T13: a padded encoder on the host reading a shifted host op; T14: the stamp writer
# ---------------------------------------------------------------------------------------------------------------------
def test_t13_a_padded_encoder_on_the_host_holds_its_positions_bias():
    model = rf.HostEncoder(_gen(0))
    flow, repr_ = rf.convert(model)
    encoder = next(n for n in repr_.execution_order() if isinstance(n, rf.Conv2DPerceptronMapper)
                   and tuple(n.padding) == (1, 1))
    encoder.perceptron.is_encoding_layer = True  # by hand: the converter places no encoder after a host op here
    x = torch.randn(48, *model.shape, generator=torch.Generator().manual_seed(700))
    rf.gauges(flow, repr_, x)
    before = rf.biases([encoder])
    rf.sequence(flow, x, "policy")
    bn = rf.op_named(repr_, "bn")
    assert getattr(bn, "_negative_shift", None) is not None, "fixture: the policy shifts the raw input's normalization"
    d = rf.field_full(rf.input_shape(flow, encoder, model), rf.own_shift(bn))
    _assert_exact(encoder, d, before[id(encoder.perceptron)], "T13 host-side encoder")


def test_t14_the_stamp_writer_keeps_each_positions_fold_where_positions_hold_their_bias():
    model = rf.PadRead(_gen(0), "none", 3, 1)
    flow, repr_, x = rf.build(model, 0)
    reader = rf.conv_readers(repr_)[-1]
    before = rf.biases([reader])
    rf.sequence(flow, x, "policy")
    compute_per_source_scales(repr_, arm_wire_value_ops=True)  # Soft Core Mapping's stamping walk after the policy
    (op,) = _sources(repr_)
    d = rf.field_full(rf.input_shape(flow, reader, model), rf.own_shift(op))
    _assert_exact(reader, d, before[id(reader.perceptron)], "T14 after the stamping walk")
    stamp = reader.perceptron.per_input_scales
    assert stamp is not None and stamp.dim() == 2, f"the stamp collapsed to {tuple(stamp.shape)}"


# ---------------------------------------------------------------------------------------------------------------------
# T17: two shifted sources joined into one reader; T18: the policy called twice
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sequence", ("policy", "seam", "install"))
@pytest.mark.parametrize("k", (1, 3))
def test_t17_a_reader_of_two_shifted_values_is_corrected_for_both(sequence, k):
    model = rf.TwoShifted(_gen(0), k)
    flow, repr_, x = rf.build(model, 0)
    reader = rf.conv_readers(repr_)[-1]
    before = rf.biases([reader])
    rf.sequence(flow, x, sequence)
    a, b = rf.op_named(repr_, "bna"), rf.op_named(repr_, "bnb")
    assert rf.own_shift(a) and rf.own_shift(b), "fixture: both normalizations are shifted"
    d = rf.field_parts(rf.input_shape(flow, reader, model), [(3, rf.own_shift(a)), (3, rf.own_shift(b))])
    _assert_exact(reader, d, before[id(reader.perceptron)], f"T17 k{k} {sequence}")


def _state(p) -> list:
    held = [p.layer.bias, getattr(p, "position_bias", None), p.per_input_scales, getattr(p, "input_shift", None)]
    return [None if t is None else t.detach().clone() for t in held]


def test_t18_the_policy_called_twice_writes_nothing_the_second_time():
    model = rf.TwoShifted(_gen(0), 3)
    flow, repr_, x = rf.build(model, 0)
    reader = rf.conv_readers(repr_)[-1]
    rf.policy(flow, x)
    first = _state(reader.perceptron)
    rf.policy(flow, x)
    second = _state(reader.perceptron)
    assert all((u is None and v is None) or (u is not None and v is not None and torch.equal(u, v))
               for u, v in zip(first, second)), "the second call wrote the reader"
    a, b = rf.op_named(repr_, "bna"), rf.op_named(repr_, "bnb")
    shape = rf.input_shape(flow, reader, model)
    want = rf.field_parts(shape, [(3, rf.own_shift(a)), (3, rf.own_shift(b))]).reshape(-1).float()
    record = reader.perceptron.input_shift.reshape(-1).float()
    record = record.expand_as(want) if record.numel() == 1 else record
    assert record.shape == want.shape and torch.equal(record, want), (
        f"the record holds {sorted(set(np.round(record.numpy(), 6).tolist()))}, the stamped sources deliver "
        f"{sorted(set(np.round(want.numpy(), 6).tolist()))}")


# ---------------------------------------------------------------------------------------------------------------------
# C7: a host attention module whose query is lifted and whose key and value are not (a hand-built mapper graph)
# ---------------------------------------------------------------------------------------------------------------------
def test_c7_a_host_attention_modules_projections_each_take_their_own_operands_field():
    from mimarsinan.mapping.mappers.structural import InputMapper
    from mimarsinan.mapping.model_representation import ModelRepresentation
    from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
    from mimarsinan.mapping.mappers.structural import ConcatMapper
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
    from mimarsinan.tuning.orchestration.signed_seam_install import _bake_shift_into_host_bias

    torch.manual_seed(5)
    inp = InputMapper((4, 6))
    query = ComputeOpMapper(inp, nn.Tanh(), input_shape=(4, 6), output_shape=(4, 6), name="query")
    keys = ComputeOpMapper(inp, nn.Sigmoid(), input_shape=(4, 6), output_shape=(4, 6), name="keys")
    mha = nn.MultiheadAttention(6, 2, batch_first=True).eval()
    attention = ComputeOpMapper([query, keys, keys], mha, input_shapes=[(4, 6)] * 3, output_shape=(4, 6),
                                module_kwargs={"need_weights": False}, output_index=0, name="attention",
                                elementwise_inputs=False)
    readers = [PerceptronMapper(query, Perceptron(5, 6, normalization=nn.Identity())),
               PerceptronMapper(attention, Perceptron(5, 6, normalization=nn.Identity()))]
    repr_ = ModelRepresentation(ConcatMapper(readers, dim=-1))
    trained = copy.deepcopy(mha)
    sigma = 0.37
    x = torch.randn(3, 4, 6)
    q, kv = torch.tanh(x), torch.sigmoid(x)
    from mimarsinan.tuning.orchestration import signed_seam_install

    if hasattr(signed_seam_install, "bake_received_shift"):
        signed_seam_install.bake_received_shift(repr_, {query: torch.tensor(sigma)})
    else:  # the tree's own bake of a lift's host carrier: every projection of the module at once
        assert _bake_shift_into_host_bias(mha, sigma)
    with torch.no_grad():
        got, _ = mha(q + sigma, kv, kv, need_weights=False)
        want, _ = trained(q, kv, kv, need_weights=False)
    err = float((got - want).abs().max())
    assert err <= 1e-5, f"C7: the attention output departs by {err:.4g} (the key and value projections take no lift)"
