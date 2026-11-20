"""[U-TGAUGE3] The placement rule (the U-TGAUGE2 verify's M-1, m-1 and m-6): every decode of a layout places each element
at its producer's gauge by its source index, from the one table the deployed program holds.

A twin holds each element of a layout (a gather, a concatenation, a flatten) at the gauge of the producer that emitted
it. The deployed program places every output element by its IR source (``output_sources``: the producer node and its
output index), as ``program_output_gauge`` reads it. A walk that decoded a layout at one producer's scale (the timing
walk read a concatenation of two perceptrons at the first one's scale), or that spread a per-channel gauge over a
flattened axis by a shape guess (a token-wise core's channels-last gauge spread channel-major), compared or computed two
units. The rule: every element at its producer's gauge by its source index, read from the IR source table of the
mapping the program is built from. Every walk reads it (the count family's, the timing family's), the stamped tables
both twins compute host operations and folds from read it, and a read with no gauge table at all refuses by name where
the gauge would differ per element.

The class, swept (one test per site): the timing walk's layout decode (the cascaded decision gate and a host operation
over the layout), the count walk's layout decode (its output, read against the value forward), a host operation over a
flattened per-channel source in both twins, a core folding that source, the source walk's table against the program's,
the wrap slots the per-channel promotion arms over a permuted source, and the flat runner's read of a bare packed
mapping.
"""

from __future__ import annotations

import types

from spiking_points import point_of_mode as _point_of_mode

import numpy as np
import pytest
import torch
import torch.nn as nn

import unit.spiking.host_gather_forms as hg

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.structural import PermuteMapper
from mimarsinan.mapping.mapping_utils import InputMapper, ModelRepresentation, PerceptronMapper
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.support.scale_broadcast import broadcast_scale_along
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
from mimarsinan.pipelining.core import nf_scm_parity
from mimarsinan.spiking.per_channel_theta import promote_per_channel_theta
from mimarsinan.spiking.theta_cotrain import promote_theta_for_exact_qat
from mimarsinan.spiking.segment_forward import SegmentForwardDriver
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge
from mimarsinan.torch_mapping.converter import convert_torch_model

T = hg.T
SEMANTICS = tuple(hg.SEMANTICS)
INPUT_MAX = 2.0
#: A token-wise core's per-channel thresholds (the verify's m-1 form).
TOKEN_THETA = (0.6, 2.4, 1.1, 3.0)


def _od():
    from mimarsinan.chip_simulation.hybrid_run import output_decode

    return output_decode


# ---- the timing family: a layout over two perceptrons at two scales (M-1) --------------------------------------------

class _Branches(nn.Module):
    """One encoding, two branches of two perceptrons each, concatenated: 3 outputs at each branch's scale."""

    def __init__(self, tail: bool):
        super().__init__()
        self.tail = tail
        self.a = nn.Sequential(nn.Linear(10, 12), nn.ReLU())
        self.b = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3), nn.ReLU())
        self.c = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3), nn.ReLU())

    def forward(self, x):
        h = self.a(x)
        out = torch.cat([self.b(h), self.c(h)], dim=1)
        return out * 0.7 + 0.3 if self.tail else out


class _OutsideJoin(nn.Module):
    """One encoding, a branch of two perceptrons joined at the output with a source outside the branch's segment: a
    host op's output (the op ends the encoding's segment), or the raw input."""

    def __init__(self, source: str):
        super().__init__()
        self.source = source
        self.a = nn.Sequential(nn.Linear(10, 12), nn.ReLU())
        self.b = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3), nn.ReLU())

    def forward(self, x):
        h = self.a(x)
        if self.source == "host_op":
            g = h * 1.7 + 0.2
            return torch.cat([self.b(g), g], dim=1)
        return torch.cat([self.b(h), x], dim=1)


def _timing_pair(tail: bool):
    torch.manual_seed(4)
    return _timing_pair_of(_Branches(tail))


def _timing_pair_of(model: nn.Module):
    """The tree's own cascaded timing pair (the converted flow walked by ``TTFSSegmentForward`` and its identity
    executor, each perceptron at its calibrated scale)."""
    import unit.spiking.test_cascaded_boundary_transcoding as cb

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.5, 0.5)
                nn.init.uniform_(m.bias, -0.05, 0.05)
    flow, executor, ir = cb._build_pair(model, (10,), 6, 32, "<=")
    return flow, executor, ir, cb._samples((10,), 64)


class _TimingWalk(nn.Module):
    def __init__(self, flow):
        super().__init__()
        from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward

        self.flow = flow
        self.walk = TTFSSegmentForward(flow.get_mapper_repr(), 32)

    def forward(self, x):
        return self.walk(x)


class _Pipeline:
    config: dict = {"device": "cpu"}
    reporter = None


def test_the_cascaded_gate_reads_a_two_scale_timing_layout_in_one_unit(monkeypatch):
    """The verify's M-1 form: the gate read 0.031 (the walk held the layout at the first branch's scale)."""
    flow, executor, ir, x = _timing_pair(tail=False)
    scales = sorted({round(float(torch.as_tensor(p.activation_scale).mean()), 6) for p in flow.get_perceptrons()})
    assert len(scales) >= 3  # the two branches' output perceptrons sit at two scales
    monkeypatch.setattr(nf_scm_parity, "_build_cascaded_identity_executor", lambda _p, _m, _ir: executor)
    agreement = nf_scm_parity._cascaded_nf_scm_agreement(_Pipeline(), _TimingWalk(flow), ir, x, min_agreement=0.0)
    assert agreement == 1.0


@pytest.mark.parametrize("tail", [False, True], ids=["layout", "host_op_over_layout"])
def test_the_timing_walk_places_each_layout_element_at_its_producer_s_gauge(tail):
    """Element by element: the walk's output against the executor's read decoded at the program's gauge, at the
    two-scale layout and at a host operation over it (which reads the layout's elements at their producers' gauges)."""
    flow, executor, _ir, x = _timing_pair(tail)
    with torch.no_grad():
        walk = _TimingWalk(flow)(x)
        decoded = _od().executor_output_values(executor, executor(x))
    torch.testing.assert_close(walk.double(), decoded.double(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("source", ["host_op", "input"])
def test_the_timing_walk_reads_an_output_element_from_outside_its_segment_as_its_source_holds_it(source):
    """An output layout's element whose producer lies outside the walk's segment (a host op's output, the raw input)
    is read as the deployed program reads it, from its source: the walk equal to the executor's read decoded at the
    program's gauge, element by element. The walk encoded those elements through its boundary at one (the scale of
    no consumer), clipped at one and on its grid: 0.9 off the executor's read (the host op's output), 0.0156 (the
    input)."""
    torch.manual_seed(4)
    flow, executor, _ir, x = _timing_pair_of(_OutsideJoin(source))
    with torch.no_grad():
        walk = _TimingWalk(flow)(x)
        decoded = _od().executor_output_values(executor, executor(x))
    torch.testing.assert_close(walk.double(), decoded.double(), rtol=1e-6, atol=1e-6)


# ---- the count family: a token-wise core's per-channel gauge on a flattened layout (m-1) ------------------------------

class _TokenFlatten(nn.Module):
    """Token-wise encoding and core (per-channel thresholds), flattened tokens-major as the output."""

    shape = (3, 5)

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(5, 7), nn.Linear(7, 4)
        self.ra, self.rb = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.rb(self.b(self.ra(self.a(x)))).flatten(1)


class _TokenFlattenHost(_TokenFlatten):
    """The flattened source read by a host linear head."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(12, 6)

    def forward(self, x):
        return self.head(super().forward(x))


class _TokenFlattenCore(_TokenFlatten):
    """The flattened source read by a core (one threshold): its fold places each input at its source's gauge."""

    def __init__(self):
        super().__init__()
        self.c, self.rc = nn.Linear(12, 6), nn.ReLU()

    def forward(self, x):
        return self.rc(self.c(super().forward(x)))


class _TokenJoinFlatten(nn.Module):
    """Two token-wise cores at one threshold each (two values), joined along their channels, flattened."""

    shape = (3, 5)

    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = nn.Linear(5, 7), nn.Linear(7, 4), nn.Linear(7, 4)
        self.ra, self.rb, self.rc = nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        h = self.ra(self.a(x))
        return torch.cat([self.rb(self.b(h)), self.rc(self.c(h))], dim=2).flatten(1)


class _TokenJoinTokens(_TokenJoinFlatten):
    """Two token-wise cores at two per-channel threshold vectors, joined along the tokens (off the gauge axis),
    flattened."""

    def forward(self, x):
        h = self.ra(self.a(x))
        return torch.cat([self.rb(self.b(h)), self.rc(self.c(h))], dim=1).flatten(1)


class _TokenFanIn(_TokenJoinFlatten):
    """A host add of the tokens-join of two per-channel cores (its gauge per element) and a third per-channel core over
    the same tokens (its gauge per channel), flattened: the add's fan-in reads operands whose gauges differ in length."""

    def __init__(self):
        super().__init__()
        self.d, self.rd = nn.Linear(7, 4), nn.ReLU()

    def forward(self, x):
        h = self.ra(self.a(x))
        joined = torch.cat([self.rb(self.b(h)), self.rc(self.c(h))], dim=1)
        return (joined + self.rd(self.d(torch.cat([h, h], dim=1)))).flatten(1)


class _TokenPermuteFlatten(_TokenFlatten):
    """The token-wise core's per-channel output with its axes swapped (channels first), flattened."""

    def forward(self, x):
        return self.rb(self.b(self.ra(self.a(x)))).permute(0, 2, 1).flatten(1)


class _TokenJoinTokens3d(_TokenJoinFlatten):
    """Two token-wise cores at two per-channel threshold vectors, joined along the tokens, kept with its tokens and
    channels as the output (rank three)."""

    def forward(self, x):
        h = self.ra(self.a(x))
        return torch.cat([self.rb(self.b(h)), self.rc(self.c(h))], dim=1)


class _TokenPermute3d(_TokenFlatten):
    """The token-wise core's per-channel output with its axes swapped (channels first), kept as the output (rank
    three)."""

    def forward(self, x):
        return self.rb(self.b(self.ra(self.a(x)))).permute(0, 2, 1)


class _TokenPermuteMean(_TokenFlatten):
    """The token-wise core's per-channel output with its axes swapped (channels first), averaged over the tokens by a
    host mean and read by a host linear head: a layout's gauge per element read by a host op, whose own gauge its
    output layout then holds."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 6)

    def forward(self, x):
        return self.head(self.rb(self.b(self.ra(self.a(x)))).permute(0, 2, 1).mean(dim=2))


TOKEN_FORMS = {
    "out": (_TokenFlatten, (INPUT_MAX, TOKEN_THETA), 1.0),
    "host": (_TokenFlattenHost, (INPUT_MAX, TOKEN_THETA), 1.0),
    # weights scaled so the core does not saturate (at full scale every output sits at its threshold, which hides
    # any fold)
    "core": (_TokenFlattenCore, (INPUT_MAX, TOKEN_THETA, 1.3), 0.25),
    "join": (_TokenJoinFlatten, (INPUT_MAX, 1.1, 2.9), 0.5),
    "tokens": (_TokenJoinTokens, (INPUT_MAX, TOKEN_THETA, (2.2, 0.7, 1.6, 0.9)), 0.5),
    "permute": (_TokenPermuteFlatten, (INPUT_MAX, TOKEN_THETA), 1.0),
    "permute_mean": (_TokenPermuteMean, (INPUT_MAX, TOKEN_THETA), 1.0),
    "tokens_3d": (_TokenJoinTokens3d, (INPUT_MAX, TOKEN_THETA, (2.2, 0.7, 1.6, 0.9)), 0.5),
    "permute_3d": (_TokenPermute3d, (INPUT_MAX, TOKEN_THETA), 1.0),
    "fan_in": (_TokenFanIn, (INPUT_MAX, TOKEN_THETA, (2.2, 0.7, 1.6, 0.9), (1.3, 0.5, 2.6, 1.9)), 0.5),
}


def _token_program(form: str, semantics: str) -> hg.Program:
    """The verify's construction (its case builder's seeded weights), LIF nodes at the form's thresholds, through the
    Soft Core Mapping sequence with the wire gauge established by the seam."""
    cls, thetas, weight_scale = TOKEN_FORMS[form]
    torch.manual_seed(0)
    net = cls()
    gen = torch.Generator().manual_seed(100)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Linear):
                m.weight.copy_((torch.rand(m.weight.shape, generator=gen) * 0.8 - 0.2) * weight_scale)
                m.bias.copy_(0.05 + 0.3 * torch.rand(m.bias.shape, generator=gen))
    net.eval()
    with torch.no_grad():
        n_out = int(net(torch.zeros(2, *cls.shape)).shape[1])
    flow = convert_torch_model(net, net.shape, n_out, device="cpu", encoding_layer_placement="subsume").eval()
    perceptrons = list(flow.get_perceptrons())
    assert len(perceptrons) == len(thetas)
    for p, theta in zip(perceptrons, thetas):
        p.set_activation_scale(torch.as_tensor(theta, dtype=torch.float32).clone())
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    repr_, ir, hybrid = hg.scm_sequence(
        flow, perceptrons, wq=False, retimed=hg.SEMANTICS[semantics][0],
        establish=lambda f: establish_deployed_wire_gauge(f, input_data_scale=INPUT_MAX))
    th = [float(torch.as_tensor(p.activation_scale).max()) for p in perceptrons]
    return hg.Program(form, cls.shape, flow, repr_, ir, hybrid, perceptrons, th, False)


def _walk(program, semantics, x):
    with torch.no_grad(), measurement_plane():
        return SegmentForwardDriver(program.repr_, T, hg.SEMANTICS[semantics][2]())(x)


def _value(program, x):
    with torch.no_grad():
        return type(program.flow).forward(program.flow, x)


def _decoded(program, semantics, x):
    flow = hg.deployed_flow(program, semantics)
    with torch.no_grad():
        return _od().executor_output_values(flow, flow(x))


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_count_walk_reads_a_token_wise_per_channel_layout_as_the_value_forward(semantics):
    """The verify's m-1 form: 0 of 32 classes against the value forward (the walk spread the per-channel gauge
    channel-major over a tokens-major flatten)."""
    program = _token_program("out", semantics)
    x = hg.samples(program, 32, 0)
    walk, value = _walk(program, semantics, x), _value(program, x)
    assert int((walk.argmax(1) == value.argmax(1)).sum()) == 32
    # each element within its producer's half step (the window's grid) of the value forward
    gauge = torch.tensor(TOKEN_THETA, dtype=torch.float64).repeat(3)
    assert bool(((walk.double() - value.double()).abs() <= gauge / T + 1e-6).all())


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_decoded_deployed_read_equals_the_count_walk_element_by_element(semantics):
    program = _token_program("out", semantics)
    x = hg.samples(program, 32, 0)
    torch.testing.assert_close(_decoded(program, semantics, x).double(), _walk(program, semantics, x).double(),
                               rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_walk_s_gauge_table_is_the_program_s_element_by_element(semantics):
    """One table: the gauge the walk decodes its output layout at is the program's output gauge (by source index)."""
    program = _token_program("out", semantics)
    like = torch.zeros(1, 12)
    np.testing.assert_allclose(hg._nf_output_gauge(program, like)[0].numpy(),
                               _od().program_output_gauge(program.hybrid), rtol=1e-6)
    np.testing.assert_allclose(_od().program_output_gauge(program.hybrid), np.tile(TOKEN_THETA, 3), rtol=1e-6)


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_a_host_operation_over_the_flattened_source_reads_the_value_forward_in_both_twins(semantics):
    """The verify's m-1 host case: both twins decoded the flattened source channel-major alike (0 of 32 classes
    against the value forward, the twins equal)."""
    program = _token_program("host", semantics)
    x = hg.samples(program, 32, 0)
    walk, value, decoded = _walk(program, semantics, x), _value(program, x), _decoded(program, semantics, x)
    assert int((walk.argmax(1) == value.argmax(1)).sum()) == 32
    torch.testing.assert_close(decoded.double(), walk.double(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_a_core_over_the_flattened_source_folds_each_input_at_its_source_s_gauge(semantics):
    """The deployed core's fold spread the source's per-channel gauge channel-major: its window counts differed from
    the walk's (93 of 192), the walk equal to the value forward."""
    program = _token_program("core", semantics)
    x = hg.samples(program, 32, 0)
    nf_out, nf_counts = _nf_window_counts(program, x, semantics)
    scm_out, scm_counts = hg.scm_window_counts(program, x, semantics)
    assert hg.count_mismatches(nf_counts, scm_counts) == {pi: 0 for pi in sorted(scm_counts)}
    torch.testing.assert_close(_od().executor_output_values(hg.deployed_flow(program, semantics), scm_out).double(),
                               nf_out.double(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("form", ["join", "tokens", "permute", "fan_in"],
                         ids=["join_along_channels_then_flatten", "join_off_the_gauge_axis", "axis_move_then_flatten",
                              "host_op_fan_in_of_a_layout_and_a_core"])
def test_every_layout_over_differing_gauges_reads_each_element_at_its_source_s(form, semantics):
    """A join along the channels of two cores at two thresholds, then flattened (each element's channel cycles fastest:
    the walk read 3 of 32 classes); a join along the tokens of two per-channel cores (off the gauge axis: the joined
    vector is no per-channel gauge, its elements are read from the IR table); an axis move of a per-channel gauge, then
    flattened (read from the IR table). The walk against the value forward, the decoded deployed read against the
    walk; a host op whose operands' gauges differ in length (a layout's per element, a core's per channel) reads each
    operand's element at its producer's gauge (the fan-in spread the per-channel one channel-major)."""
    program = _token_program(form, semantics)
    x = hg.samples(program, 32, 0)
    walk, value = _walk(program, semantics, x).double(), _value(program, x).double()
    assert int((walk.argmax(1) == value.argmax(1)).sum()) >= 30
    if form != "fan_in":  # a layout output: each element within its producer's grid step of the value forward
        gauge = torch.as_tensor(_od().program_output_gauge(program.hybrid))
        assert bool(((walk - value).abs() <= gauge / T + 1e-6).all())
    else:  # a host add: within the sum of its two operands' grid steps (each operand's largest gauge)
        assert float((walk - value).abs().max()) <= (3.0 + 2.6) / T + 1e-6
    torch.testing.assert_close(_decoded(program, semantics, x).double(), walk, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("form", ["tokens_3d", "permute_3d"], ids=["join_off_the_gauge_axis", "axis_move"])
def test_a_rank_three_output_layout_reads_each_element_at_its_source_s_gauge(form, semantics):
    """An output layout kept with its tokens and channels (no flatten), whose gauge is one per element: the walk reads
    each element of the rank-three value at its producer's gauge (within its producer's grid step of the value
    forward) and the decoded deployed read equals it. A per-element gauge not read over the sample's elements left the
    walk at the boundary table's one value: 0.96 (the join) and 1.34 (the axis move) off the value forward."""
    program = _token_program(form, semantics)
    x = hg.samples(program, 32, 0)
    walk, value = _walk(program, semantics, x).double(), _value(program, x).double()
    assert walk.dim() == 3
    gauge = torch.as_tensor(_od().program_output_gauge(program.hybrid)).view(1, *walk.shape[1:])
    assert bool(((walk - value).abs() <= gauge / T + 1e-6).all())
    torch.testing.assert_close(_decoded(program, semantics, x).double().reshape(walk.shape), walk, rtol=1e-6,
                               atol=1e-6)


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_a_host_reduction_over_a_permuted_per_channel_source_reads_the_value_forward_in_both_twins(semantics):
    """The walk decodes the permuted source (a gauge per element, rank 3) at each element's producer's gauge for the
    host mean, and reads the mean's output layout at the gauge the mean emits, as the deployed program does: the walk
    within a grid step of the value forward, the decoded deployed read equal to the walk. The per-channel gauge spread
    along the permuted source's tokens axis left both twins 0.85 off the value forward (30 of 32 classes); the mean's
    emitted gauge, not held along a channel axis of its output, was refused by the placement before it read the
    wrapper's broadcast."""
    program = _token_program("permute_mean", semantics)
    x = hg.samples(program, 32, 0)
    walk, value = _walk(program, semantics, x).double(), _value(program, x).double()
    assert int((walk.argmax(1) == value.argmax(1)).sum()) == 32
    # the head's input within the core's largest grid step of the value mean, through the head's weights
    bound = max(TOKEN_THETA) / T * float(program.flow.get_mapper_repr()._exec_order[-1].module.weight.abs().sum(1).max())
    assert float((walk - value).abs().max()) <= bound + 1e-6
    torch.testing.assert_close(_decoded(program, semantics, x).double(), walk, rtol=1e-6, atol=1e-6)


# ---- the per-channel promotion's wrap slots: a host reduction over a permuted per-channel source ----------------------

class _GraphFlow(PerceptronFlow):
    """A flow around a hand-built mapper graph."""

    def __init__(self, mapper_repr):
        super().__init__("cpu")
        self.input_activation = nn.Identity()
        self._mapper_repr = mapper_repr

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def get_input_activation(self):
        return self.input_activation

    def set_input_activation(self, activation):
        self.input_activation = activation

    def forward(self, x):
        return self._mapper_repr(x)


def _wire_to_value(node, wire):
    wrapper = node._maybe_wrap_for_scales()
    scale = torch.as_tensor(node.output_scale)
    return wire * broadcast_scale_along(scale, wire, getattr(wrapper, "channel_axis", None))


def _promote_by_quantiles(flow, theta):
    report = promote_per_channel_theta(flow, [theta], [1.0])
    return report.promoted


def _promote_for_training(flow, theta):
    flow.get_perceptrons()[0].set_activation_scale(torch.tensor(theta))
    return tuple(promote_theta_for_exact_qat(flow)["per_channel"])


@pytest.mark.parametrize("promote", [_promote_by_quantiles, _promote_for_training],
                         ids=["calibration_promotion", "trainable_theta_promotion"])
def test_the_per_channel_promotion_arms_a_permuted_source_s_slots_at_each_element_s_gauge(promote):
    """A token-wise core promoted to per-channel thresholds (by its quantiles at calibration, or to a trainable theta),
    its output permuted (channels first), reduced over the tokens by a host mean and read by a terminal host linear
    head (an eligible promotion). The slots each promotion arms decode the permuted source's rates at each element's
    producer's gauge: the wire domain's mean and head equal the value forward's (the slots armed without the
    placement spread the per-channel gauge along the tokens axis, the mean read 3.17 off at most, 30 of 30 elements
    wrong)."""
    tokens, channels = 3, 6
    torch.manual_seed(0)
    producer = PerceptronMapper(InputMapper((tokens, channels)), Perceptron(channels, channels, name="producer"))
    mean = ComputeOpMapper(PermuteMapper(producer, (0, 2, 1)), ComputeAdapter(torch.mean, kwargs={"dim": 2}),
                           input_shapes=[(channels, tokens)], name="mean")
    head = ComputeOpMapper(mean, nn.Linear(channels, 2), name="head")
    flow = _GraphFlow(ModelRepresentation(head)).eval()
    with torch.no_grad():
        flow(torch.randn(2, tokens, channels))
    theta = [0.5 + 0.4 * c for c in range(channels)]
    assert promote(flow, theta) == ("producer",)
    values = torch.rand(5, channels, tokens, generator=torch.Generator().manual_seed(1)) * 2.0
    rates = values / torch.tensor(theta).view(1, channels, 1)
    with torch.no_grad():
        mean_wire = mean.forward_scale_normalized(rates)
        head_wire = head.forward_scale_normalized(mean_wire)
        mean_value = values.mean(dim=2)
        head_value = head.module(mean_value)
    torch.testing.assert_close(_wire_to_value(mean, mean_wire), mean_value, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(_wire_to_value(head, head_wire), head_value, rtol=1e-5, atol=1e-5)


def test_the_layout_walk_reads_each_ir_node_s_outputs_at_their_positions_in_their_producer(monkeypatch):
    """An IR node may hold any subset of its producer's value (a core per token, per position, per slice of its
    neurons): the walk reads each of its outputs at that output's position in the producer's sample, not at its own
    output index (the same gauge only where the producer's gauge repeats over the positions the node holds)."""
    import mimarsinan.mapping.map_model_to_ir as mmi

    producer, layout = types.SimpleNamespace(name="producer"), types.SimpleNamespace(name="layout")
    graph = types.SimpleNamespace(execution_order=lambda: [producer, layout])
    # IR node 7 holds the producer's odd elements; the layout reads node 7's three outputs in reverse
    table = mmi.SourceTable(sources={id(layout): (np.array([7, 7, 7]), np.array([2, 1, 0]))},
                            shapes={id(producer): (6,)}, owners={7: (id(producer), np.array([1, 3, 5]))})
    monkeypatch.setattr(mmi, "mapper_source_table", lambda _graph: table)
    placed = mmi.walk_layout_gauge(graph, layout, lambda _p, shape: np.arange(10.0, 10.0 + shape[0]))
    np.testing.assert_array_equal(placed, [15.0, 13.0, 11.0])


def test_a_residual_lane_join_keeps_its_parts_gauges_where_they_read_right(monkeypatch):
    """The residual merge's lane join (a residual add's two operands, one shape, flattened in order): parts at one
    value each, or with their channels leading, keep the joined gauges (read in contiguous blocks, no table read);
    channels-last parts at per-channel gauges are placed from the IR table."""
    import mimarsinan.mapping.map_model_to_ir as mmi
    from mimarsinan.mapping.mappers.structural import InputMapper
    from mimarsinan.mapping.model_representation import ModelRepresentation
    from mimarsinan.mapping.support.residual_merge import _ResidualConcatMapper

    a, b = InputMapper((4,)), InputMapper((4,))  # two producers of one shape (the operands' stand-ins)
    join = _ResidualConcatMapper([a, b])
    placement = mmi.LayoutGaugePlacement(ModelRepresentation(join))
    read: list = []
    monkeypatch.setattr(mmi, "walk_layout_gauge", lambda *args: read.append(args) or np.arange(8.0))
    one = {a: torch.full((4,), 1.5), b: torch.full((4,), 2.5)}
    joined = torch.cat([one[a], one[b]])
    assert placement(join, joined, one) is joined and not read
    monkeypatch.setattr(mmi, "gauge_axis", lambda _node: mmi.LEADING)
    vectors = {a: torch.tensor([0.5, 1.0, 1.5, 2.0]), b: torch.tensor([2.0, 0.5, 1.5, 1.0])}
    joined = torch.cat([vectors[a], vectors[b]])
    assert placement(join, joined, vectors) is joined and not read
    monkeypatch.setattr(mmi, "gauge_axis", lambda _node: -1)  # channels last
    placed = placement(join, joined, vectors)
    assert read and torch.equal(placed, torch.arange(8.0, dtype=joined.dtype))


def _nf_window_counts(program, x, semantics):
    """The walk's window counts per perceptron, each record divided by its threshold along its channel axis (the last:
    a per-channel threshold is a linear perceptron's)."""
    rec: dict = {}
    with torch.no_grad(), measurement_plane():
        out = SegmentForwardDriver(program.repr_, T, hg.SEMANTICS[semantics][2]())(x, node_value_recorder=rec)
    counts = {}
    for i, p in enumerate(program.perceptrons):
        if id(p) in rec:
            th = torch.as_tensor(p.activation_scale, dtype=rec[id(p)].dtype).reshape(-1)
            v = rec[id(p)] / (th if th.numel() > 1 else float(th))
            counts[i] = torch.round(v.reshape(x.shape[0], -1) * T).double().numpy()
    return out, counts


# ---- the flat runner: a bare packed mapping holds no gauge table (m-6) ------------------------------------------------

def _bare_mapping(thresholds):
    """A bare packed mapping (no program gauge table): one output core whose neurons hold ``thresholds``."""
    from mimarsinan.code_generation.cpp_chip_model_types import SpikeSource

    core = types.SimpleNamespace(activation_scale=torch.tensor(thresholds, dtype=torch.float32),
                                 neurons_per_core=len(thresholds))
    sources = np.array([SpikeSource(0, i) for i in range(len(thresholds))], dtype=object)
    return types.SimpleNamespace(cores=[core], output_sources=sources)


def _flat_runner(mapping, counts, monkeypatch):
    """The flat runner's read with its simulator stubbed: the driver returns ``counts`` and ranks them at the gauge
    the runner hands it (the driver's own ranking)."""
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.chip_simulation.simulation_runner import flat

    counts = np.asarray(counts, dtype=np.float64)
    seen: dict = {}

    class _Driver:
        def __init__(self, *a, **k):
            pass

        def predict_spiking(self, _data, _steps, _delay, output_gauge=None):
            seen["gauge"] = output_gauge
            return NevresimDriver._simulator_output_to_predictions(None, list(counts.reshape(-1)), counts.shape[1],
                                                                   output_gauge)

    monkeypatch.setattr(flat, "NevresimDriver", _Driver)
    monkeypatch.setattr(flat, "ChipLatency", lambda _m: types.SimpleNamespace(calculate=lambda: 0))
    monkeypatch.setattr(flat, "flat_membrane_slices", lambda *a, **k: [])
    runner = object.__new__(flat.SimulationFlatMixin)
    attrs = dict(mapping=mapping, simulation_length=T, input_size=1, working_directory=".", weight_type=float,
                 spike_generation_mode="Uniform", firing_mode="Default", thresholding_mode="<", spiking_mode="lif", spiking_point=_point_of_mode("lif"),
                 threshold_type=float, nevresim_connectivity_mode=None, simulation_step_timeout_s=None,
                 soma_law=None, membrane_readout=False, test_data=[(None, None)] * counts.shape[0])
    for key, value in attrs.items():
        setattr(runner, key, value)
    runner._evaluate_chip_output = lambda predictions: seen.setdefault("predictions", list(predictions)) and 0.0
    return runner, seen


def test_a_bare_packed_mapping_refuses_an_output_whose_gauge_differs_per_element(monkeypatch):
    """Its counts rank as its values only at one gauge: at per-neuron thresholds the read refuses by name."""
    mapping = _bare_mapping([0.5, 1.0, 4.0])
    runner, _seen = _flat_runner(mapping, [[10, 6, 2]], monkeypatch)
    with pytest.raises(_od().OutputGaugeError, match="no gauge table"):
        runner._run_flat_mapping(mapping)


def test_a_bare_packed_mapping_at_one_gauge_ranks_its_counts(monkeypatch):
    mapping = _bare_mapping([1.5, 1.5, 1.5])
    runner, seen = _flat_runner(mapping, [[10, 6, 2], [1, 7, 3]], monkeypatch)
    runner._run_flat_mapping(mapping)
    assert seen["predictions"] == [0, 1]


def test_the_flat_runner_hands_the_driver_its_program_s_gauge(monkeypatch):
    """A program's classes are its decoded outputs' (the verify's VM15: a runner that hands the driver no gauge ranks
    raw counts)."""
    program = _token_program("out", "streamed_lif")
    gauge = _od().program_output_gauge(program.hybrid)
    counts = np.zeros((2, 12))
    counts[0, [0, 3]] = (T, 0.5 * T)  # raw counts rank output 0 first, the decoded values output 3 (gauge 3.0 vs 0.6)
    counts[1, [1, 2]] = (T, T)
    runner, seen = _flat_runner(program.hybrid, counts, monkeypatch)
    runner._run_flat_mapping(program.hybrid)
    np.testing.assert_allclose(seen["gauge"], gauge)
    assert seen["predictions"] == [3, 1]
