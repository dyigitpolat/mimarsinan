"""[U-SC2 A-1, D-USC-1] Per-channel gauges carried along the channel axis through channel-changing host ops and joins.

The scale walk carries one gauge per CHANNEL of a node's output: a perceptron's per-output-channel theta, the input's
unit gauge, and, through a host op, its sources' gauges. On U-SC's tree a host op passed its source's vector through
unchanged even when it changed the channel count (DenseNet-BC's stem convolution, 3 to 2k channels, kept the input's
3-vector; each dense unit's host convolution kept its core's 4k-vector), so a concatenation's vector was misaligned with
its channels (``cat_1``: 19 entries for 12 channels) and the ``ScaleNormalizingWrapper`` of its first reader broadcast
it along the LAST axis, mean-filling it: every channel was decoded at one gauge (S1's cached RC4A model: 1.78 to 1.89
against the stem channels' 1.0, ``final_units/usc/probes/out/concat_decode_s1.json``). The deployed fold of a core
reading such a concatenation was mean-collapsed the same way, so the twins agreed with each other and both computed a
function the model does not.

Now every node's vector is as long as its output's channel count (a host op re-spreads its uniform gauge over its own
output channels; a channel-changing armed op emits one gauge; a concatenation along the channel axis concatenates the
aligned parts), and the wrapper decodes a per-channel vector along the channel axis. The deployed fold of a core reading
the concatenation reads the same aligned vector, so the twins stay in agreement (``test_mixed_currency_gather``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.scale_propagation import perceptron_source_out_scale, walk_out_scales
from mimarsinan.mapping.mappers.structural import ConcatMapper, InputMapper
from mimarsinan.spiking.segment_partition import perceptron_of
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge

from unit.spiking.host_gather_forms import T, lif_net

#: The forms with a concatenation: DenseNet-BC's mixed concatenation read by a core, by a host op, nested, and a
#: DenseNet-BC program (two dense blocks, a transition, published order).
FORMS = ("absolute_concat", "absolute_concat_host", "nested_gather_host_read", "nested_gather_core_read", "densenet_bc")
_RTOL = 1e-6


def _established(flow, input_data_scale: float = 1.0):
    establish_deployed_wire_gauge(flow, input_data_scale=input_data_scale)
    return flow.get_mapper_repr()


def _output_shapes(repr_, shape) -> dict:
    """node -> its per-sample output shape, from one plain walk of the mapper graph (batch 2)."""
    order, deps, _ = repr_._ensure_exec_graph()
    torch.manual_seed(11)
    x = torch.rand(2, *shape)
    values: dict = {}
    shapes: dict = {}
    with torch.no_grad():
        for node in order:
            d = deps.get(node, [])
            if not d:
                v = node.forward(x)
            elif len(d) == 1:
                v = node.forward(values[d[0]])
            else:
                v = node.forward(tuple(values[k] for k in d))
            values[node] = v
            shapes[node] = tuple(int(s) for s in v.shape[1:])
    return shapes


def _channels(shape: tuple) -> int:
    """The channel count of a per-sample output: its leading axis (every node of these channel-first or flat forms)."""
    return int(shape[0])


def _gauge_vectors(repr_) -> dict:
    """The scale walk's per-node vectors (the walk Weight Quantization and Soft Core Mapping run; idempotent after the
    seam)."""
    return walk_out_scales(repr_, lambda node, deps, out: node.propagate_source_scale(deps, out))


class _Ones(nn.Module):
    """A stand-in module that emits ones of a fixed per-sample shape whatever it reads."""

    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, *inputs, **kwargs):
        return torch.ones(1, *self.shape, dtype=torch.float32)


def _true_gauge_map(node, deps: dict, shapes: dict) -> torch.Tensor:
    """The gauge each element of ``node``'s output is held at (its per-sample shape), from the producers: a perceptron
    emits its theta per output channel, an armed host op what its own wrapper emits (its output normalization), the
    input and an absolute host chain rooted at it unit gauge, an unarmed host op its source's (uniform) gauge over its own
    output, and a layout node (a concatenation, a reshape) moves its sources' elements, so its map is its own layout
    applied to theirs."""
    shape = shapes[node]
    p = perceptron_of(node)
    if p is not None:
        theta = perceptron_source_out_scale(p).double().reshape(-1)
        return theta.reshape(-1, *([1] * (len(shape) - 1))).expand(*shape).clone()
    if isinstance(node, InputMapper):
        return torch.ones(shape, dtype=torch.float64)
    sources = deps.get(node, [])
    if isinstance(node, ComputeOpMapper) and node.output_scale is not None:
        wrapper = node._maybe_wrap_for_scales(_Ones(shape))
        with torch.no_grad():
            emitted = wrapper(*[torch.ones(1, *shapes[d]) for d in sources]).double()[0]
        return 1.0 / emitted
    below = [_true_gauge_map(d, deps, shapes) for d in sources]
    if getattr(node, "moves_elements_only", False):
        with torch.no_grad():
            moved = node.forward(below[0].unsqueeze(0) if len(below) == 1 else tuple(b.unsqueeze(0) for b in below))
        return moved.double()[0]
    assert len(below) == 1, f"fixture: {type(node).__name__} {getattr(node, 'name', None)!r} has one source"
    if tuple(below[0].shape) == tuple(shape):
        return below[0]
    assert torch.all(below[0] == below[0].reshape(-1)[0]), "an unarmed shape-changing op only ever passes a uniform gauge"
    return below[0].reshape(-1)[:1].expand(*shape).clone()


def _wrapper_decode(reader, in_shape: tuple) -> torch.Tensor:
    """The per-element gauge the reader's deployed ScaleNormalizingWrapper multiplies its input by: the reader's own
    wrapper around an identity module, read on ones with the output normalization at one."""
    wrapper = reader._maybe_wrap_for_scales(nn.Identity())
    wrapper.output_scale = torch.ones(1)
    with torch.no_grad():
        return wrapper(torch.ones(1, *in_shape, dtype=torch.float32)).double()[0]


def _rel_error(applied: torch.Tensor, true: torch.Tensor) -> float:
    return float(((applied - true) / true).abs().max())


def _decode_errors(flow, shape) -> list:
    """``(concatenation, reader, max relative decode error)`` for every reader of every concatenation: a host reader's
    wrapper decode per element against the element's producer gauge, a core reader's fold (``per_input_scales``, its
    input axis) against the same, per element of a flat read and per channel of a patch read."""
    repr_ = flow.get_mapper_repr()
    order, deps, _ = repr_._ensure_exec_graph()
    shapes = _output_shapes(repr_, shape)
    consumers: dict = {}
    for node in order:
        for d in deps.get(node, []):
            consumers.setdefault(d, []).append(node)
    def readers(node, gauge):
        """The first consumers past layout nodes, each with the gauge map it reads (the layout applied to ``gauge``)."""
        for c in consumers.get(node, []):
            if getattr(c, "moves_elements_only", False) and len(deps.get(c, [])) == 1:
                with torch.no_grad():
                    yield from readers(c, c.forward(gauge.unsqueeze(0)).double()[0])
            else:
                yield c, gauge

    rows = []
    for cat in (n for n in order if isinstance(n, ConcatMapper)):
        for reader, true in readers(cat, _true_gauge_map(cat, deps, shapes)):
            p = perceptron_of(reader)
            if p is not None:
                assert p.per_input_scales is not None
                fold = p.per_input_scales.double().reshape(-1)
                if fold.numel() == true.numel():
                    err = _rel_error(fold, true.reshape(-1))
                else:
                    per_channel = true.reshape(true.shape[0], -1)
                    assert torch.all(per_channel == per_channel[:, :1]), "a patch read needs a per-channel gauge"
                    err = _rel_error(fold.reshape(true.shape[0], -1), per_channel[:, :1])
            elif isinstance(reader, ComputeOpMapper) and reader.output_scale is not None:
                err = _rel_error(_wrapper_decode(reader, tuple(true.shape)), true)
            else:
                continue
            rows.append((getattr(cat, "_name", None), getattr(reader, "name", None) or type(reader).__name__, err))
    return rows


# ---------------------------------------------------------------------------
# One gauge per output channel at every node
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_every_node_carries_one_gauge_per_output_channel(form):
    """U-SC: the stem keeps the input's 3-vector over its 8 channels, a dense unit's host convolution its core's
    16-vector over 4 channels, ``cat_1`` 19 entries for 12 channels."""
    shape, flow, _, _ = lif_net(form, 0, 1.7)
    repr_ = _established(flow)
    shapes = _output_shapes(repr_, shape)
    vectors = _gauge_vectors(repr_)
    wrong = [(type(n).__name__, getattr(n, "name", None) or getattr(n, "_name", None), int(v.numel()), _channels(shapes[n]))
             for n, v in vectors.items() if v is not None and int(v.reshape(-1).numel()) != _channels(shapes[n])]
    assert wrong == []


# ---------------------------------------------------------------------------
# Every concatenation's readers decode each channel at its producer's gauge (the decode-error probe)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_every_concatenation_reader_decodes_each_channel_at_its_producer_gauge(form):
    """The decode-error probe of D-USC-1 on the fixtures: 0 error at every reader of every concatenation (U-SC: one mean
    gauge for every channel, the stem channels' 1.0 read at the concatenation's mean)."""
    shape, flow, _, _ = lif_net(form, 0, 1.7)
    _established(flow)
    rows = _decode_errors(flow, shape)
    assert rows, "fixture: the form has a concatenation read by a core or an armed host op"
    assert [r for r in rows if r[2] > _RTOL] == []


def test_the_densenet_bc_block_decodes_every_concatenation_at_a_non_unit_input_scale():
    """The same probe with the boundary walk's input scale away from one (S1's sessions run 2.0 and above): the source
    walk's gauges, which the deployed decode reads, do not depend on it."""
    shape, flow, _, _ = lif_net("densenet_bc", 1, 2.2)
    _established(flow, input_data_scale=2.0)
    rows = _decode_errors(flow, shape)
    assert rows and [r for r in rows if r[2] > _RTOL] == []


def test_a_channel_changing_host_op_with_a_per_channel_input_emits_one_gauge_over_its_own_channels():
    """DenseNet-BC's transition convolution reads a per-channel gauge (the stem's unit gauge beside the units' thetas)
    and changes the channel count: it decodes per input channel and emits one gauge, spread over its output channels."""
    shape, flow, _, _ = lif_net("densenet_bc", 0, 1.7)
    repr_ = _established(flow)
    shapes = _output_shapes(repr_, shape)
    vectors = _gauge_vectors(repr_)
    conv = next(n for n in repr_.execution_order() if isinstance(n, ComputeOpMapper) and n.name == "trans1_conv")
    assert conv.per_source_scales is not None and conv.output_scale is not None
    (src,) = conv.per_source_scales
    assert src.numel() == _channels(shapes[repr_._deps[conv][0]])
    assert not torch.all(src == src[0]), "fixture: the transition reads a per-channel gauge"
    out = vectors[conv].reshape(-1)
    assert out.numel() == _channels(shapes[conv]) and torch.all(out == out[0])
    assert float(torch.as_tensor(conv.output_scale).reshape(-1)[0]) == pytest.approx(float(out[0]), rel=1e-6)


# ---------------------------------------------------------------------------
# NF and SCM count-exact at the input scale the CIFAR members run (S1 and S4: 2.02)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("semantics", ("windowed_lif", "streamed_lif"))
@pytest.mark.parametrize("form", ("absolute_concat", "absolute_concat_host", "densenet_bc"))
def test_nf_and_scm_are_count_exact_at_a_non_unit_input_scale(form, semantics):
    """An input-rooted absolute chain read by a core (DenseNet-BC's first core reads relu1 of bn1 of the stem; the
    absolute concatenation's core reads the stem) held raw values at unit gauge in the deployed program (the IR's
    divisor kappa_fold / kappa_buf was 1 and the core's fold the input root's unit gauge) while the boundary table, the
    core's entry currency and the NF walk read it at the input scale. At 2.0 the twins disagreed (densenet_bc: 2,654 of
    43,008 counts at the first core; absolute_concat: 18 of 96, final_units/usc2/probes/out/ids_parity_new.txt). The
    seam now arms such an op at its boundary-table gauge, so the fold and the divisor follow the one gauge."""
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales

    from unit.spiking.host_gather_forms import build, compare

    def establish(flow):
        establish_deployed_wire_gauge(flow, input_data_scale=2.0)
        compute_per_source_scales(flow.get_mapper_repr(), arm_wire_value_ops=True)
        propagate_boundary_input_scales(flow, input_data_scale=2.0)

    program = build(form, 0, 1.6, semantics, False, establish=establish)
    mism, total, out_diff = compare(program, semantics, n=16)
    assert total > 0
    assert mism == {pi: 0 for pi in mism}, (form, mism, total)
    assert out_diff < 1e-3, out_diff


# ---------------------------------------------------------------------------
# The route (b) member: S1 (DenseNet-BC-100, k 12, published order) at random init
# ---------------------------------------------------------------------------


def _s1_member():
    from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.torch_mapping.converter import convert_torch_model

    shape, classes = (3, 32, 32), 100
    builder = BUILDERS_REGISTRY["cifar_densenet_bc"]("cpu", shape, classes, {})
    torch.manual_seed(0)
    model = build_model(builder, {"depth": 100, "growth_rate": 12, "compression": 0.5, "layer_order": "published"},
                        encoding_placement="subsume").eval()
    with torch.no_grad():
        model(torch.zeros(1, *shape))
    flow = model if hasattr(model, "get_mapper_repr") else convert_torch_model(
        model, shape, classes, device="cpu", encoding_layer_placement="subsume")
    for i, p in enumerate(flow.get_perceptrons()):
        p.set_activation_scale(torch.tensor(1.0 + 0.37 * (i % 7 + 1)))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    return flow, shape


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_the_s1_member_carries_aligned_gauges_and_decodes_every_concatenation_exactly():
    """S1 at random init: one gauge per output channel at every node, and 0 decode error at every reader of its 48
    concatenations (S1's cached RC4A model read 78 to 89 percent on the stem channels)."""
    flow, shape = _s1_member()
    repr_ = _established(flow, input_data_scale=2.0)
    shapes = _output_shapes(repr_, shape)
    vectors = _gauge_vectors(repr_)
    wrong = [getattr(n, "name", None) or getattr(n, "_name", None) for n, v in vectors.items()
             if v is not None and int(v.reshape(-1).numel()) != _channels(shapes[n])]
    assert wrong == []
    rows = _decode_errors(flow, shape)
    assert len({r[0] for r in rows}) >= 48
    assert [r for r in rows if r[2] > _RTOL] == []
