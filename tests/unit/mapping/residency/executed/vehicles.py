"""Vehicles of the executed residency plan: a seeded multi-level LIF chain, built on both planes."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import ConcatMapper, InputMapper, ReshapeMapper, SubscriptMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.noc.wire_census import census_of_walk
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.residency_rules import ProgramGraph
from mimarsinan.mapping.support.schedule.residency_composition import ShapeOnlyResidency, compose_shape_only_program
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8
IN = 12
#: Output-tiled at 8 neurons per core, so the units span 3, 3, 2, 2 and 1 tiles: a multi-level chain.
WIDTHS = (24, 24, 16, 16, 6)
MAX_AXONS, MAX_NEURONS = 25, 8


def _lif(width: int, fan_in: int) -> Perceptron:
    p = Perceptron(width, fan_in, normalization=nn.Identity())
    p.base_activation = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    p.activation = p.base_activation
    with torch.no_grad():
        p.layer.weight.copy_((torch.rand(width, fan_in) - 0.5) * 2.0)
        if p.layer.bias is not None:
            p.layer.bias.fill_(0.3)
    return p


def chain_repr(seed: int = 0, widths: Sequence[int] = WIDTHS) -> ModelRepresentation:
    """Encode, then one LIF hop per width, seeded weights; perceptron indices assigned."""
    torch.manual_seed(seed)
    enc = _lif(IN, IN)
    enc.is_encoding_layer = True
    enc.use_cycle_accurate_trains = True
    node = PerceptronMapper(InputMapper((IN,)), enc)
    fan_in = IN
    for width in widths:
        node = PerceptronMapper(node, _lif(width, fan_in))
        fan_in = width
    rep = ModelRepresentation(node)
    mark_encoding_layers(rep)
    rep.assign_perceptron_indices()
    return rep


def _host(source, width: int, name: str) -> ComputeOpMapper:
    return ComputeOpMapper(source, nn.Identity(), input_shape=(width,), output_shape=(width,), name=name)


def _chain(node, widths: Sequence[int], fan_in: int):
    for width in widths:
        node = PerceptronMapper(node, _lif(width, fan_in))
        fan_in = width
    return node


def branchy_repr(kind: str, seed: int = 0) -> ModelRepresentation:
    """Branchy LIF graphs on which the IR-run segments (``partition_ir_graph``) and the shape-only plane's segments
    (``depth_partition``) group the neural cores differently.

    ``"edge_class"``: enc -> A; A -> host -> B; A -> C; concat(B, C) -> D -> out. C sits after the host op in IR
    order, so the edge A -> C is a segment boundary in the IR-run segments and a streamed edge in the re-segmented
    ones. ``"intra_to_cross"``: enc -> host_0; host_0 -> Q -> host_1; host_0 -> P; C reads concat(host_1, P);
    C -> out. The edge P -> C is streamed in the IR-run segments and a segment boundary in the re-segmented ones (the
    only changed edge, the other direction of ``"edge_class"``). ``"cores_only"``: the two branches of host_0
    (A -> A2 -> host_1 -> B, and C -> C2) join at host_2; C and C2 move to A's segment, but every neural-to-neural
    edge keeps its class. ``"deep_mismatch"``: as ``"cores_only"`` with a four-layer first branch, so the moved
    branch (which reads host_0) lands in a deep segment whose later passes also read carried rasters.

    Review round 3 (the shape-only plane's parity on non-chain graphs): ``"skip"``: L3 reads concat(L1, L2), a skip
    two levels deep (a skewed carry); ``"balanced_skip"``: two paths of equal depth merge (no skew);
    ``"late_host_read"``: a deep chain after host_0 whose last layer also reads host_0 (a host-op skip into a chain);
    ``"wide_branches"``: three two-layer branches of host_0 merged on chip; ``"two_hosts"``: a chain fed by host_1
    whose last layer also reads host_1; ``"moved_to_output"``: a short branch of host_0 joined to a deep one at the
    output layer.

    Closing round 2 (a skew of two levels under a collapsing discipline): ``"skew3"``: enc -> L1 -> L2 -> L3, and L4
    reads concat(L3, L1), then L5; L1 is read at latencies 1 and 3.

    U1 exactness: ``"late_skip"``: enc -> A -> B -> C, D reads concat(C, B), then E; at C 3 the capacity split's
    first-overflow cut ([A, B, C], [D, E]) is inexact while the cut [A], [B, C, D], [E] is exact at the same budget.
    ``"deep_skip"``: enc -> L1 -> L2, L3 reads concat(L1, L2), then four layers (G5's census vehicle).

    U1 exactness round 2 (the re-verifier's shift vehicles, ``RV/scripts/probe_shift.py``, built in its order so the
    seeded weights are the same): ``"shift_probe"``: enc -> host_0 -> D1 -> D2 -> D3 -> host_1 -> Q; host_0 -> S;
    X reads S; C2 reads concat(X, S); C1 reads concat(Q, S); host_2 reads concat(C1, C2); out. The fused stage holds
    S, X, C2, Q and C1 and raises S to C2's latency minus one, so C1 reads S one level on structurally but not as
    executed. ``"shift_streamed"``: as ``"shift_probe"`` with a narrower Q and C1 and a second reader Q2 -> C3 of
    host_1, joined at host_2."""
    torch.manual_seed(seed)
    enc = _lif(IN, IN)
    enc.is_encoding_layer = True
    enc.use_cycle_accurate_trains = True
    e = PerceptronMapper(InputMapper((IN,)), enc)
    if kind == "edge_class":
        a = PerceptronMapper(e, _lif(8, IN))
        b = PerceptronMapper(_host(a, 8, "host_h"), _lif(8, 8))
        c = PerceptronMapper(a, _lif(8, 8))
        out = PerceptronMapper(PerceptronMapper(ConcatMapper([b, c]), _lif(8, 16)), _lif(6, 8))
    elif kind == "intra_to_cross":
        h0 = _host(e, IN, "host_0")
        h1 = _host(PerceptronMapper(h0, _lif(8, IN)), 8, "host_1")
        p = PerceptronMapper(h0, _lif(8, IN))
        out = PerceptronMapper(PerceptronMapper(ConcatMapper([h1, p]), _lif(8, 16)), _lif(6, 8))
    elif kind == "deep_mismatch":
        h0 = _host(e, IN, "host_0")
        b = PerceptronMapper(_host(_chain(h0, (16, 16, 16, 16), IN), 16, "host_1"), _lif(8, 16))
        c2 = _chain(h0, (16, 8), IN)
        d = PerceptronMapper(_host(ConcatMapper([b, c2]), 16, "host_2"), _lif(16, 16))
        out = PerceptronMapper(d, _lif(6, 16))
    elif kind == "skip":
        l1 = PerceptronMapper(e, _lif(8, IN))
        l2 = PerceptronMapper(l1, _lif(8, 8))
        out = PerceptronMapper(PerceptronMapper(ConcatMapper([l1, l2]), _lif(8, 16)), _lif(4, 8))
    elif kind == "skew3":
        l1 = _chain(e, (8,), IN)
        l3 = _chain(l1, (8, 8), 8)
        out = PerceptronMapper(PerceptronMapper(ConcatMapper([l3, l1]), _lif(8, 16)), _lif(4, 8))
    elif kind == "late_skip":
        b = _chain(e, (8, 8), IN)
        out = PerceptronMapper(PerceptronMapper(ConcatMapper([_chain(b, (8,), 8), b]), _lif(8, 16)), _lif(4, 8))
    elif kind == "deep_skip":
        l1 = _chain(e, (8,), IN)
        l2 = _chain(l1, (8,), 8)
        out = _chain(PerceptronMapper(ConcatMapper([l1, l2]), _lif(16, 16)), (16, 16, 8, 4), 16)
    elif kind in ("shift_probe", "shift_streamed"):
        h0 = _host(e, IN, "host_0")
        h1 = _host(_chain(h0, (8, 8, 8), IN), 8, "host_1")
        q = PerceptronMapper(h1, _lif(16, 8))
        s = PerceptronMapper(h0, _lif(8, IN))
        c1 = PerceptronMapper(ConcatMapper([q, s]), _lif(16, 24))
        c2 = PerceptronMapper(ConcatMapper([PerceptronMapper(s, _lif(8, 8)), s]), _lif(8, 16))
        if kind == "shift_probe":
            out = PerceptronMapper(_host(ConcatMapper([c1, c2]), 24, "host_2"), _lif(4, 24))
        else:
            c3 = PerceptronMapper(PerceptronMapper(h1, _lif(16, 8)), _lif(8, 16))
            c1 = PerceptronMapper(ConcatMapper([PerceptronMapper(h1, _lif(8, 8)), s]), _lif(8, 16))
            out = PerceptronMapper(_host(ConcatMapper([c1, c3, c2]), 24, "host_2"), _lif(4, 24))
    elif kind == "balanced_skip":
        a = PerceptronMapper(e, _lif(8, IN))
        merged = ConcatMapper([PerceptronMapper(a, _lif(8, 8)), PerceptronMapper(a, _lif(8, 8))])
        out = _chain(PerceptronMapper(merged, _lif(8, 16)), (8, 4), 8)
    elif kind == "late_host_read":
        h0 = _host(e, IN, "host_0")
        last = PerceptronMapper(ConcatMapper([_chain(h0, (8, 8, 8), IN), h0]), _lif(8, 8 + IN))
        out = PerceptronMapper(last, _lif(4, 8))
    elif kind == "wide_branches":
        h0 = _host(e, IN, "host_0")
        merged = PerceptronMapper(ConcatMapper([_chain(h0, (16, 8), IN) for _ in range(3)]), _lif(16, 24))
        out = PerceptronMapper(merged, _lif(6, 16))
    elif kind == "two_hosts":
        h1 = _host(_chain(_host(e, IN, "host_0"), (16, 8), IN), 8, "host_1")
        last = PerceptronMapper(ConcatMapper([_chain(h1, (8, 8, 8), 8), h1]), _lif(8, 16))
        out = PerceptronMapper(last, _lif(4, 8))
    elif kind == "moved_to_output":
        h0 = _host(e, IN, "host_0")
        b = PerceptronMapper(_host(_chain(h0, (16, 16, 16), IN), 16, "host_1"), _lif(8, 16))
        out = PerceptronMapper(ConcatMapper([b, _chain(h0, (16, 8), IN)]), _lif(6, 16))
    else:
        h0 = _host(e, IN, "host_0")
        a2 = PerceptronMapper(PerceptronMapper(h0, _lif(16, IN)), _lif(16, 16))
        b = PerceptronMapper(_host(a2, 16, "host_1"), _lif(8, 16))
        c2 = PerceptronMapper(PerceptronMapper(h0, _lif(16, IN)), _lif(8, 16))
        d = PerceptronMapper(_host(ConcatMapper([b, c2]), 16, "host_2"), _lif(16, 16))
        out = PerceptronMapper(d, _lif(6, 16))
    rep = ModelRepresentation(out)
    mark_encoding_layers(rep)
    rep.assign_perceptron_indices()
    return rep


def split_level_repr() -> ModelRepresentation:
    """The design scout's dither witness (``u1_design/scripts/probe_dither.py``): X reads host_0, Y reads host_1 (one
    level), Z joins them, then out; at C 1 the pass of Y reads host_1 at another offset than the fused stage."""
    torch.manual_seed(0)
    enc = _lif(IN, IN)
    enc.is_encoding_layer = True
    enc.use_cycle_accurate_trains = True
    e = PerceptronMapper(InputMapper((IN,)), enc)
    x = PerceptronMapper(_host(e, IN, "host_0"), _lif(8, IN))
    y = PerceptronMapper(_host(e, IN, "host_1"), _lif(8, IN))
    rep = ModelRepresentation(PerceptronMapper(PerceptronMapper(ConcatMapper([x, y]), _lif(8, 16)), _lif(4, 8)))
    mark_encoding_layers(rep)
    rep.assign_perceptron_indices()
    return rep


def random_rep(seed: int) -> ModelRepresentation:
    """The re-verifiers' U1 fuzz vehicle (``RV2/scripts/fuzz_u1.py``, verbatim, so its seeds name the same graphs):
    a random branchy LIF DAG of 4-9 ops with skips, at most two host ops and narrow layers, every sink joined into
    the output."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    enc = _lif(IN, IN)
    enc.is_encoding_layer = True
    enc.use_cycle_accurate_trains = True
    nodes = [(PerceptronMapper(InputMapper((IN,)), enc), IN, "e")]
    hosts = 0
    for k in range(rng.randint(4, 9)):
        if k and rng.random() < 0.2 and hosts < 2:
            src = rng.choice(nodes[max(0, len(nodes) - 3):])
            nodes.append((_host(src[0], src[1], f"host_{hosts}"), src[1], "h"))
            hosts += 1
            continue
        first = nodes[-1] if rng.random() < 0.6 else rng.choice(nodes)
        srcs = [first]
        if rng.random() < 0.5 and len(nodes) > 1:
            other = rng.choice(nodes)
            if other is not first and first[1] + other[1] <= 24:
                srcs.append(other)
        width_in = sum(s[1] for s in srcs)
        if width_in > 24:
            srcs, width_in = [first], first[1]
        node = srcs[0][0] if len(srcs) == 1 else ConcatMapper([s[0] for s in srcs])
        width = rng.choice([4, 8, 8, 16])
        nodes.append((PerceptronMapper(node, _lif(width, width_in)), width, "p"))
    last = nodes[-1]
    extra = [n for n in nodes[1:-1] if n[2] == "p"]
    srcs = [last]
    if extra:
        cand = rng.choice(extra)
        if last[1] + cand[1] <= 24:
            srcs.append(cand)
    node = srcs[0][0] if len(srcs) == 1 else ConcatMapper([s[0] for s in srcs])
    rep = ModelRepresentation(PerceptronMapper(node, _lif(4, sum(s[1] for s in srcs))))
    mark_encoding_layers(rep)
    rep.assign_perceptron_indices()
    return rep


def wide_rep(seed: int, max_w: int = 40, hosts_max: int = 2, widths: Sequence[int] = (4, 8, 8, 16, 12)):
    """The review's coalescing family (``review_1/rv_fuzz.py``, variant ``coalesce``): ``random_rep``'s generator with
    fan-in up to ``max_w`` on 25-axon cores, so IR coalescing splits wide fan-ins into psum groups."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    enc = _lif(IN, IN)
    enc.is_encoding_layer = True
    enc.use_cycle_accurate_trains = True
    nodes = [(PerceptronMapper(InputMapper((IN,)), enc), IN, "e")]
    hosts = 0
    for k in range(rng.randint(4, 10)):
        if k and rng.random() < 0.25 and hosts < hosts_max:
            src = rng.choice(nodes[max(0, len(nodes) - 3):])
            nodes.append((_host(src[0], src[1], f"host_{hosts}"), src[1], "h"))
            hosts += 1
            continue
        first = nodes[-1] if rng.random() < 0.6 else rng.choice(nodes)
        srcs = [first]
        if rng.random() < 0.5 and len(nodes) > 1:
            other = rng.choice(nodes)
            if other is not first and first[1] + other[1] <= max_w:
                srcs.append(other)
        width_in = sum(s[1] for s in srcs)
        if width_in > max_w:
            srcs, width_in = [first], first[1]
        node = srcs[0][0] if len(srcs) == 1 else ConcatMapper([s[0] for s in srcs])
        width = rng.choice(list(widths))
        nodes.append((PerceptronMapper(node, _lif(width, width_in)), width, "p"))
    last = nodes[-1]
    extra = [n for n in nodes[1:-1] if n[2] == "p"]
    srcs = [last]
    if extra:
        cand = rng.choice(extra)
        if last[1] + cand[1] <= max_w:
            srcs.append(cand)
    node = srcs[0][0] if len(srcs) == 1 else ConcatMapper([s[0] for s in srcs])
    rep = ModelRepresentation(PerceptronMapper(node, _lif(4, sum(s[1] for s in srcs))))
    mark_encoding_layers(rep)
    rep.assign_perceptron_indices()
    return rep


def token_repr(deep: Sequence[int] = (8, 8, 8), tokens: int = 2, width: int = 4, deep_tokens: Sequence[int] = (0,),
               seed: int = 0) -> ModelRepresentation:
    """The review's multi-instance family (``review_1/probe_copack.py``): a token-wise FC (``ReshapeMapper``, one IR node
    per token, one bank) whose ``deep_tokens`` feed a chain and whose other tokens feed the output layer directly, so the
    fused segment runs the instances of one IR depth tier at two latencies."""
    torch.manual_seed(seed)
    enc = _lif(IN, IN)
    enc.is_encoding_layer = True
    enc.use_cycle_accurate_trains = True
    e = PerceptronMapper(InputMapper((IN,)), enc)
    p = PerceptronMapper(ReshapeMapper(e, (tokens, IN // tokens)), _lif(width, IN // tokens))
    deep_src = [SubscriptMapper(p, k) for k in deep_tokens]
    shallow = [SubscriptMapper(p, k) for k in range(tokens) if k not in deep_tokens]
    d = _chain(deep_src[0] if len(deep_src) == 1 else ConcatMapper(deep_src), deep, width * len(deep_tokens))
    rep = ModelRepresentation(PerceptronMapper(ConcatMapper([d] + shallow), _lif(4, deep[-1] + width * len(shallow))))
    mark_encoding_layers(rep)
    rep.assign_perceptron_indices()
    return rep


def cores(count: int) -> List[dict]:
    return [{"max_axons": MAX_AXONS, "max_neurons": MAX_NEURONS, "count": count}]


def layout_plane(rep: ModelRepresentation):
    """The shape-only plane: specs, the walk's wire census and the walk's program graph."""
    mapping = LayoutIRMapping(max_axons=MAX_AXONS, max_neurons=MAX_NEURONS, collect_wire_census=True)
    specs = mapping.collect_layout_softcores(rep)
    return specs, census_of_walk(mapping), ProgramGraph.of_walk(mapping)


def shape_only(specs, census, program, count: int, residency_plan: dict, max_schedule_passes: int = 8):
    """The shape-only plane's composition of the vehicle at ``count`` cores (one core type)."""
    return compose_shape_only_program(
        specs, [LayoutHardCoreType(MAX_AXONS, MAX_NEURONS, count)],
        ShapeOnlyResidency(residency_plan, census, program), allow_coalescing=False,
        allow_splitting=False, max_schedule_passes=max_schedule_passes)


def ir_of(rep: ModelRepresentation, relays: bool = False, coalescing: bool = False):
    """The IR as the pipeline maps it (``SoftCoreMappingStep``): relays inserted when ``lif_depth_balancing_relays``
    is declared (the mapping's window pass, [U-TGAUGE5] dead in-segment reads off first), then the IR latency pass,
    which orders a packed segment's cores for the executors that read it;
    ``coalescing``: wide fan-ins split into psum groups (``IRMapping(allow_coalescing=True)``)."""
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=MAX_AXONS, max_neurons=MAX_NEURONS,
                   allow_coalescing=coalescing).map(rep)
    if relays:
        from mimarsinan.mapping.latency.depth_balancing import balance_cycle_windows

        balance_cycle_windows(ir, cascade=True, armed=True, thresholding_mode="<=")
    IRLatency(ir).calculate()
    return ir


def capabilities(residency_plan: Optional[dict] = None, **kwargs) -> ChipCapabilities:
    return ChipCapabilities(allow_scheduling=True, residency_plan=residency_plan, **kwargs)


def build(ir, count: int, residency_plan: Optional[dict] = None, retimed_level_stages: bool = False, **kwargs):
    caps = capabilities(residency_plan, **kwargs)
    return build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=cores(count), strategy=MappingStrategy.resolve(caps),
        max_schedule_passes=caps.max_schedule_passes, retimed_level_stages=retimed_level_stages)


def fragments(specs, census, count: int, residency_plan: Optional[dict] = None, program=None, **kwargs):
    caps = capabilities(residency_plan, **kwargs)
    return collect_noc_fragments(
        softcores=specs, core_types=[LayoutHardCoreType(MAX_AXONS, MAX_NEURONS, count)], census=census,
        program=program, **caps.layout_kwargs())


def neural_passes(hybrid) -> list:
    return [s for s in hybrid.stages if s.kind == "neural"]


def built_pass_layers(hybrid, ir) -> List[Tuple[int, ...]]:
    """Per executed pass, the sorted perceptron indices of the IR cores it holds."""
    layer_of: Dict[int, int] = {int(n.id): n.perceptron_index for n in ir.get_neural_cores()}
    out = []
    for stage in neural_passes(hybrid):
        ids = {int(p["ir_node_id"]) for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core
               for p in placements}
        out.append(tuple(sorted({layer_of[i] for i in ids})))
    return out


def fragment_pass_layers(frag, specs) -> List[Tuple[int, ...]]:
    return [tuple(sorted({specs[i].perceptron_index for i, _ in placements})) for placements in frag.pass_placements]


def lif_flow(hybrid) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow((IN,), hybrid, simulation_length=T, spiking_mode="lif",
                                 cycle_accurate_lif_forward=True, thresholding_mode="<=")
