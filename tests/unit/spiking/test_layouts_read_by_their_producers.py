"""[U-TGAUGE4] The placement rule for both walk families (the U-TGAUGE3 verify's M-1 and M-2, and the further sites of
the class): every read of a layout's element, in every walk and at every core input, takes every quantity the deployed
program holds per producer from that producer, never from the layout or from the core that reads it.

The deployed program holds per producer the gauge its buffer holds each element at, its value complete in its own
window, and, at a reading core, the fold the core's weights apply to that input (``per_input_scales``). The sites, one
test each:

- the timing walk read a layout of producers at different cascade depths over the layout's own window (M-1): the
  cascaded gate read 0.53 to 0.98, the walk up to a whole gauge off the deployed read;
- the timing core read every input at its one input scale, the mean of its sources, where the deployed core folds
  each input at its producer's gauge (M-2's fold), and encoded an outside producer's elements at that mean (M-2's
  encode): the gate read down to 0.0625;
- a timing core read an in-segment producer more than one hop shallower at its own previous cycle, not at the
  producer's window (the gate read 0.16 to 0.94);
- a timing output perceptron at a per-channel scale (the verify's W22 gap);
- the source walk held a layout at one value where its sources differ (a stack, the verify's m-1), kept a join's
  vector as one gauge per channel where a part is a layout placed per element (an axis move), and refused a reshape
  the identity program's table places;
- a gather that holds one producer of a join at differing gauges was read at the join's boundary mean: by the count
  walk's decode and by the timing core's input scale;
- a fold that cannot hold its sources' gauges per input spread their mean; it refuses by name.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import unit.spiking.host_gather_forms as hg
import unit.spiking.layout_trees as lt

from mimarsinan.chip_simulation.hybrid_run.output_decode import executor_output_values
from mimarsinan.mapping.mappers.source_gauges import read_source_gauges
from mimarsinan.mapping.mappers.structural import (
    ConcatMapper, InputMapper, PermuteMapper, ReshapeMapper, StackMapper, SubscriptMapper)
from mimarsinan.mapping.mapping_utils import ModelRepresentation, PerceptronMapper
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.core import nf_scm_parity
from mimarsinan.spiking.segment_forward import SegmentForwardDriver

S = 32


# ---- the timing family: the tree's cascaded pair ---------------------------------------------------------------------

def _branch(i, o, depth):
    dims = [i] + [8] * (depth - 1) + [o]
    layers: list = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


class _Join(nn.Module):
    """Branches of the given cascade depths over one encoding, joined (``enc``: the encoding's own output joined too),
    read as the output, by a core (``core``), or by a host affine op then a core (``host_core``)."""

    def __init__(self, depths, read="none", enc=False, width=12):
        super().__init__()
        self.read, self.enc = read, enc
        self.a = nn.Sequential(nn.Linear(10, width), nn.ReLU())
        self.br = nn.ModuleList(_branch(width, 3, d) for d in depths)
        n = 3 * len(depths) + (width if enc else 0)
        if read in ("core", "host_core"):
            self.f = nn.Sequential(nn.Linear(n, 6), nn.ReLU())

    def forward(self, x):
        h = self.a(x)
        out = torch.cat([b(h) for b in self.br] + ([h] if self.enc else []), dim=1)
        if self.read == "core":
            return self.f(out)
        if self.read == "host_core":
            return self.f(out * 0.7 + 0.3)
        return out


FORMS = {
    "unequal_depth": lambda: _Join([2, 1]),
    "three_depths": lambda: _Join([2, 1, 3]),
    "encoding_mix": lambda: _Join([2, 2], enc=True, width=4),
    "three_scales_core_head": lambda: _Join([2, 2, 2], "core"),
    "three_depths_host_head": lambda: _Join([2, 1, 3], "host_core"),
    "unequal_depth_core": lambda: _Join([2, 1], "core"),
    "three_depths_core": lambda: _Join([1, 2, 3], "core"),
}


def _pair(form: str, seed: int):
    import unit.spiking.test_cascaded_boundary_transcoding as cb

    torch.manual_seed(seed)
    model = FORMS[form]()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.5, 0.5)
                nn.init.uniform_(m.bias, -0.05, 0.05)
    model.eval()
    with torch.no_grad():
        n_out = int(model(torch.zeros(2, 10)).shape[1])
    flow, executor, ir = cb._build_pair(model, (10,), n_out, S, "<=")
    torch.manual_seed(3 + seed)
    return flow, executor, ir, torch.rand(64, 10, dtype=torch.float64)


class _Walk(nn.Module):
    def __init__(self, flow):
        super().__init__()
        from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward

        self.flow = flow
        self.walk = TTFSSegmentForward(flow.get_mapper_repr(), S)

    def forward(self, x):
        return self.walk(x)


class _Pipeline:
    config: dict = {"device": "cpu"}
    reporter = None


def _gate_and_read(form, seed, monkeypatch):
    flow, executor, ir, x = _pair(form, seed)
    monkeypatch.setattr(nf_scm_parity, "_build_cascaded_identity_executor", lambda _p, _m, _ir: executor)
    gate = nf_scm_parity._cascaded_nf_scm_agreement(_Pipeline(), _Walk(flow), ir, x, min_agreement=0.0)
    with torch.no_grad():
        walk = _Walk(flow)(x).double()
        deployed = executor_output_values(executor, executor(x)).double()
    return gate, walk, deployed


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("form", ["unequal_depth", "three_depths", "encoding_mix"])
def test_a_timing_layout_of_producers_at_different_cascade_depths_is_read_from_each_producer_s_window(
        form, seed, monkeypatch):
    """M-1: the layout was accumulated over its own window, from its deepest producer's depth (gate 0.53 to 0.98; a
    saturated encoding element read 0.0 where the deployed read holds 1.0)."""
    gate, walk, deployed = _gate_and_read(form, seed, monkeypatch)
    assert gate == 1.0
    torch.testing.assert_close(walk, deployed, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("form", ["three_scales_core_head", "three_depths_host_head"],
                         ids=["fold_of_in_segment_producers", "encode_of_an_outside_producer"])
def test_a_timing_core_reads_each_input_at_its_producer_s_gauge(form, seed, monkeypatch):
    """M-2: a core over three producers at three scales folded every input at the mean (gate down to 0.625), and a core
    over a host op over such a join encoded the op's elements at that mean (gate down to 0.0625)."""
    gate, walk, deployed = _gate_and_read(form, seed, monkeypatch)
    assert gate == 1.0
    torch.testing.assert_close(walk, deployed, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("form", ["unequal_depth_core", "three_depths_core"])
def test_a_timing_core_reads_a_shallower_producer_from_that_producer_s_window(form, seed, monkeypatch):
    """A core of the segment reading a join of producers one, two and three hops deep read the shallower ones at its
    own previous cycle, late by the depth gap and their early spikes lost (gate 0.16 to 0.94)."""
    gate, walk, deployed = _gate_and_read(form, seed, monkeypatch)
    assert gate == 1.0
    torch.testing.assert_close(walk, deployed, rtol=1e-6, atol=1e-6)


# ---- hand-built graphs (the toolchain's layout mappers) ---------------------------------------------------------------

def _perceptron(src, out, inp, name, gauge, perceptrons, gauges, gen):
    p = Perceptron(out, inp, name=name)
    with torch.no_grad():
        p.layer.weight.copy_(torch.rand(p.layer.weight.shape, generator=gen) * 1.5 - 0.6)
        p.layer.bias.copy_(torch.rand(p.layer.bias.shape, generator=gen) * 0.15)
    perceptrons.append(p)
    gauges.append(gauge)
    return PerceptronMapper(src, p)


def _tree(build, text: str) -> lt.Tree:
    """A tree of the property test's form built by ``build(perceptron) -> root``."""
    perceptrons: list = []
    gauges: list = []
    gen = torch.Generator().manual_seed(11)

    def perceptron(src, out, inp, name, gauge):
        return _perceptron(src, out, inp, name, gauge, perceptrons, gauges, gen)

    root = build(perceptron, InputMapper((lt.TOKENS, lt.FEATURES)))
    return lt.Tree(0, text, ModelRepresentation(root), perceptrons, gauges)


def _timing_read(tree: lt.Tree):
    walk, executor = lt.timing_program(tree)
    torch.manual_seed(7)
    x = torch.rand(16, *tree.shape, dtype=torch.float64)
    with torch.no_grad():
        ours = walk(x).double().reshape(16, -1)
        deployed = executor_output_values(executor, executor(x)).double().reshape(16, -1)
    return ours, deployed


def _count_read(tree: lt.Tree, semantics: str, **pruning):
    program = lt.count_program(tree, semantics, **pruning)
    x = hg.samples(program, 16, 0)
    with torch.no_grad(), measurement_plane():
        ours = SegmentForwardDriver(program.repr_, hg.T, hg.SEMANTICS[semantics][2]())(x).double().reshape(16, -1)
    flow = hg.deployed_flow(program, semantics)
    with torch.no_grad():
        deployed = executor_output_values(flow, flow(x)).double().reshape(16, -1)
    return ours, deployed


def _per_channel_output(perceptron, enc):
    p1 = perceptron(enc, 4, 5, "p1", 1.4)
    return perceptron(p1, 3, 4, "out", [0.6, 2.2, 1.3])


def test_a_timing_output_perceptron_at_a_per_channel_scale_is_decoded_per_channel():
    """The verify's W22 gap: no test read a timing perceptron at a per-channel scale (a walk decoding its output at the
    mean of its scales passed every test)."""
    tree = _tree(lambda perceptron, inp: _per_channel_output(perceptron, perceptron(inp, 5, lt.FEATURES, "enc", 1.0)),
                 "per-channel output")
    ours, deployed = _timing_read(tree)
    assert float(deployed.abs().max()) > 0.0
    torch.testing.assert_close(ours, deployed, rtol=1e-6, atol=1e-6)


def _stack_of_three(perceptron, inp):
    """Three token producers at three one-value gauges stacked, flattened and read by a core (the verify's m-1)."""
    enc = perceptron(inp, 4, lt.FEATURES, "enc", 1.0)
    parts = [perceptron(enc, 3, 4, f"p{i}", g) for i, g in enumerate((0.5, 1.5, 3.0))]
    flat = ReshapeMapper(StackMapper(parts), (3 * lt.TOKENS * 3,))
    return perceptron(flat, 3, 3 * lt.TOKENS * 3, "reader", 1.2)


def test_a_stack_of_producers_at_three_gauges_is_placed_per_element_and_folded_per_input():
    """The stack's propagation keeps its first source's one value, and the placement passed one value without reading
    the sources: the reader folded all its inputs at 0.5."""
    tree = _tree(_stack_of_three, "stack")
    for p, g in zip(tree.perceptrons, tree.gauges):
        p.set_activation_scale(torch.as_tensor(g, dtype=torch.float32))
    with torch.no_grad():
        tree.repr_(torch.rand(2, *tree.shape))
    tree.repr_.assign_perceptron_indices()
    compute_per_source_scales(tree.repr_)
    expected = [g for g in (0.5, 1.5, 3.0) for _ in range(lt.TOKENS * 3)]
    reader = tree.perceptrons[-1]
    np.testing.assert_allclose(torch.as_tensor(reader.per_input_scales).numpy(), expected, rtol=1e-6)
    stack = next(n for n in tree.repr_._exec_order if isinstance(n, StackMapper))
    np.testing.assert_allclose(torch.as_tensor(read_source_gauges(tree.repr_)[stack]).numpy(), expected, rtol=1e-6)


@pytest.mark.parametrize("semantics", sorted(hg.SEMANTICS))
def test_both_twins_read_a_stack_of_producers_at_three_gauges_alike(semantics):
    tree = _tree(_stack_of_three, "stack")
    torch.testing.assert_close(*_count_read(tree, semantics), rtol=1e-6, atol=1e-6)
    tree = _tree(_stack_of_three, "stack")
    torch.testing.assert_close(*_timing_read(tree), rtol=1e-6, atol=1e-6)


def _moved_join(perceptron, inp):
    """A producer's tokens-by-channels output with its axes moved (a per-element gauge), joined along the last axis
    with a producer of another gauge, flattened, joined with a third, and read by a core."""
    enc = perceptron(inp, 5, lt.FEATURES, "enc", 1.0)
    a = perceptron(enc, 2, 5, "a", [2.4, 1.0])  # (2, 2) channels at two gauges
    b = perceptron(enc, 3, 5, "b", 1.9)
    c = perceptron(enc, 4, 5, "c", 0.8)
    join = ConcatMapper([b, PermuteMapper(a, (0, 2, 1))], dim=2, rank=3)  # (2, 5): b's channels, then a's tokens
    flat = ReshapeMapper(join, (10,))
    outer = ConcatMapper([ReshapeMapper(c, (8,)), flat], dim=1, rank=2)
    return perceptron(outer, 3, 18, "reader", 1.1)


def test_a_join_over_an_axis_move_is_placed_by_its_elements_sources():
    """U-TGAUGE3's channel-join form took the moved part's per-element vector as one gauge per channel: the join kept a
    vector of 7 entries for 5 channels, and both twins folded the reader's inputs off their producers' gauges."""
    tree = _tree(_moved_join, "moved join")
    for p, g in zip(tree.perceptrons, tree.gauges):
        p.set_activation_scale(torch.as_tensor(g, dtype=torch.float32))
    with torch.no_grad():
        tree.repr_(torch.rand(2, *tree.shape))
    tree.repr_.assign_perceptron_indices()
    compute_per_source_scales(tree.repr_)
    # c's 8 elements at 0.8; per token: b's three channels at 1.9, then a's channel 0 and channel 1 (its two tokens)
    expected = [0.8] * 8 + [1.9, 1.9, 1.9, 2.4, 2.4] + [1.9, 1.9, 1.9, 1.0, 1.0]
    np.testing.assert_allclose(torch.as_tensor(tree.perceptrons[-1].per_input_scales).numpy(), expected, rtol=1e-6)


@pytest.mark.parametrize("semantics", sorted(hg.SEMANTICS))
def test_both_twins_read_a_join_over_an_axis_move_alike(semantics):
    torch.testing.assert_close(*_count_read(_tree(_moved_join, "moved join"), semantics), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(*_timing_read(_tree(_moved_join, "moved join")), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("family", ["timing", "count"])
def test_a_reshape_over_a_misaligned_join_is_placed_not_refused(family):
    """The generator's tree 41 (axis moves, a join along the tokens, joins along the channels, a flatten): U-TGAUGE3's
    channel-join form kept a vector misaligned with the joined channels, and its reshape form then refused the flatten
    by name (``a reshape to (24,) of a 17-channel gauge``), a layout the identity program's table places."""
    rec = lt.read(41, family)
    assert "refused" not in rec, rec
    assert rec["max_abs"] <= 1e-6, rec


def _gathered_join(read: str):
    def build(perceptron, inp):
        """Two producers at two gauges joined along the tokens; a gather holds the first producer's token only (one
        gauge) while the join's boundary walk holds the mean of the two; read as the output (the second producer's
        core is then dead, as IR pruning removes it) or by a core beside the second producer's output."""
        enc = perceptron(inp, 5, lt.FEATURES, "enc", 1.0)
        a = perceptron(enc, 3, 5, "a", 2.3)
        b = perceptron(enc, 3, 5, "b", 0.7)
        gathered = SubscriptMapper(ConcatMapper([a, b], dim=1, rank=3), 0)  # a's first token, at 2.3
        if read == "out":
            return gathered
        core = perceptron(gathered, 2, 3, "reader", 1.6)
        return ConcatMapper([core, ReshapeMapper(b, (6,))], dim=1, rank=2)
    return build


@pytest.mark.parametrize("semantics", sorted(hg.SEMANTICS))
def test_the_count_walk_decodes_a_gather_of_one_producer_at_that_producer_s_gauge(semantics):
    """The count walk decoded the gathered layout at the boundary table's scalar, the join's own mean (1.5 for 2.3):
    35 percent low against the deployed read."""
    ours, deployed = _count_read(_tree(_gathered_join("out"), "gather out"), semantics, prune=True)
    assert float(deployed.abs().max()) > 0.0
    torch.testing.assert_close(ours, deployed, rtol=1e-6, atol=1e-6)


def test_a_timing_core_over_a_gather_of_one_producer_folds_it_at_that_producer_s_gauge():
    """The timing core's input scale is the boundary walk's, the join's mean (1.5), while the deployed core folds the
    gathered producer's 2.3: the walk read 0.19 off the deployed read."""
    ours, deployed = _timing_read(_tree(_gathered_join("core"), "gather core"))
    torch.testing.assert_close(ours, deployed, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("family", ["count", "timing"])
def test_a_fold_that_differs_within_an_input_channel_is_held_per_core_position(family):
    """[U-PFOLD, the composed head's rule replacing the fold's named refusal] A token-wise core reading a layout whose
    gauge differs within one input channel (the tokens at two gauges): its fold is held one row per core position, so
    each core folds its own token at its producer's gauge, and both families read the tree as the deployed program."""
    def build(perceptron, inp):
        enc = perceptron(inp, 5, lt.FEATURES, "enc", 1.0)
        a = perceptron(ReshapeMapper(SubscriptMapper(enc, 0), (1, 5)), 3, 5, "a", 2.3)
        b = perceptron(ReshapeMapper(SubscriptMapper(enc, 1), (1, 5)), 3, 5, "b", 0.7)
        return perceptron(ConcatMapper([a, b], dim=1, rank=3), 2, 3, "token_reader", 1.0)

    for semantics in (sorted(hg.SEMANTICS) if family == "count" else [None]):
        tree = _tree(build, "token reader")
        ours, deployed = _count_read(tree, semantics) if family == "count" else _timing_read(tree)
        assert tree.perceptrons[-1].per_input_scales.dim() == 2, "one row per core position"
        assert float(deployed.abs().max()) > 0.0
        torch.testing.assert_close(ours, deployed, rtol=1e-6, atol=1e-6)
