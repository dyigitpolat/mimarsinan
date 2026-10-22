"""The IR traversal of the mapper graph is iterative (``Mapper.map_to_ir``, an explicit stack).

On the base ``map_to_ir`` recursed through its sources (three Python frames per mapper), so a deep mapper graph ended
in a loud ``RecursionError`` at the default recursion limit: EfficientNet-B4 to B7, EfficientNetV2-S/M/L and ConvNeXt-S/B/L
once their stochastic depth is inlined (scout probe 8). The walk now maps every unmapped source first, in the post-order
the recursion visited (each mapper's sources in the order it reads them), so the IR's nodes, ids and sources are the
recursive walk's, and no mapper nests a frame.
"""

from __future__ import annotations

import contextlib
import io
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronFanInError, PerceptronMapper
from mimarsinan.mapping.mappers.structural import ConcatMapper, InputMapper, ReshapeMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

DEPTH = 2000


def _perceptron(out_features: int, in_features: int, seed: int) -> Perceptron:
    torch.manual_seed(seed)
    return Perceptron(out_features, in_features, normalization=nn.Identity())


class _Add(nn.Module):
    def forward(self, a, b):
        return a + b


def _ir_mapping() -> IRMapping:
    return IRMapping(q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None, allow_coalescing=False,
                     hardware_bias=False)


def _chain(depth: int) -> ModelRepresentation:
    """Input (4,) then ``depth`` reshapes then a Linear(4, 3): ``depth + 2`` mappers on one path."""
    node: Mapper = InputMapper((4,))
    for _ in range(depth):
        node = ReshapeMapper(node, (4,))
    return ModelRepresentation(PerceptronMapper(node, _perceptron(3, 4, 0)))


def _dag() -> ModelRepresentation:
    """Shared sources, a fan-in join and a concat: a mapper reached on two paths is mapped once, at its first read."""
    inp = InputMapper((4,))
    a = PerceptronMapper(inp, _perceptron(4, 4, 1))
    b = PerceptronMapper(a, _perceptron(4, 4, 2))
    c = PerceptronMapper(inp, _perceptron(4, 4, 3))
    join = ComputeOpMapper([b, a], _Add(), input_shapes=[(4,), (4,)], output_shape=(4,), name="join")
    cat = ConcatMapper([c, join, b], dim=1, rank=2)
    return ModelRepresentation(PerceptronMapper(cat, _perceptron(2, 12, 4)))


def _recursive_post_order(root: Mapper) -> list:
    """The order the recursive walk mapped in: each mapper after its sources, in the order it reads them."""
    seen, order = set(), []

    def visit(mapper):
        if id(mapper) in seen:
            return
        seen.add(id(mapper))
        for source in mapper.get_source_mappers():
            visit(source)
        order.append(mapper)

    visit(root)
    return order


def _digest(graph) -> list:
    rows = []
    for node in graph.nodes:
        sources = [(s.node_id, s.index) for s in np.asarray(node.input_sources, dtype=object).flatten()]
        matrix = node.core_matrix if isinstance(node, NeuralCore) else None
        weights = None if matrix is None else matrix.tobytes()
        rows.append((type(node).__name__, node.id, node.name, sources, weights))
    return rows


def test_a_chain_of_2000_mappers_maps_at_the_default_recursion_limit():
    assert sys.getrecursionlimit() <= 1000 + 64  # the interpreter's default (pytest adds a few frames, not a limit)
    with contextlib.redirect_stdout(io.StringIO()):
        graph = _ir_mapping().map(_chain(DEPTH))
    cores = [node for node in graph.nodes if isinstance(node, NeuralCore)]
    assert len(cores) == 1 and cores[0].core_matrix.shape == (5, 3)
    sources = np.asarray(cores[0].input_sources, dtype=object).flatten()
    assert [(s.node_id, s.index) for s in sources if s.node_id != -3] == [(-2, i) for i in range(4)]  # -3: the bias


def test_the_layout_mapping_of_a_deep_chain_does_not_recurse_either():
    layout = LayoutIRMapping(max_axons=None, max_neurons=None, allow_coalescing=False, hardware_bias=False)
    with contextlib.redirect_stdout(io.StringIO()):
        layout.map(_chain(DEPTH))
    assert len(layout.layout_softcores) == 1


def test_mappers_are_mapped_in_the_recursive_post_order(monkeypatch):
    repr_ = _dag()
    want = _recursive_post_order(repr_.output_layer_mapper)
    mapped = []
    for mapper in want:
        own = mapper._map_to_ir

        def logged(ir_mapping, own=own, mapper=mapper):
            sources = own(ir_mapping)
            mapped.append(mapper)  # on completion: the order its IR nodes were emitted in
            return sources

        monkeypatch.setattr(mapper, "_map_to_ir", logged)
    with contextlib.redirect_stdout(io.StringIO()):
        _ir_mapping().map(repr_)
    assert [id(m) for m in mapped] == [id(m) for m in want]


def test_the_ir_of_a_dag_is_the_recursive_walks():
    """Node ids, names, sources and weights equal those of a walk that maps each mapper by recursion (the base's)."""
    repr_ = _dag()
    with contextlib.redirect_stdout(io.StringIO()):
        iterative = _digest(_ir_mapping().map(repr_))
    repr_ = _dag()
    ir_mapping = _ir_mapping()
    for mapper in _recursive_post_order(repr_.output_layer_mapper):  # every source first: map_to_ir hits the cache
        mapper.map_to_ir(ir_mapping)
    with contextlib.redirect_stdout(io.StringIO()):
        reference = _digest(ir_mapping.map(repr_))
    assert iterative == reference
    assert [row[0] for row in iterative].count("NeuralCore") == 4 and [row[0] for row in iterative].count(
        "ComputeOp") == 1


def test_a_second_ir_mapping_maps_the_graph_again():
    repr_ = _dag()
    with contextlib.redirect_stdout(io.StringIO()):
        first, second = _ir_mapping().map(repr_), _ir_mapping().map(repr_)
    assert _digest(first) == _digest(second)


def test_a_refusal_deep_in_the_graph_propagates_as_itself():
    node: Mapper = InputMapper((4,))
    for _ in range(DEPTH):
        node = ReshapeMapper(node, (4,))
    node = ReshapeMapper(node, (2, 2, 1))  # per-sample (2, 2, 1): neither flattens to 3 nor has it on an axis
    with pytest.raises(PerceptronFanInError), contextlib.redirect_stdout(io.StringIO()):
        _ir_mapping().map(ModelRepresentation(PerceptronMapper(node, _perceptron(2, 3, 5))))


def test_a_source_that_is_not_a_mapper_is_read_through_its_own_map_to_ir():
    """A duck-typed source (``map_to_ir`` alone) is left to the reader, as the recursive walk left it."""

    class _Source:
        def map_to_ir(self, _ir_mapping):
            return np.array([[IRSource(-2, i) for i in range(4)]], dtype=object)

    ir_mapping = _ir_mapping()
    out = PerceptronMapper(_Source(), _perceptron(3, 4, 6)).map_to_ir(ir_mapping)
    assert np.asarray(out, dtype=object).size == 3 and len(ir_mapping.nodes) == 1
