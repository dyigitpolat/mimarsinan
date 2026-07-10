"""ModelRepresentation graph accessors: execution order and consumer map."""

import torch

from mimarsinan.mapping.mappers.leading_dim import Ensure2DMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


def _chain():
    torch.manual_seed(0)
    inp = InputMapper((4,))
    ensure = Ensure2DMapper(inp)
    node_a = PerceptronMapper(ensure, Perceptron(8, 4))
    node_b = PerceptronMapper(node_a, Perceptron(3, 8))
    return inp, ensure, node_a, node_b, ModelRepresentation(node_b)


def test_execution_order_is_deps_first():
    inp, ensure, node_a, node_b, repr_ = _chain()
    order = repr_.execution_order()
    assert order == [inp, ensure, node_a, node_b]


def test_consumer_map_reverses_dependency_edges():
    inp, ensure, node_a, node_b, repr_ = _chain()
    consumers = repr_.consumer_map()
    assert consumers[id(inp)] == [ensure]
    assert consumers[id(ensure)] == [node_a]
    assert consumers[id(node_a)] == [node_b]
    assert id(node_b) not in consumers
