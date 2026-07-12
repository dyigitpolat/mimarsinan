"""Shared channel-axis walk: aligned-consumer discovery over the mapper DAG.

This is the one graph-walk SSOT behind M4 scale migration AND the LIF affine
fold's consumer discovery. Real converted graphs interleave pass-through
mappers (Ensure2D, leading-dim merges) between perceptrons and terminate in
host ComputeOps, so the walk — not direct-edge adjacency — is the only honest
way to find channel-aligned consumers.
"""

import torch
import torch.nn as nn

from mimarsinan.mapping.channel_axis_walk import (
    channel_aligned_consumer_targets,
    consumer_columns_unmediated,
)
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.leading_dim import Ensure2DMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


def _interleaved_chain(perceptrons, in_features=4):
    """Perceptron chain with Ensure2D between hops — the REAL converted-graph
    shape (no direct perceptron->perceptron edges)."""
    node = Ensure2DMapper(InputMapper((in_features,)))
    perceptron_nodes = []
    for p in perceptrons:
        node = PerceptronMapper(node, p)
        perceptron_nodes.append(node)
        node = Ensure2DMapper(node)
    return perceptron_nodes, node


class TestConsumerColumnsUnmediated:
    def test_plain_linear_consumer_is_unmediated(self):
        torch.manual_seed(0)
        assert consumer_columns_unmediated(Perceptron(3, 8))

    def test_input_wire_op_mediates(self):
        torch.manual_seed(0)
        p = Perceptron(3, 8)
        p.append_input_wire_op(nn.ReLU())
        assert not consumer_columns_unmediated(p)

    def test_per_input_scales_mediate(self):
        torch.manual_seed(0)
        p = Perceptron(3, 8)
        p.per_input_scales = torch.ones(8)
        assert not consumer_columns_unmediated(p)


class TestWalkThroughPassThroughNodes:
    def test_finds_consumer_through_ensure2d(self):
        torch.manual_seed(0)
        p1, p2 = Perceptron(8, 4), Perceptron(3, 8)
        (n1, _n2), out = _interleaved_chain([p1, p2])
        consumers = ModelRepresentation(out).consumer_map()
        assert channel_aligned_consumer_targets(n1, consumers) == ((p2,), ())

    def test_terminal_producer_reaching_output_is_none(self):
        torch.manual_seed(0)
        p1, p2 = Perceptron(8, 4), Perceptron(3, 8)
        (_n1, n2), out = _interleaved_chain([p1, p2])
        consumers = ModelRepresentation(out).consumer_map()
        assert channel_aligned_consumer_targets(n2, consumers) is None

    def test_host_linear_compute_op_is_module_target(self):
        torch.manual_seed(0)
        p1 = Perceptron(8, 4)
        (n1,), tail = _interleaved_chain([p1])
        classifier = nn.Linear(8, 3)
        out = ComputeOpMapper(tail, classifier, name="classifier")
        consumers = ModelRepresentation(out).consumer_map()
        assert channel_aligned_consumer_targets(n1, consumers) == ((), (classifier,))

    def test_mean_over_non_channel_axis_passes_through(self):
        torch.manual_seed(0)
        inp = InputMapper((4, 8))
        mean_node = ComputeOpMapper(
            inp,
            ComputeAdapter(torch.mean, kwargs={"dim": 1}),
            input_shapes=((4, 8),),
        )
        p = Perceptron(3, 8)
        out = PerceptronMapper(Ensure2DMapper(mean_node), p)
        consumers = ModelRepresentation(out).consumer_map()
        assert channel_aligned_consumer_targets(inp, consumers) == ((p,), ())


class TestWalkVoids:
    def test_mediated_consumer_voids_the_walk(self):
        torch.manual_seed(0)
        p1, p2 = Perceptron(8, 4), Perceptron(3, 8)
        p2.append_input_wire_op(nn.ReLU())
        (n1, _n2), out = _interleaved_chain([p1, p2])
        consumers = ModelRepresentation(out).consumer_map()
        assert channel_aligned_consumer_targets(n1, consumers) is None

    def test_fan_out_with_one_bad_path_voids_the_walk(self):
        """Fan-out closure: a residual-style add join voids the producer even
        though its other path reaches a clean perceptron."""
        torch.manual_seed(0)
        p1, p2 = Perceptron(8, 4), Perceptron(8, 8)
        (n1, n2), _tail = _interleaved_chain([p1, p2])
        join = ComputeOpMapper([n1, n2], ComputeAdapter(torch.add))
        consumers = ModelRepresentation(join).consumer_map()
        assert channel_aligned_consumer_targets(n1, consumers) is None
