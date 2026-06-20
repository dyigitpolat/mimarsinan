"""Graphviz writers emit valid DOT for minimal fixtures."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.visualization.graphviz import write_ir_graph_dot


def _minimal_ir_graph() -> IRGraph:
    core = NeuralCore(
        id=0,
        name="n0",
        input_sources=np.array([IRSource(-2, 0)], dtype=object),
        core_matrix=np.ones((1, 1)),
        threshold=1.0,
        activation_scale=1.0,
    )
    return IRGraph(nodes=[core], output_sources=np.array([IRSource(0, 0)], dtype=object))


def test_write_ir_graph_dot(tmp_path):
    out = tmp_path / "ir.dot"
    write_ir_graph_dot(_minimal_ir_graph(), str(out))
    text = out.read_text(encoding="utf-8")
    assert "digraph" in text
    assert "NeuralCore" in text or "n0" in text


def test_softcore_flowchart_dot_emits_fc_estimate():
    """V6: the flowchart routes per-node estimation through the polymorphic
    ``flowchart_node_estimate``; an FC node yields a real HW core estimate."""
    import torch.nn as nn

    from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
    from mimarsinan.mapping.mappers.structural import InputMapper
    from mimarsinan.mapping.model_representation import ModelRepresentation
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
    from mimarsinan.visualization.softcore_flowchart_dot import (
        generate_softcore_flowchart_dot,
    )

    p = Perceptron(32, 16, normalization=nn.Identity(), base_activation_name="ReLU")
    repr_ = ModelRepresentation(PerceptronMapper(InputMapper((16,)), p))
    dot = generate_softcore_flowchart_dot(
        repr_, input_shape=(16,), max_axons=256, max_neurons=256
    )
    assert "digraph SoftCoreFlowchart" in dot
    assert "SW perceptrons=1 (in_features=16, out_features=32)" in dot
    assert "cores" in dot

