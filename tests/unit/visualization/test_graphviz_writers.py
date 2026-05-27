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

