"""Mode-aware bias-only liveness semantics for IR pruning."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_liveness import NodeLiveness, compute_liveness
from mimarsinan.mapping.ir_pruning import prune_ir_graph
from mimarsinan.mapping.liveness_semantics import (
    bias_can_activate,
    lif_bias_can_fire,
    ttfs_continuous_bias_can_activate,
    ttfs_quantized_bias_can_activate,
)


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


T = 4
TH = 1.0
SMALL_BIAS = 0.1


def _bias_only_graph():
    """Zero matrix, off axon, small positive bias; B consumes A."""
    a = NeuralCore(
        id=0,
        name="A",
        input_sources=_src([(-1, 0)]),
        core_matrix=np.zeros((1, 1), dtype=np.float64),
        hardware_bias=np.array([SMALL_BIAS], dtype=np.float64),
        threshold=TH,
        latency=0,
    )
    b = NeuralCore(
        id=1,
        name="B",
        input_sources=_src([(0, 0), (-3, 0)]),
        core_matrix=np.array([[1.0], [0.0]], dtype=np.float64),
        threshold=TH,
        latency=1,
    )
    return IRGraph(nodes=[a, b], output_sources=_src([(1, 0)]))


class TestBiasCanActivateHelpers:
    def test_continuous_ttfs_small_positive_bias(self):
        assert ttfs_continuous_bias_can_activate(bias=np.array([SMALL_BIAS]))

    def test_lif_subthreshold_same_bias(self):
        assert not lif_bias_can_fire(
            bias=np.array([SMALL_BIAS]),
            threshold=TH,
            simulation_steps=T,
        )

    def test_quantized_ttfs_subthreshold_same_bias(self):
        assert not ttfs_quantized_bias_can_activate(
            bias=np.array([SMALL_BIAS]),
            threshold=TH,
            simulation_steps=T,
        )

    def test_bias_can_activate_routes_by_mode(self):
        assert bias_can_activate(
            spiking_mode="ttfs",
            bias=np.array([SMALL_BIAS]),
            threshold=TH,
            simulation_steps=T,
        )
        assert not bias_can_activate(
            spiking_mode="lif",
            bias=np.array([SMALL_BIAS]),
            threshold=TH,
            simulation_steps=T,
        )
        assert not bias_can_activate(
            spiking_mode="ttfs_quantized",
            bias=np.array([SMALL_BIAS]),
            threshold=TH,
            simulation_steps=T,
        )


class TestComputeLivenessSpikingMode:
    def test_continuous_ttfs_bias_only_not_dead(self):
        graph = _bias_only_graph()
        result = compute_liveness(
            graph, simulation_steps=T, spiking_mode="ttfs",
        )
        assert result.per_node[0] == NodeLiveness.BIAS_ONLY
        assert "ttfs" in result.reasons[0]

    def test_lif_same_core_still_dead(self):
        graph = _bias_only_graph()
        result = compute_liveness(
            graph, simulation_steps=T, spiking_mode="lif",
        )
        assert result.per_node[0] == NodeLiveness.DEAD

    def test_quantized_ttfs_same_core_dead(self):
        graph = _bias_only_graph()
        result = compute_liveness(
            graph, simulation_steps=T, spiking_mode="ttfs_quantized",
        )
        assert result.per_node[0] == NodeLiveness.DEAD


class TestSoftCoreMappingStepSpikingMode:
    def test_soft_core_mapping_passes_spiking_mode_to_prune(self):
        from pathlib import Path

        scm_path = Path(__file__).resolve().parents[3] / (
            "src/mimarsinan/pipelining/pipeline_steps/soft_core_mapping_step.py"
        )
        text = scm_path.read_text(encoding="utf-8")
        assert "prune_ir_graph(" in text
        assert "spiking_mode=str(" in text
        assert "_apply_ttfs_quantization_bias_compensation" in text


class TestPruneIrGraphSpikingMode:
    def test_prune_preserves_continuous_ttfs_bias_only_node(self):
        graph = _bias_only_graph()
        pruned = prune_ir_graph(
            graph, simulation_steps=T, spiking_mode="ttfs",
        )
        ids = {n.id for n in pruned.nodes if isinstance(n, NeuralCore)}
        assert 0 in ids
        b = next(n for n in pruned.nodes if n.id == 1)
        assert any(
            isinstance(s, IRSource) and s.node_id == 0
            for s in b.input_sources.flatten()
        )

    def test_prune_lif_removes_subthreshold_bias_only(self):
        graph = _bias_only_graph()
        pruned = prune_ir_graph(
            graph, simulation_steps=T, spiking_mode="lif",
        )
        ids = {n.id for n in pruned.nodes if isinstance(n, NeuralCore)}
        assert 0 not in ids
