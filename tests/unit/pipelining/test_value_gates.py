"""[mvm] value-domain gates: arming, typed skips, R-edge and C-edge FATALs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import mimarsinan.pipelining.core.gates.value_gates as gates
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.pipelining.core.spike_count_gate import (
    run_spike_count_certificate_gate,
)
from mimarsinan.torch_mapping.converter import convert_torch_model


def _pipeline(config):
    return SimpleNamespace(config=config, cache={})


def _mvm_chain():
    torch.manual_seed(5)
    model = nn.Sequential(
        nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4)
    ).eval()
    from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron

    flow = convert_torch_model(
        model, (8,), 4, device="cpu", packaging=MVM_PACKAGING
    ).eval()
    for p in flow.get_perceptrons():
        fuse_into_perceptron(p, device="cpu")
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=64, max_neurons=64
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 99}],
    )
    return flow, ir, hybrid


class TestArming:
    def test_armed_for_mvm_plans(self):
        pipeline = _pipeline({"core_semantics": "mvm"})
        assert gates.value_certificate_gate_armed(pipeline) is True

    def test_unarmed_for_spiking_plans(self):
        pipeline = _pipeline({"spiking_mode": "lif"})
        assert gates.value_certificate_gate_armed(pipeline) is False

    def test_unarmed_at_zero_samples(self):
        pipeline = _pipeline({"core_semantics": "mvm", "value_parity_samples": 0})
        assert gates.value_certificate_gate_armed(pipeline) is False

    def test_counts_gate_typed_skips_under_mvm(self):
        pipeline = _pipeline({"core_semantics": "mvm", "spike_count_parity_samples": 2})
        assert run_spike_count_certificate_gate(
            pipeline, model=None, ir_graph=None, hybrid_mapping=None,
        ) is None


class TestREdge:
    def test_pass_on_faithful_identity(self, monkeypatch):
        flow, ir, _ = _mvm_chain()
        pipeline = _pipeline({"core_semantics": "mvm", "device": "cpu"})
        monkeypatch.setattr(
            gates, "_certificate_samples",
            lambda pipeline, model, n: torch.randn(n, 8),
        )
        gates.run_model_value_parity_gate(pipeline, flow, ir)

    def test_fatal_on_defective_mapping(self, monkeypatch):
        flow, ir, _ = _mvm_chain()
        for node in ir.nodes:
            if type(node).__name__ == "NeuralCore" and node.core_matrix is not None:
                node.core_matrix = node.core_matrix.copy()
                node.core_matrix[0, 0] += 0.5
                break
        pipeline = _pipeline({"core_semantics": "mvm", "device": "cpu"})
        monkeypatch.setattr(
            gates, "_certificate_samples",
            lambda pipeline, model, n: torch.randn(n, 8),
        )
        with pytest.raises(RuntimeError, match="R-edge"):
            gates.run_model_value_parity_gate(pipeline, flow, ir)


class TestCEdge:
    def test_pass_on_faithful_packing(self, monkeypatch):
        flow, ir, hybrid = _mvm_chain()
        pipeline = _pipeline({"core_semantics": "mvm", "device": "cpu"})
        monkeypatch.setattr(
            gates, "_certificate_samples",
            lambda pipeline, model, n: torch.randn(n, 8),
        )
        cert = gates.run_value_twin_certificate_gate(pipeline, flow, ir, hybrid)
        assert cert is not None and cert.passed
        assert cert.neuron_windows_compared > 0

    def test_fatal_on_corrupted_packed_program(self, monkeypatch):
        flow, ir, hybrid = _mvm_chain()
        seg = hybrid.get_neural_segments()[0]
        seg.cores[0].core_matrix = seg.cores[0].get_core_matrix().copy()
        seg.cores[0].core_matrix[0, 0] += 1.0
        pipeline = _pipeline({"core_semantics": "mvm", "device": "cpu"})
        monkeypatch.setattr(
            gates, "_certificate_samples",
            lambda pipeline, model, n: torch.randn(n, 8),
        )
        with pytest.raises(RuntimeError, match="value twin certificate FAILED"):
            gates.run_value_twin_certificate_gate(pipeline, flow, ir, hybrid)

    def test_typed_skip_when_unarmed(self):
        pipeline = _pipeline({"core_semantics": "mvm", "value_parity_samples": 0})
        assert gates.run_value_twin_certificate_gate(
            pipeline, None, None, None
        ) is None
