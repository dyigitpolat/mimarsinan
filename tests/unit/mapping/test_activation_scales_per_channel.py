"""Per-channel (θ-cotrain) activation-scale tables for the NF↔SCM identity gate.

A ``ttfs_theta_cotrain`` node carries a 1-D ``activation_scale`` (len == out
features).  ``compute_node_output_scales`` must surface the per-channel vector
(not crash on ``float(tensor.item())`` or collapse to a scalar mean), while the
scalar path stays byte-identical to its prior behaviour.
"""

import numpy as np
import torch

from conftest import make_tiny_ir_graph

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.support.activation_scales import (
    compute_node_input_scales,
    compute_node_output_scales,
)


def _per_channel_ir_graph(node_id=0):
    """Tiny two-core graph with a per-output-channel activation_scale on one core."""
    ir = make_tiny_ir_graph()
    target = next(n for n in ir.nodes if isinstance(n, NeuralCore) and n.id == node_id)
    out_features = target.core_matrix.shape[1]
    scale = torch.arange(1, out_features + 1, dtype=torch.float32)
    target.activation_scale = scale
    return ir, node_id, scale


class TestComputeNodeOutputScalesPerChannel:
    def test_scalar_node_returns_python_float(self):
        ir = make_tiny_ir_graph()
        scales = compute_node_output_scales(ir)
        for node in ir.get_neural_cores():
            value = scales[node.id]
            assert isinstance(value, float)
            assert value == 1.0

    def test_per_channel_node_returns_vector(self):
        ir, node_id, scale = _per_channel_ir_graph()
        scales = compute_node_output_scales(ir)

        value = scales[node_id]
        assert isinstance(value, np.ndarray)
        assert value.shape == (scale.numel(),)
        np.testing.assert_allclose(value, scale.numpy())

    def test_single_element_scale_stays_scalar(self):
        """A 1-element activation_scale must behave exactly as the scalar path."""
        ir = make_tiny_ir_graph()
        node = next(iter(ir.get_neural_cores()))
        node.activation_scale = torch.tensor([3.0])
        scales = compute_node_output_scales(ir)
        assert isinstance(scales[node.id], float)
        assert scales[node.id] == 3.0

    def test_per_channel_not_collapsed_to_mean(self):
        ir, node_id, scale = _per_channel_ir_graph()
        scales = compute_node_output_scales(ir)
        value = scales[node_id]
        # Mean-collapse would yield a single scalar; per-channel must keep all entries.
        assert not np.isscalar(value)
        assert value.shape[0] == scale.numel()
        assert not np.allclose(value, float(scale.float().mean()))


class TestComputeNodeInputScalesPerChannel:
    def test_scalar_path_byte_identical(self):
        ir = make_tiny_ir_graph()
        in_scales = compute_node_input_scales(ir)
        for node in ir.get_neural_cores():
            assert isinstance(in_scales[node.id], float)


class TestComputeOpScalesStayScalar:
    """A ComputeOp's own in/out scale must stay a scalar even when its sources
    are per-channel: the hybrid executor (``resolve_stage_compute_scales``) casts
    it via ``float(...)``, and a value-preserving op needs a single cancelling
    factor (``in_scale == out_scale``)."""

    def _mixer_ir_with_per_channel_cores(self):
        from unit.mapping.test_identity_hybrid_mapping import (
            _make_mini_mixer_ir_graph,
        )

        ir, _ = _make_mini_mixer_ir_graph()
        for node in ir.get_neural_cores():
            node.activation_scale = torch.linspace(0.8, 1.2, node.get_output_count())
        return ir

    def test_compute_op_output_scale_scalar_with_per_channel_sources(self):
        ir = self._mixer_ir_with_per_channel_cores()
        out_scales = compute_node_output_scales(ir)
        in_scales = compute_node_input_scales(ir)
        compute_ops = [n for n in ir.nodes if not n.__class__.__name__ == "NeuralCore"]
        assert compute_ops, "mixer graph must contain a ComputeOp to exercise this"
        for op in compute_ops:
            assert isinstance(out_scales[op.id], float)
            assert isinstance(in_scales[op.id], float)
            assert in_scales[op.id] == out_scales[op.id]

    def test_neural_cores_keep_per_channel_in_mixer(self):
        ir = self._mixer_ir_with_per_channel_cores()
        out_scales = compute_node_output_scales(ir)
        for node in ir.get_neural_cores():
            assert isinstance(out_scales[node.id], np.ndarray)
            assert out_scales[node.id].shape == (node.get_output_count(),)


class TestIdentityMappingPerChannel:
    def test_build_identity_does_not_raise_on_per_channel(self):
        ir, node_id, scale = _per_channel_ir_graph()
        mapping = build_identity_hybrid_mapping(ir_graph=ir)
        stored = mapping.node_activation_scales[node_id]
        assert isinstance(stored, np.ndarray)
        np.testing.assert_allclose(stored, scale.numpy())

    def test_build_identity_scalar_graph_unchanged(self):
        ir = make_tiny_ir_graph()
        mapping = build_identity_hybrid_mapping(ir_graph=ir)
        for node in ir.get_neural_cores():
            assert isinstance(mapping.node_activation_scales[node.id], float)
            assert mapping.node_activation_scales[node.id] == 1.0


class TestCascadedIdentityExecutorPerChannel:
    """The cascaded NF↔SCM gate builds an identity spiking flow and runs the
    contract executor; a per-channel ComputeOp graph must not crash it (the
    pre-fix ``float(tensor.item())`` / ``float(vector)`` failure class)."""

    def test_per_channel_mixer_contract_runs(self):
        from mimarsinan.chip_simulation.ttfs.ttfs_executor import (
            run_ttfs_hybrid_contract,
        )

        from unit.mapping.test_identity_hybrid_mapping import (
            _make_mini_mixer_ir_graph,
        )

        ir, input_shape = _make_mini_mixer_ir_graph()
        for node in ir.get_neural_cores():
            node.activation_scale = torch.linspace(0.8, 1.2, node.get_output_count())

        mapping = build_identity_hybrid_mapping(ir_graph=ir)
        torch.manual_seed(3)
        x = torch.rand(1, *input_shape).reshape(1, -1).double().numpy()
        run = run_ttfs_hybrid_contract(
            mapping, x, simulation_length=4,
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized",
        )
        assert run.record.segments, "executor produced no segment records"
