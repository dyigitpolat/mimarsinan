"""
Stress tests for the IR module.

Tests edge cases in NeuralCore execution, IRGraph validation, and ComputeOp.
"""

import pytest
import torch
import numpy as np

from mimarsinan.mapping.ir import IRSource, IRGraph, NeuralCore, ComputeOp


class TestNeuralCoreExecution:
    """Tests that probe NeuralCore.execute for numerical edge cases."""

    def test_all_zero_weights_produces_zero(self):
        """Zero weights with nonzero input should produce zero output."""
        w = np.zeros((2, 3), dtype=np.float32)
        sources = np.array([IRSource(-2, 0), IRSource(-2, 1)], dtype=object)
        core = NeuralCore(id=0, name="zero", input_sources=sources, core_matrix=w)
        x = torch.tensor([[5.0, 10.0]])
        out = core.execute(x, {})
        assert (out == 0).all()

    def test_threshold_field_is_ignored(self):
        """
        DESIGN ISSUE: NeuralCore.execute does not use the threshold field.
        Changing threshold should not affect output.
        """
        w = np.array([[1.0], [1.0]], dtype=np.float32)
        sources = np.array([IRSource(-2, 0), IRSource(-2, 1)], dtype=object)

        core_t1 = NeuralCore(id=0, name="c", input_sources=sources,
                             core_matrix=w, threshold=1.0)
        core_t100 = NeuralCore(id=0, name="c", input_sources=sources,
                               core_matrix=w, threshold=100.0)

        x = torch.tensor([[2.0, 3.0]])
        out_t1 = core_t1.execute(x, {})
        out_t100 = core_t100.execute(x, {})

        assert torch.allclose(out_t1, out_t100), \
            "Threshold field has no effect on execute(). " \
            "If this is intentional (threshold only used in spiking sim), " \
            "this is fine. If it should affect output, this is a bug."

    def test_activation_scale_clamps_output(self):
        """Output should be clamped to [0, activation_scale]."""
        w = np.array([[10.0]], dtype=np.float32)
        sources = np.array([IRSource(-2, 0)], dtype=object)
        core = NeuralCore(id=0, name="c", input_sources=sources,
                          core_matrix=w, activation_scale=torch.tensor(2.0))
        x = torch.tensor([[5.0]])
        out = core.execute(x, {})
        assert out.item() <= 2.0 + 1e-5, \
            "Output should be clamped to activation_scale"

    def test_large_weight_matrix(self):
        """Stress test with a large core."""
        n_axons, n_neurons = 256, 128
        w = np.random.randn(n_axons, n_neurons).astype(np.float32) * 0.1
        sources = np.array([IRSource(-2, i) for i in range(n_axons)], dtype=object)
        core = NeuralCore(id=0, name="big", input_sources=sources, core_matrix=w)
        x = torch.randn(8, n_axons)
        out = core.execute(x, {})
        assert out.shape == (8, n_neurons)
        assert not torch.isnan(out).any()

    def test_single_neuron_single_axon(self):
        """Minimal core: 1 input, 1 output."""
        w = np.array([[3.0]], dtype=np.float32)
        sources = np.array([IRSource(-2, 0)], dtype=object)
        core = NeuralCore(id=0, name="tiny", input_sources=sources, core_matrix=w)
        x = torch.tensor([[2.0]])
        out = core.execute(x, {})
        # W^T @ x = 3 * 2 = 6, then relu(6) = 6, then clamp to activation_scale (default 1.0)
        assert out.item() == pytest.approx(1.0, abs=1e-5), \
            "Default activation_scale=1.0 should clamp output to 1.0"

    def test_negative_input_to_negative_weight(self):
        """Negative * negative = positive, which should pass through ReLU."""
        w = np.array([[-2.0]], dtype=np.float32)
        sources = np.array([IRSource(-2, 0)], dtype=object)
        core = NeuralCore(id=0, name="c", input_sources=sources,
                          core_matrix=w, activation_scale=torch.tensor(100.0))
        x = torch.tensor([[-3.0]])
        out = core.execute(x, {})
        # W^T @ x = -2 * -3 = 6, relu(6) = 6
        assert out.item() == pytest.approx(6.0, abs=1e-5)


class TestComputeOpStress:
    def test_add_mismatched_halves(self):
        """What happens if input size is odd (half_size doesn't divide evenly)?"""
        sources = np.array([IRSource(-2, i) for i in range(5)], dtype=object)
        op = ComputeOp(id=0, name="add", input_sources=sources,
                       op_type="add", params={"half_size": 2})
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        # a = x[:, :2] = [1, 2], b = x[:, 2:] = [3, 4, 5]
        # a + b will fail due to shape mismatch
        with pytest.raises(RuntimeError):
            op.execute_on_gathered(x)

    def test_max_pool2d_all_negative(self):
        """MaxPool with all-negative values should keep the least negative."""
        sources = np.array([IRSource(-2, i) for i in range(4)], dtype=object)
        op = ComputeOp(id=0, name="pool", input_sources=sources,
                       op_type="max_pool2d",
                       input_shape=(1, 2, 2),
                       params={"kernel_size": 2, "stride": 2, "padding": 0})
        x = torch.tensor([[-4.0, -3.0, -2.0, -1.0]])
        out = op.execute_on_gathered(x)
        assert out.item() == pytest.approx(-1.0)

    def test_layer_norm_known_values(self):
        """LayerNorm with known weights should produce hand-computable output."""
        dim = 4
        sources = np.array([IRSource(-2, i) for i in range(dim)], dtype=object)
        weight = [1.0] * dim
        bias = [0.0] * dim
        op = ComputeOp(
            id=0, name="ln", input_sources=sources,
            op_type="layer_norm",
            params={
                "weight": weight, "bias": bias,
                "normalized_shape": [dim], "eps": 1e-5,
            })
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = op.execute_on_gathered(x)
        expected = torch.nn.functional.layer_norm(x, [dim])
        assert torch.allclose(out, expected, atol=1e-4)

    def test_gelu_known_values(self):
        """GELU(0) = 0, GELU(large_positive) ≈ large_positive."""
        sources = np.array([IRSource(-2, 0), IRSource(-2, 1)], dtype=object)
        op = ComputeOp(id=0, name="gelu", input_sources=sources, op_type="gelu")
        x = torch.tensor([[0.0, 10.0]])
        out = op.execute_on_gathered(x)
        assert out[0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert out[0, 1].item() == pytest.approx(10.0, abs=0.01)

    @pytest.mark.xfail(
        reason="BUG: ComputeOp identity op crashes on empty batch. "
               "x.view(x.shape[0], -1) cannot reshape 0-element tensor "
               "because the -1 dimension is ambiguous with 0 elements.",
        strict=True,
        raises=RuntimeError,
    )
    def test_empty_input_tensor(self):
        """Empty batch dimension should propagate without crashing."""
        sources = np.array([IRSource(-2, 0)], dtype=object)
        op = ComputeOp(id=0, name="id", input_sources=sources, op_type="identity")
        x = torch.empty(0, 1)
        out = op.execute_on_gathered(x)
        assert out.shape[0] == 0


class TestIRGraphValidation:
    def test_duplicate_node_ids(self):
        """Two nodes with the same ID — validate should flag this or handle it."""
        w = np.ones((1, 1), dtype=np.float32)
        s = np.array([IRSource(-2, 0)], dtype=object)
        c1 = NeuralCore(id=0, name="a", input_sources=s, core_matrix=w)
        c2 = NeuralCore(id=0, name="b", input_sources=s, core_matrix=w)
        out = np.array([IRSource(0, 0)], dtype=object)
        g = IRGraph(nodes=[c1, c2], output_sources=out)

        # This is a degenerate case. validate() may or may not catch it.
        errors = g.validate()
        # At minimum, get_node_by_id should return one of them
        node = g.get_node_by_id(0)
        assert node is not None

    def test_cyclic_reference(self):
        """Node A references node B, node B references node A."""
        w = np.ones((1, 1), dtype=np.float32)
        s_a = np.array([IRSource(1, 0)], dtype=object)
        s_b = np.array([IRSource(0, 0)], dtype=object)
        a = NeuralCore(id=0, name="a", input_sources=s_a, core_matrix=w)
        b = NeuralCore(id=1, name="b", input_sources=s_b, core_matrix=w)
        out = np.array([IRSource(1, 0)], dtype=object)
        g = IRGraph(nodes=[a, b], output_sources=out)

        # validate() doesn't check for cycles currently
        errors = g.validate()
        # Just document that cycles aren't detected
        assert isinstance(errors, list)

    def test_output_references_nonexistent_node(self):
        """Output sources reference a node that doesn't exist."""
        w = np.ones((1, 1), dtype=np.float32)
        s = np.array([IRSource(-2, 0)], dtype=object)
        c = NeuralCore(id=0, name="c", input_sources=s, core_matrix=w)
        out = np.array([IRSource(99, 0)], dtype=object)
        g = IRGraph(nodes=[c], output_sources=out)
        errors = g.validate()
        assert len(errors) > 0, "Should detect output referencing non-existent node"
