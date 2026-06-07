"""Per-node value recorder for the cascaded TTFS segment driver.

The recorder is both the NF-side capture for the cascaded NF↔SCM parity gate
and the bisect instrument: every perceptron node's decoded value must equal the
identity-mapped contract executor's per-core outputs (the deployed semantics).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward

S = 4


def _convert(model, input_shape, num_classes):
    from mimarsinan.torch_mapping.converter import convert_torch_model
    return convert_torch_model(model, input_shape, num_classes, device="cpu")


def _install_ttfs(flow):
    for p in flow.get_perceptrons():
        p.set_activation(TTFSActivation(
            T=S,
            activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale,
            bias=p.layer.bias,
            thresholding_mode="<=",
            encoding=getattr(p, "is_encoding_layer", False),
        ))
    return flow.double()


class _TinyMLP(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(d_h, d_out)
        self.act2 = nn.ReLU()

    def forward(self, x):
        return self.act2(self.fc2(self.act1(self.fc1(x))))


def _identity_mapping_for(flow):
    from mimarsinan.mapping.ir import NeuralCore
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.latency.ir import IRLatency
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        build_identity_hybrid_mapping,
    )
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    ir_graph = IRMapping(
        q_max=1, firing_mode="TTFS", max_axons=2048, max_neurons=2048,
    ).map(repr_)
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            node.threshold = 1.0
            node.parameter_scale = torch.tensor(1.0)
    IRLatency(ir_graph).calculate()
    return build_identity_hybrid_mapping(ir_graph=ir_graph)


def _grouped_driver_values(flow, node_values):
    perceptrons = flow.get_perceptrons()
    grouped = {}
    for node, value in node_values.items():
        p = getattr(node, "perceptron", None)
        if p is None:
            continue
        pi = perceptrons.index(p)
        scale = torch.as_tensor(p.activation_scale).double().clamp(min=1e-12)
        grouped[pi] = (value.double().reshape(-1) / scale).numpy()
    return grouped


class TestRecorderMechanics:
    def test_records_every_perceptron_node_and_output_matches(self):
        torch.manual_seed(0)
        flow = _install_ttfs(_convert(_TinyMLP(6, 5, 4), (6,), 4))
        driver = TTFSSegmentForward(flow.get_mapper_repr(), S)
        x = torch.rand(1, 6, dtype=torch.float64)
        out, node_values = driver.forward_with_node_values(x)
        recorded = _grouped_driver_values(flow, node_values)
        assert set(recorded) == {0, 1}
        # The driver's plain forward is unchanged by recording.
        with torch.no_grad():
            torch.testing.assert_close(out, driver(x), rtol=0, atol=0)

    def test_recorder_off_by_default(self):
        torch.manual_seed(0)
        flow = _install_ttfs(_convert(_TinyMLP(6, 5, 4), (6,), 4))
        driver = TTFSSegmentForward(flow.get_mapper_repr(), S)
        assert driver._driver.policy.node_value_recorder is None


class TestCascadedDriverExecutorParity:
    """The R8 lock: the NF driver equals the GENUINE identity-mapped cascade
    executor bit-for-bit in clean (float64, unquantized) arithmetic.

    NOTE the reference: the contract runner's record for ``ttfs_cycle_based``
    is the ANALYTICAL staircase composition (a separate SANA-FE reference) —
    the greedy cascade legitimately fires early relative to it. The deployed
    dynamics live in the hybrid flow's cascade executor, compared here.
    """

    def _assert_final_outputs_equal(self, flow, input_shape, x):
        import torch.nn as nn
        from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

        mapping = _identity_mapping_for(flow)
        driver = TTFSSegmentForward(flow.get_mapper_repr(), S)
        executor = SpikingHybridCoreFlow(
            input_shape, mapping, S, nn.Identity(), "TTFS", "TTFS", "<=",
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded",
        ).eval()
        with torch.no_grad():
            nf_out = driver(x)
            scm_out = executor(x).double() / S  # count-scaled -> value domain
        np.testing.assert_allclose(
            nf_out.numpy(), scm_out.numpy(), rtol=0, atol=1e-9,
            err_msg="cascaded NF driver diverges from the deployed executor",
        )

    def test_single_segment_mlp_offload(self):
        from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

        torch.manual_seed(0)
        flow = _convert(_TinyMLP(8, 6, 4), (8,), 4)
        mark_encoding_layers(flow.get_mapper_repr(), placement="offload")
        flow = _install_ttfs(flow)
        x = torch.rand(4, 8, dtype=torch.float64)
        self._assert_final_outputs_equal(flow, (8,), x)

    def test_single_segment_mlp_subsume(self):
        torch.manual_seed(0)
        flow = _install_ttfs(_convert(_TinyMLP(8, 6, 4), (8,), 4))
        x = torch.rand(4, 8, dtype=torch.float64)
        self._assert_final_outputs_equal(flow, (8,), x)

    def test_layer_replacement_with_refresh_keeps_parity(self):
        """The 2026-06-07 offload incident class: a step that REPLACES
        ``perceptron.layer`` (normalization fusion / bring_back_bias) with a
        different bias orphans ``TTFSActivation._bias``; the driver then
        subtracts a stale bias and diverges from the deployed executor. The
        replacement seams must call ``refresh_perceptron_bias_references``."""
        import torch.nn as nn
        from mimarsinan.models.nn.activations.ttfs_spiking import (
            refresh_perceptron_bias_references,
        )
        from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

        torch.manual_seed(0)
        flow = _convert(_TinyMLP(8, 6, 4), (8,), 4)
        mark_encoding_layers(flow.get_mapper_repr(), placement="offload")
        flow = _install_ttfs(flow)
        # mimic fusion: replace p0.layer with a new Linear carrying a shifted bias
        p0 = flow.get_perceptrons()[0]
        new_layer = nn.Linear(8, 6).double()
        with torch.no_grad():
            new_layer.weight.copy_(p0.layer.weight)
            new_layer.bias.copy_(p0.layer.bias + 0.4)
        p0.layer = new_layer
        refresh_perceptron_bias_references(p0)
        # IR is built from the post-replacement model (as the pipeline does)
        x = torch.rand(4, 8, dtype=torch.float64)
        self._assert_final_outputs_equal(flow, (8,), x)

    def test_multisegment_mixer_offload_nonunit_scales(self):
        from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
        from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

        torch.manual_seed(0)
        m = TorchMLPMixerCore(
            input_shape=(1, 28, 28), num_classes=10,
            patch_n_1=4, patch_m_1=4, patch_c_1=6, fc_w_1=8, fc_w_2=6,
        ).eval()
        flow = _convert(m, (1, 28, 28), 10)
        mark_encoding_layers(flow.get_mapper_repr(), placement="offload")
        for i, p in enumerate(flow.get_perceptrons()):
            p.set_activation_scale(1.5 + 0.5 * (i % 3))
        flow = _install_ttfs(flow)
        x = torch.rand(4, 1, 28, 28, dtype=torch.float64)
        self._assert_final_outputs_equal(flow, (1, 28, 28), x)
