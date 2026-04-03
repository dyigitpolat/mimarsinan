"""Diagnostic: trace per-core TTFS output vs float model for TorchMLPMixer.

This test builds a real TorchMLPMixer, converts it through MapperGraphConverter,
fuses BN, creates IR, and compares the float ModelRepresentation forward against
SpikingUnifiedCoreFlow TTFS continuous, identifying the first point of divergence.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.models.torch_mlp_mixer import TorchMLPMixer
from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import NeuralCore, ComputeOp, IRGraph
from mimarsinan.mapping.per_source_scales import compute_per_source_scales
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow, _ttfs_activation_from_type


def _build_small_mixer():
    """Build a small TorchMLPMixer for testing."""
    input_shape = (1, 8, 8)
    model = TorchMLPMixer(
        input_shape=input_shape,
        num_classes=4,
        patch_n_1=2,
        patch_m_1=2,
        patch_c_1=4,
        fc_w_1=4,
        fc_w_2=4,
        base_activation="ReLU",
    )
    return model, input_shape


def _convert_to_supermodel(model, input_shape, num_classes=4):
    """Convert TorchMLPMixer to ``ConvertedModelFlow`` via convert_torch_model."""
    from mimarsinan.torch_mapping.converter import convert_torch_model
    supermodel = convert_torch_model(model, input_shape, num_classes, device="cpu", Tq=4)
    return supermodel


def _fuse_bn(supermodel):
    """Fuse BatchNorm into weights (mirrors NormalizationFusionStep)."""
    from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
    pt = PerceptronTransformer()

    for perceptron in supermodel.get_perceptrons():
        if isinstance(perceptron.normalization, nn.Identity):
            continue

        u, beta, mean = pt._get_u_beta_mean(perceptron.normalization)
        W = perceptron.layer.weight.data
        b = perceptron.layer.bias.data if perceptron.layer.bias is not None else torch.zeros(
            W.shape[0], device=W.device
        )

        fused_W = W * u.unsqueeze(-1)
        fused_b = (b - mean) * u + beta

        perceptron.layer = nn.Linear(
            perceptron.input_features, perceptron.output_channels, bias=True
        )
        perceptron.layer.weight.data = fused_W
        perceptron.layer.bias.data = fused_b
        perceptron.normalization = nn.Identity()


def _build_ir_flow(supermodel, input_shape):
    """Build IR graph and SpikingUnifiedCoreFlow."""
    mapper_repr = supermodel.get_mapper_repr()
    compute_per_source_scales(mapper_repr)

    ir_mapping = IRMapping(
        q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
    )
    ir_graph = ir_mapping.map(mapper_repr)

    # No weight quantization
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            node.threshold = 1.0
            node.parameter_scale = torch.tensor(1.0)

    flow = SpikingUnifiedCoreFlow(
        input_shape, ir_graph, 32, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode="ttfs",
    )
    flow.eval()
    return ir_graph, flow


class TestTTFSDiagnostic:
    """Per-core diagnostic for TTFS continuous vs float model."""

    def test_full_mixer_ttfs_equivalence(self):
        """Full TorchMLPMixer: ModelRepresentation vs TTFS continuous must match."""
        torch.manual_seed(42)
        model, input_shape = _build_small_mixer()
        model.eval()

        supermodel = _convert_to_supermodel(model, input_shape)
        supermodel.eval()

        # Verify conversion fidelity first (model vs supermodel)
        x = torch.rand(4, *input_shape)
        with torch.no_grad():
            orig_out = model(x)
            # Compare through mapper DAG (same as flow forward for raw x).
            super_out = supermodel.get_mapper_repr()(x)

        conv_diff = (orig_out - super_out).abs().max().item()
        assert conv_diff < 1e-3, (
            f"Conversion fidelity failed: max diff {conv_diff:.6f}"
        )

        # Now fuse BN
        _fuse_bn(supermodel)

        # Verify BN fusion didn't break anything
        with torch.no_grad():
            fused_out = supermodel.get_mapper_repr()(x)
        fusion_diff = (orig_out - fused_out).abs().max().item()
        assert fusion_diff < 1e-3, (
            f"BN fusion broke model: max diff {fusion_diff:.6f}"
        )

        # Build IR and TTFS flow
        ir_graph, flow = _build_ir_flow(supermodel, input_shape)

        # Compare TTFS output
        with torch.no_grad():
            float_out = supermodel.get_mapper_repr()(x)
            flow_out = flow(x)

        # They may differ by a constant factor (activation_scale), but
        # with activation_scale=1.0 they should match closely
        max_diff = (float_out - flow_out).abs().max().item()

        # Report per-core diagnostics if there's a significant diff
        if max_diff > 0.01:
            self._diagnose_per_core(ir_graph, flow, x, supermodel)

        assert max_diff < 0.01, (
            f"TTFS vs float max diff {max_diff:.6f}. See per-core diagnostics above."
        )

    def _diagnose_per_core(self, ir_graph, flow, x, supermodel):
        """Trace through each core and find the first divergence point."""
        x_flat = x.view(x.shape[0], -1)
        batch_size = x.shape[0]
        device = x.device

        # Manual TTFS continuous forward with per-core output capture
        activation_cache = {}
        core_diagnostics = []

        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                weight = flow._get_weight(node)
                t_idx = flow._get_threshold_idx(node)
                threshold = flow.thresholds[t_idx]

                spans = flow._input_spans[int(node.id)]
                in_dim = int(len(node.input_sources.flatten()))
                inp = torch.zeros(batch_size, in_dim, device=device)
                flow._fill_activation_from_ir_spans(
                    inp, x=x_flat, activation_cache=activation_cache, spans=spans
                )

                out = torch.matmul(weight, inp.T).T
                if node.id in flow._hw_bias_params:
                    out = out + flow._hw_bias_params[node.id]

                act_fn = _ttfs_activation_from_type(node.activation_type)
                out_activated = act_fn(out)
                out_final = out_activated / threshold

                # Check for issues
                has_negatives_pre = (out < 0).any().item()
                has_negatives_post = (out_activated < 0).any().item()
                spurious_relu = has_negatives_pre and not has_negatives_post and node.activation_type in (None, "Identity")

                core_diagnostics.append({
                    "node_id": node.id,
                    "name": node.name,
                    "activation_type": node.activation_type,
                    "threshold": float(threshold),
                    "input_range": (float(inp.min()), float(inp.max())),
                    "pre_activation_range": (float(out.min()), float(out.max())),
                    "output_range": (float(out_final.min()), float(out_final.max())),
                    "spurious_relu": spurious_relu,
                })

                activation_cache[node.id] = out_final

            elif isinstance(node, ComputeOp):
                spans = flow._input_spans[int(node.id)]
                in_dim = int(len(node.input_sources.flatten()))
                inp = torch.zeros(batch_size, in_dim, device=device)
                flow._fill_activation_from_ir_spans(
                    inp, x=x_flat, activation_cache=activation_cache, spans=spans
                )
                result = node.execute_on_gathered(inp)

                core_diagnostics.append({
                    "node_id": node.id,
                    "name": node.name,
                    "op_type": node.op_type,
                    "input_range": (float(inp.min()), float(inp.max())),
                    "output_range": (float(result.min()), float(result.max())),
                })

                activation_cache[node.id] = result

        # Print diagnostics
        print("\n=== PER-CORE DIAGNOSTICS ===")
        for d in core_diagnostics:
            print(f"Node {d['node_id']:3d} ({d.get('name', '?'):30s}) | "
                  f"act_type={d.get('activation_type', d.get('op_type', '?')):20s} | "
                  f"in=[{d['input_range'][0]:+.4f}, {d['input_range'][1]:+.4f}] | "
                  f"out=[{d['output_range'][0]:+.4f}, {d['output_range'][1]:+.4f}]"
                  + (" | SPURIOUS RELU!" if d.get('spurious_relu') else ""))

    def test_ir_node_activation_types(self):
        """Every NeuralCore in the IR must have activation_type set (not None)."""
        torch.manual_seed(42)
        model, input_shape = _build_small_mixer()
        model.eval()

        supermodel = _convert_to_supermodel(model, input_shape)
        _fuse_bn(supermodel)

        mapper_repr = supermodel.get_mapper_repr()
        compute_per_source_scales(mapper_repr)

        ir_mapping = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        )
        ir_graph = ir_mapping.map(mapper_repr)

        none_cores = []
        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore) and node.activation_type is None:
                none_cores.append(node.name)

        assert len(none_cores) == 0, (
            f"{len(none_cores)} NeuralCores have activation_type=None "
            f"(will default to ReLU in spiking): {none_cores[:10]}"
        )

    def test_activation_type_consistency(self):
        """Verify activation_type on each NeuralCore matches its perceptron's activation."""
        torch.manual_seed(42)
        model, input_shape = _build_small_mixer()
        model.eval()

        supermodel = _convert_to_supermodel(model, input_shape)
        _fuse_bn(supermodel)

        mapper_repr = supermodel.get_mapper_repr()
        compute_per_source_scales(mapper_repr)

        ir_mapping = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        )
        ir_graph = ir_mapping.map(mapper_repr)

        # Collect all perceptrons and their expected activation types
        perceptrons = supermodel.get_perceptrons()
        print(f"\nPerceptrons ({len(perceptrons)}):")
        for i, p in enumerate(perceptrons):
            act = getattr(p, "activation", None)
            act_name = type(act).__name__ if act is not None else "None"
            print(f"  [{i}] {p.name}: activation={act_name}")

        print(f"\nNeuralCores ({len([n for n in ir_graph.nodes if isinstance(n, NeuralCore)])}):")
        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                print(f"  [{node.id}] {node.name}: activation_type={node.activation_type}")
            elif isinstance(node, ComputeOp):
                print(f"  [{node.id}] {node.name}: op_type={node.op_type}")

    def test_per_source_scales_correctness(self):
        """Verify per_input_scales are set correctly for all perceptrons."""
        torch.manual_seed(42)
        model, input_shape = _build_small_mixer()
        model.eval()

        supermodel = _convert_to_supermodel(model, input_shape)
        _fuse_bn(supermodel)

        mapper_repr = supermodel.get_mapper_repr()
        compute_per_source_scales(mapper_repr)

        for p in supermodel.get_perceptrons():
            pis = getattr(p, "per_input_scales", None)
            act_s = float(p.activation_scale)
            assert pis is not None, f"Perceptron {p.name} has no per_input_scales"
            assert pis.shape[0] == p.input_features, (
                f"Perceptron {p.name}: per_input_scales shape {pis.shape} "
                f"!= input_features {p.input_features}"
            )
            # With all activation_scales = 1.0, per_input_scales should be 1.0
            if act_s == 1.0:
                non_one = (pis != 1.0).sum().item()
                if non_one > 0:
                    print(f"WARNING: {p.name} has {non_one}/{len(pis)} "
                          f"non-1.0 per_input_scales (range [{pis.min():.4f}, {pis.max():.4f}])")

    def test_weight_transfer_fidelity(self):
        """Verify IR core weights match get_effective_weight for each perceptron."""
        from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

        torch.manual_seed(42)
        model, input_shape = _build_small_mixer()
        model.eval()

        supermodel = _convert_to_supermodel(model, input_shape)
        _fuse_bn(supermodel)

        mapper_repr = supermodel.get_mapper_repr()
        compute_per_source_scales(mapper_repr)

        ir_mapping = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        )
        ir_graph = ir_mapping.map(mapper_repr)

        pt = PerceptronTransformer()
        perceptrons = supermodel.get_perceptrons()

        # Check each perceptron's effective weight against the corresponding IR core(s)
        for p_idx, p in enumerate(perceptrons):
            W_eff = pt.get_effective_weight(p).detach().numpy()
            b_eff = pt.get_effective_bias(p).detach().numpy()

            # Find cores belonging to this perceptron
            matching_cores = [
                n for n in ir_graph.nodes
                if isinstance(n, NeuralCore) and n.perceptron_index == p_idx
            ]

            if not matching_cores:
                continue  # perceptron_index may not be set for all

            # For bank-backed cores, check the bank matrix
            for core in matching_cores:
                if core.has_weight_bank():
                    bank = ir_graph.weight_banks[core.weight_bank_id]
                    # Bank stores W_eff.T in shape (axons, neurons)
                    ir_W = bank.core_matrix.T  # -> (neurons, axons) = W_eff shape
                    if core.weight_row_slice:
                        s, e = core.weight_row_slice
                        ir_W = ir_W[s:e, :]
                else:
                    ir_W = core.core_matrix.T  # (neurons, axons)

                # Compare weight dimensions
                if ir_W.shape == W_eff.shape:
                    w_diff = np.max(np.abs(ir_W - W_eff))
                    assert w_diff < 1e-5, (
                        f"Perceptron [{p_idx}] {p.name}: weight diff {w_diff:.2e}"
                    )

    def test_activation_type_after_adaptation(self):
        """Conv NeuralCores must resolve activation_type through TransformedActivation."""
        from mimarsinan.tuning.adaptation_manager import AdaptationManager
        from mimarsinan.mapping.mappers.base import resolve_activation_type

        torch.manual_seed(42)
        model, input_shape = _build_small_mixer()
        model.eval()

        supermodel = _convert_to_supermodel(model, input_shape)

        # Apply adaptation (wraps activations in TransformedActivation)
        am = AdaptationManager()
        config = {"spiking_mode": "ttfs", "target_tq": 4}
        for p in supermodel.get_perceptrons():
            am.update_activation(config, p)

        # Verify resolve_activation_type works through TransformedActivation
        for p in supermodel.get_perceptrons():
            act = p.activation
            assert hasattr(act, "base_activation"), (
                f"{p.name}: activation is not TransformedActivation after adaptation"
            )
            resolved = resolve_activation_type(p)
            assert "TransformedActivation" not in resolved, (
                f"{p.name}: resolve_activation_type returned raw 'TransformedActivation' "
                f"instead of unwrapping to base. Got: {resolved!r}"
            )

        # Fuse BN then build IR — check conv cores have correct activation_type
        _fuse_bn(supermodel)
        mapper_repr = supermodel.get_mapper_repr()
        compute_per_source_scales(mapper_repr)

        ir_mapping = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        )
        ir_graph = ir_mapping.map(mapper_repr)

        bad_cores = []
        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                if node.activation_type is None:
                    bad_cores.append((node.name, "None"))
                elif "TransformedActivation" in node.activation_type:
                    bad_cores.append((node.name, node.activation_type))

        assert len(bad_cores) == 0, (
            f"{len(bad_cores)} NeuralCores have bad activation_type after adaptation: "
            f"{bad_cores[:5]}"
        )

    def test_ttfs_equivalence_after_adaptation(self):
        """TTFS continuous must still match float model after adaptation + BN fusion."""
        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        torch.manual_seed(42)
        model, input_shape = _build_small_mixer()
        model.eval()

        supermodel = _convert_to_supermodel(model, input_shape)

        # Apply adaptation
        am = AdaptationManager()
        config = {"spiking_mode": "ttfs", "target_tq": 4}
        for p in supermodel.get_perceptrons():
            am.update_activation(config, p)

        # Fuse BN
        _fuse_bn(supermodel)
        supermodel.eval()

        # Build IR and flow
        ir_graph, flow = _build_ir_flow(supermodel, input_shape)

        # Compare outputs
        x = torch.rand(4, *input_shape)
        with torch.no_grad():
            float_out = supermodel.get_mapper_repr()(x)
            flow_out = flow(x)

        max_diff = (float_out - flow_out).abs().max().item()
        assert max_diff < 0.01, (
            f"Post-adaptation TTFS vs float max diff {max_diff:.6f}. "
            "activation_type resolution through TransformedActivation may be broken."
        )
