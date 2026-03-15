"""Test that all model builders produce models that can be verified for soft-core mapping.

For each builder in BUILDERS_REGISTRY:
- Build the model with a minimal configuration.
- For native builders (category='native'): get mapper repr directly.
- For torch builders (category='torch'): convert to mapper repr via convert_torch_model.
- Run verify_soft_core_mapping and assert feasible=True.
- Run suggest_hardware_config and assert a valid config is returned.

Large models (VGG16, ViT, SqueezeNet) are marked slow.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mapping_verifier import verify_soft_core_mapping
from mimarsinan.mapping.hw_config_suggester import suggest_hardware_config
from mimarsinan.pipelining.model_registry import ModelRegistry


# ── Shared test helpers ─────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
PIPELINE_CONFIG = {"target_tq": 32, "device": "cpu"}


def _build_and_get_mapper_repr(model_type, input_shape, num_classes, model_config,
                                max_axons=1024, max_neurons=1024):
    """Build model using the registry builder and return its mapper repr."""
    from mimarsinan.models.builders import BUILDERS_REGISTRY
    from mimarsinan.torch_mapping.converter import convert_torch_model

    builder_cls = BUILDERS_REGISTRY[model_type]
    builder = builder_cls(
        device=DEVICE,
        input_shape=input_shape,
        num_classes=num_classes,
        max_axons=max_axons,
        max_neurons=max_neurons,
        pipeline_config=PIPELINE_CONFIG,
    )
    raw_model = builder.build(model_config)

    category = ModelRegistry.get_category(model_type)
    if category == "torch":
        raw_model.eval()
        with torch.no_grad():
            try:
                _ = raw_model(torch.randn(1, *input_shape))
            except Exception:
                pass
        supermodel = convert_torch_model(
            raw_model,
            input_shape=input_shape,
            num_classes=num_classes,
            device="cpu",
            Tq=32,
        )
        return supermodel.get_mapper_repr()
    else:
        # Native model: raw_model IS the Supermodel — run a dummy forward to
        # initialize LazyBatchNorm and other lazy modules before mapping.
        raw_model.eval()
        with torch.no_grad():
            try:
                _ = raw_model(torch.randn(2, *input_shape))
            except Exception:
                pass
        return raw_model.get_mapper_repr()


def _check_builder(model_type, input_shape, num_classes, model_config,
                   max_axons=1024, max_neurons=1024):
    """Full pipeline: build → get mapper repr → verify → suggest."""
    model_repr = _build_and_get_mapper_repr(
        model_type, input_shape, num_classes, model_config, max_axons, max_neurons
    )

    # Step 1: verify soft-core mapping
    result = verify_soft_core_mapping(
        model_repr,
        max_axons=max_axons,
        max_neurons=max_neurons,
    )
    assert result.feasible, (
        f"{model_type}: soft-core mapping failed: {result.error}"
    )
    assert result.num_neural_cores > 0, f"{model_type}: no neural cores"
    assert result.max_input_size > 0
    assert result.max_output_size > 0

    # Step 2: suggest hardware config
    suggestion = suggest_hardware_config(result.softcores)
    assert suggestion.total_cores > 0, f"{model_type}: suggestion has 0 cores"
    assert len(suggestion.core_types) >= 1, f"{model_type}: suggestion has no core types"

    return result, suggestion


# ══════════════════════════════════════════════════════════
# NATIVE BUILDERS
# ══════════════════════════════════════════════════════════

class TestSimpleMLPBuilder:
    def test_build_and_map(self):
        _check_builder(
            "simple_mlp",
            input_shape=(1, 28, 28),
            num_classes=10,
            model_config={"mlp_width_1": 64, "mlp_width_2": 32},
        )

    def test_softcores_count(self):
        result, _ = _check_builder(
            "simple_mlp",
            input_shape=(1, 8, 8),
            num_classes=4,
            model_config={"mlp_width_1": 16, "mlp_width_2": 8},
        )
        assert result.num_neural_cores >= 2  # at least input→hidden, hidden→output

    def test_suggestion_feasible_for_model(self):
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        result, suggestion = _check_builder(
            "simple_mlp",
            input_shape=(1, 28, 28),
            num_classes=10,
            model_config={"mlp_width_1": 128, "mlp_width_2": 64},
        )
        verification = verify_hardware_config(result.softcores, suggestion.core_types)
        assert verification["feasible"], (
            f"Suggested config not feasible: {verification['errors']}"
        )


# ══════════════════════════════════════════════════════════
# TORCH SEQUENTIAL BUILDERS
# ══════════════════════════════════════════════════════════

class TestTorchSequentialLinearBuilderMapping:
    def test_build_and_map(self):
        _check_builder(
            "torch_sequential_linear",
            input_shape=(1, 28, 28),
            num_classes=10,
            model_config={"hidden_dims": [64, 32]},
        )

    def test_two_hidden_layers(self):
        result, suggestion = _check_builder(
            "torch_sequential_linear",
            input_shape=(16,),
            num_classes=8,
            model_config={"hidden_dims": [32, 16]},
        )
        # With mm+mm fusion + Identity→ComputeOp, final classifier (no activation)
        # becomes a ComputeOp or fuses with the preceding layer.
        assert result.num_neural_cores >= 1

    def test_suggestion_sufficient(self):
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        result, suggestion = _check_builder(
            "torch_sequential_linear",
            input_shape=(1, 28, 28),
            num_classes=10,
            model_config={"hidden_dims": [128]},
        )
        v = verify_hardware_config(result.softcores, suggestion.core_types)
        assert v["feasible"]


class TestTorchSequentialConvBuilderMapping:
    def test_build_and_map(self):
        # Use smaller input/conv to keep flattened size < max_axons
        _check_builder(
            "torch_sequential_conv",
            input_shape=(1, 16, 16),
            num_classes=10,
            model_config={"conv_out_channels": 4, "hidden_dims": [32]},
            max_axons=2048,
            max_neurons=1024,
        )

    def test_produces_neural_cores_and_compute_ops(self):
        """Conv model should produce both neural cores (conv+fc) and at least 1 ComputeOp."""
        from mimarsinan.mapping.ir_mapping import IRMapping
        from mimarsinan.models.builders import BUILDERS_REGISTRY
        from mimarsinan.torch_mapping.converter import convert_torch_model

        builder = BUILDERS_REGISTRY["torch_sequential_conv"](
            device=DEVICE,
            input_shape=(1, 28, 28),
            num_classes=10,
            max_axons=1024,
            max_neurons=1024,
            pipeline_config=PIPELINE_CONFIG,
        )
        raw = builder.build({"conv_out_channels": 4, "hidden_dims": [32]})
        raw.eval()
        supermodel = convert_torch_model(raw, input_shape=(1, 28, 28), num_classes=10)
        mapper_repr = supermodel.get_mapper_repr()

        ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=1024, max_neurons=1024)
        ir_graph = ir.map(mapper_repr)
        assert len(ir_graph.get_neural_cores()) >= 2
        assert len(ir_graph.get_compute_ops()) >= 1  # MaxPool

    def test_suggestion_sufficient(self):
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        result, suggestion = _check_builder(
            "torch_sequential_conv",
            input_shape=(1, 16, 16),
            num_classes=10,
            model_config={"conv_out_channels": 4, "hidden_dims": [32]},
            max_axons=2048,
            max_neurons=1024,
        )
        v = verify_hardware_config(result.softcores, suggestion.core_types)
        assert v["feasible"]


class TestMlpMixerBuilderMapping:
    # Default wizard config: patch_c_1=32 != num_patches=16, catches BN absorption bugs
    _CONFIG = {
        "patch_n_1": 4,
        "patch_m_1": 4,
        "patch_c_1": 32,
        "fc_w_1": 64,
        "fc_w_2": 64,
    }

    def test_build_and_map(self):
        _check_builder(
            "mlp_mixer",
            input_shape=(1, 28, 28),
            num_classes=10,
            model_config=self._CONFIG,
            max_axons=2048,
            max_neurons=2048,
        )

    def test_produces_multiple_cores(self):
        result, _ = _check_builder(
            "mlp_mixer",
            input_shape=(1, 28, 28),
            num_classes=10,
            model_config=self._CONFIG,
            max_axons=2048,
            max_neurons=2048,
        )
        # MLP-Mixer has many FC layers so should produce many cores
        assert result.num_neural_cores >= 4

    def test_suggestion_sufficient(self):
        from mimarsinan.mapping.mapping_verifier import verify_hardware_config
        result, suggestion = _check_builder(
            "mlp_mixer",
            input_shape=(1, 28, 28),
            num_classes=10,
            model_config=self._CONFIG,
            max_axons=2048,
            max_neurons=2048,
        )
        v = verify_hardware_config(result.softcores, suggestion.core_types)
        assert v["feasible"]


# ══════════════════════════════════════════════════════════
# LARGE TORCH BUILDERS (marked slow — skip in fast CI)
# ══════════════════════════════════════════════════════════

@pytest.mark.slow
class TestTorchVGG16BuilderMapping:
    def test_build_succeeds(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY
        builder = BUILDERS_REGISTRY["torch_vgg16"](
            device=DEVICE,
            input_shape=(3, 32, 32),
            num_classes=10,
            max_axons=4096,
            max_neurons=4096,
            pipeline_config=PIPELINE_CONFIG,
        )
        model = builder.build({})
        assert isinstance(model, nn.Module)

    def test_representability(self):
        """VGG16 should be representable (Conv2d + BatchNorm + ReLU + MaxPool + Linear)."""
        from mimarsinan.models.builders import BUILDERS_REGISTRY
        from mimarsinan.torch_mapping.converter import check_representability
        builder = BUILDERS_REGISTRY["torch_vgg16"](
            device=DEVICE,
            input_shape=(3, 32, 32),
            num_classes=10,
            max_axons=4096,
            max_neurons=4096,
            pipeline_config=PIPELINE_CONFIG,
        )
        model = builder.build({})
        report = check_representability(model, input_shape=(3, 32, 32))
        assert report.is_representable, f"VGG16 not representable: {report.summary()}"

    def test_build_and_map(self):
        # VGG16's first FC layer has 512*7*7 = 25088 inputs; need max_axons >= 25088.
        _check_builder(
            "torch_vgg16",
            input_shape=(3, 32, 32),
            num_classes=10,
            model_config={},
            max_axons=25600,
            max_neurons=4096,
        )


@pytest.mark.slow
class TestTorchSqueezeNet11BuilderMapping:
    def test_build_succeeds(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY
        builder = BUILDERS_REGISTRY["torch_squeezenet11"](
            device=DEVICE,
            input_shape=(3, 32, 32),
            num_classes=10,
            max_axons=4096,
            max_neurons=4096,
            pipeline_config=PIPELINE_CONFIG,
        )
        model = builder.build({})
        assert isinstance(model, nn.Module)

    def test_representability(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY
        from mimarsinan.torch_mapping.converter import check_representability
        builder = BUILDERS_REGISTRY["torch_squeezenet11"](
            device=DEVICE,
            input_shape=(3, 32, 32),
            num_classes=10,
            max_axons=4096,
            max_neurons=4096,
            pipeline_config=PIPELINE_CONFIG,
        )
        model = builder.build({})
        report = check_representability(model, input_shape=(3, 32, 32))
        assert report.is_representable, f"SqueezeNet not representable: {report.summary()}"

    def test_build_and_map(self):
        _check_builder(
            "torch_squeezenet11",
            input_shape=(3, 32, 32),
            num_classes=10,
            model_config={},
            max_axons=4096,
            max_neurons=4096,
        )


@pytest.mark.slow
class TestTorchViTBuilderMapping:
    def test_build_succeeds(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY
        builder = BUILDERS_REGISTRY["torch_vit"](
            device=DEVICE,
            input_shape=(3, 32, 32),
            num_classes=10,
            max_axons=8192,
            max_neurons=8192,
            pipeline_config=PIPELINE_CONFIG,
        )
        model = builder.build({})
        assert isinstance(model, nn.Module)

    @pytest.mark.xfail(
        reason="torchvision ViT uses MultiheadAttention which has no mimarsinan mapping"
    )
    def test_representability(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY
        from mimarsinan.torch_mapping.converter import check_representability
        builder = BUILDERS_REGISTRY["torch_vit"](
            device=DEVICE,
            input_shape=(3, 32, 32),
            num_classes=10,
            max_axons=8192,
            max_neurons=8192,
            pipeline_config=PIPELINE_CONFIG,
        )
        model = builder.build({})
        report = check_representability(model, input_shape=(3, 32, 32))
        assert report.is_representable, f"ViT not representable: {report.summary()}"

    @pytest.mark.xfail(
        reason="torchvision ViT uses MultiheadAttention which has no mimarsinan mapping"
    )
    def test_build_and_map(self):
        _check_builder(
            "torch_vit",
            input_shape=(3, 32, 32),
            num_classes=10,
            model_config={},
            max_axons=8192,
            max_neurons=8192,
        )


# ══════════════════════════════════════════════════════════
# PARAMETRIC: Lightweight builders sanity check
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("model_type,input_shape,num_classes,config,max_ax,max_neu", [
    (
        "simple_mlp", (1, 8, 8), 4,
        {"mlp_width_1": 16, "mlp_width_2": 8}, 256, 256
    ),
    (
        "torch_sequential_linear", (16,), 4,
        {"hidden_dims": [16, 8]}, 256, 256
    ),
    (
        "torch_sequential_conv", (1, 16, 16), 4,
        {"conv_out_channels": 4, "hidden_dims": [16]}, 512, 512
    ),
    (
        "mlp_mixer", (1, 28, 28), 10,
        {"patch_n_1": 4, "patch_m_1": 4, "patch_c_1": 16, "fc_w_1": 32, "fc_w_2": 32},
        2048, 2048
    ),
])
def test_lightweight_builders_mapping(model_type, input_shape, num_classes, config, max_ax, max_neu):
    """Parametric test ensuring lightweight builders produce feasible mappings."""
    from mimarsinan.mapping.mapping_verifier import verify_hardware_config

    result, suggestion = _check_builder(
        model_type, input_shape, num_classes, config, max_ax, max_neu
    )
    assert result.feasible
    assert result.num_neural_cores > 0

    # Verify the suggestion is actually sufficient
    verification = verify_hardware_config(result.softcores, suggestion.core_types)
    assert verification["feasible"], (
        f"{model_type}: Suggested config not feasible: {verification['errors']}"
    )
