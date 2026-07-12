"""Unit tests for TorchMLPMixerCore and TorchMLPMixerCoreBuilder."""

import pytest
import torch

from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
from mimarsinan.models.builders.torch_mlp_mixer_core_builder import TorchMLPMixerCoreBuilder
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
import torch.nn as nn


class TestTorchMLPMixerCore:
    def test_forward_shape(self):
        model = TorchMLPMixerCore(
            input_shape=(3, 32, 32),
            num_classes=10,
            patch_n_1=4,
            patch_m_1=4,
            patch_c_1=32,
            fc_w_1=64,
            fc_w_2=64,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_mnist_shape(self):
        model = TorchMLPMixerCore(
            input_shape=(1, 28, 28),
            num_classes=10,
            patch_n_1=4,
            patch_m_1=4,
            patch_c_1=16,
            fc_w_1=32,
            fc_w_2=32,
        )
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)


class TestTorchMLPMixerCoreConversion:
    """Verify conversion fidelity and that all mixer FCs are chip-packaged perceptrons."""

    def test_converted_flow_forward_matches_original(self):
        """Converted flow output must numerically match the original model."""
        from mimarsinan.torch_mapping.converter import convert_torch_model

        model = TorchMLPMixerCore(
            input_shape=(1, 28, 28),
            num_classes=10,
            patch_n_1=4,
            patch_m_1=4,
            patch_c_1=32,
            fc_w_1=64,
            fc_w_2=64,
        )
        model.eval()
        flow = convert_torch_model(model, input_shape=(1, 28, 28), num_classes=10)
        flow.eval()

        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            orig_out = model(x)
            conv_out = flow(x)

        assert orig_out.shape == conv_out.shape
        assert torch.allclose(orig_out, conv_out, atol=1e-3), (
            f"Output mismatch — max diff: {(orig_out - conv_out).abs().max().item():.6f}. "
            "The converted flow does not faithfully reproduce the original model."
        )

    def test_all_mixer_fc_perceptrons_chip_supported(self):
        """Every mixer FC (8 total) plus the patch embed must have chip-supported activation.

        get_perceptrons() includes: patch embed (Conv mapper absorbs BN+ReLU, chip-supported),
        then 8 mixer FCs (all chip-supported for Core). Classifier is Identity and excluded by
        PerceptronMapper.owned_perceptron_groups (not chip-supported). So we expect 9 total,
        all 9 chip-supported.
        """
        from mimarsinan.torch_mapping.converter import convert_torch_model

        model = TorchMLPMixerCore(
            input_shape=(1, 28, 28),
            num_classes=10,
            patch_n_1=4,
            patch_m_1=4,
            patch_c_1=16,
            fc_w_1=32,
            fc_w_2=32,
        )
        model.eval()
        flow = convert_torch_model(model, input_shape=(1, 28, 28), num_classes=10)
        perceptrons = flow.get_perceptrons()

        assert len(perceptrons) == 9, (
            f"Expected 9 perceptrons (1 patch + 8 mixer FCs). Got {len(perceptrons)}."
        )
        # All perceptrons in get_perceptrons() have real activations (not Identity)
        for p in perceptrons:
            assert not isinstance(p.base_activation, nn.Identity), (
                f"Perceptron {p.name} should not have Identity activation"
            )


class TestTorchMLPMixerCoreBuilder:
    def test_build_returns_module(self):
        builder = TorchMLPMixerCoreBuilder(
            device=torch.device("cpu"),
            input_shape=(3, 32, 32),
            num_classes=10,
            pipeline_config={"target_tq": 32},
        )
        config = {
            "patch_n_1": 4,
            "patch_m_1": 4,
            "patch_c_1": 32,
            "fc_w_1": 64,
            "fc_w_2": 64,
        }
        model = builder.build(config)
        assert isinstance(model, torch.nn.Module)
        assert isinstance(model, TorchMLPMixerCore)

    def test_build_output_shape(self):
        builder = TorchMLPMixerCoreBuilder(
            device=torch.device("cpu"),
            input_shape=(3, 32, 32),
            num_classes=10,
            pipeline_config={"target_tq": 32},
        )
        config = {
            "patch_n_1": 4,
            "patch_m_1": 4,
            "patch_c_1": 32,
            "fc_w_1": 64,
            "fc_w_2": 64,
        }
        model = builder.build(config)
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_mlp_mixer_core_registered_torch_category(self):
        assert ModelRegistry.get_category("mlp_mixer_core") == "torch"

    def test_get_nas_search_options(self):
        opts = TorchMLPMixerCoreBuilder.get_nas_search_options((1, 28, 28))
        assert "patch_n_1" in opts
        assert "fc_w_1" in opts
        assert 4 in opts["patch_n_1"]

    def test_validate_config(self):
        assert TorchMLPMixerCoreBuilder.validate_config(
            {"patch_n_1": 4, "patch_m_1": 4},
            {},
            (1, 28, 28),
        ) is True
        assert TorchMLPMixerCoreBuilder.validate_config(
            {"patch_n_1": 5, "patch_m_1": 4},
            {},
            (1, 28, 28),
        ) is False


# ── normalization option ────────────────────────────────────────────────────

_BASE_CONFIG = {
    "base_activation": "ReLU",
    "patch_n_1": 4,
    "patch_m_1": 4,
    "patch_c_1": 32,
    "fc_w_1": 64,
    "fc_w_2": 64,
}

# Default-build state-dict keys pinned BEFORE the normalization option existed;
# normalization="none" (and a missing key) must reproduce them exactly.
_PINNED_DEFAULT_STATE_DICT_KEYS = [
    "patch_embed.weight",
    "patch_embed.bias",
    "patch_bn.weight",
    "patch_bn.bias",
    "patch_bn.running_mean",
    "patch_bn.running_var",
    "patch_bn.num_batches_tracked",
    "mixer_blocks.0.fc1.weight",
    "mixer_blocks.0.fc1.bias",
    "mixer_blocks.0.fc2.weight",
    "mixer_blocks.0.fc2.bias",
    "mixer_blocks.1.fc1.weight",
    "mixer_blocks.1.fc1.bias",
    "mixer_blocks.1.fc2.weight",
    "mixer_blocks.1.fc2.bias",
    "mixer_blocks.2.fc1.weight",
    "mixer_blocks.2.fc1.bias",
    "mixer_blocks.2.fc2.weight",
    "mixer_blocks.2.fc2.bias",
    "mixer_blocks.3.fc1.weight",
    "mixer_blocks.3.fc1.bias",
    "mixer_blocks.3.fc2.weight",
    "mixer_blocks.3.fc2.bias",
    "classifier.weight",
    "classifier.bias",
]


def _build_mixer(config=None, seed=0):
    builder = TorchMLPMixerCoreBuilder(
        device=torch.device("cpu"),
        input_shape=(1, 28, 28),
        num_classes=10,
        pipeline_config={"target_tq": 8},
    )
    torch.manual_seed(seed)
    return builder.build({**_BASE_CONFIG, **(config or {})})


class TestTorchMLPMixerCoreNormalizationOption:
    def test_default_state_dict_structure_is_pinned(self):
        model = _build_mixer()
        assert list(model.state_dict().keys()) == _PINNED_DEFAULT_STATE_DICT_KEYS

    def test_none_option_matches_missing_key_exactly(self):
        default_model = _build_mixer()
        none_model = _build_mixer({"normalization": "none"})
        default_sd = default_model.state_dict()
        none_sd = none_model.state_dict()
        assert list(default_sd.keys()) == list(none_sd.keys())
        for key in default_sd:
            assert torch.equal(default_sd[key], none_sd[key]), key

    def test_unknown_normalization_value_fails_loud(self):
        with pytest.raises(ValueError, match="normalization"):
            _build_mixer({"normalization": "layer"})

    def test_batch_installs_channels_last_bn_on_every_mixing_fc(self):
        from mimarsinan.models.nn.layers import ChannelsLastBatchNorm1d
        from mimarsinan.models.torch_mlp_mixer_core import (
            _ChannelMixerCore,
            _TokenMixerCore,
        )

        model = _build_mixer({"normalization": "batch"})
        num_patches = 16
        assert len(model.mixer_blocks) == 4
        for block in model.mixer_blocks:
            assert isinstance(block.bn1, ChannelsLastBatchNorm1d)
            assert isinstance(block.bn2, ChannelsLastBatchNorm1d)
            if isinstance(block, _TokenMixerCore):
                assert block.bn1.num_features == _BASE_CONFIG["fc_w_1"]
                assert block.bn2.num_features == num_patches
            else:
                assert isinstance(block, _ChannelMixerCore)
                assert block.bn1.num_features == _BASE_CONFIG["fc_w_2"]
                assert block.bn2.num_features == _BASE_CONFIG["patch_c_1"]

    def test_batch_state_dict_is_default_plus_bn_keys(self):
        model = _build_mixer({"normalization": "batch"})
        keys = list(model.state_dict().keys())
        bn_keys = [k for k in keys if ".bn1." in k or ".bn2." in k]
        assert [k for k in keys if k not in bn_keys] == _PINNED_DEFAULT_STATE_DICT_KEYS
        assert len(bn_keys) == 4 * 2 * 5  # 4 blocks x 2 BNs x 5 BN entries

    def test_batch_forward_shape(self):
        model = _build_mixer({"normalization": "batch"})
        x = torch.randn(4, 1, 28, 28)
        model.train()
        assert model(x).shape == (4, 10)
        model.eval()
        with torch.no_grad():
            assert model(x).shape == (4, 10)

    def test_batch_train_step_smoke_loss_decreases(self):
        torch.manual_seed(0)
        model = _build_mixer({"normalization": "batch"})
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        losses = []
        for _ in range(8):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


class TestTorchMLPMixerCoreNormalizationConversion:
    """The BN'd mixer must convert with BN carried per-perceptron and NF-fold
    back to a mapping-equivalent norm-free staircase chain."""

    def _converted_flow(self, normalization, seed=0):
        from mimarsinan.torch_mapping.converter import convert_torch_model

        model = _build_mixer({"normalization": normalization}, seed=seed)
        # Move BN running stats off their init point so folding is non-trivial.
        model.train()
        for _ in range(2):
            model(torch.randn(8, 1, 28, 28))
        model.eval()
        flow = convert_torch_model(model, input_shape=(1, 28, 28), num_classes=10)
        flow.eval()
        return model, flow

    def test_bn_mixer_perceptrons_carry_channels_last_bn(self):
        from mimarsinan.models.nn.layers import ChannelsLastBatchNorm1d

        _, flow = self._converted_flow("batch")
        perceptrons = flow.get_perceptrons()
        assert len(perceptrons) == 9
        mixing = [
            p for p in perceptrons
            if isinstance(p.normalization, ChannelsLastBatchNorm1d)
        ]
        assert len(mixing) == 8, (
            "every mixing FC perceptron must carry the channels-last BN, got "
            f"{[type(p.normalization).__name__ for p in perceptrons]}"
        )

    def test_bn_converted_flow_matches_original(self):
        model, flow = self._converted_flow("batch")
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            orig_out = model(x)
            conv_out = flow(x)
        assert torch.allclose(orig_out, conv_out, atol=1e-3), (
            f"max diff: {(orig_out - conv_out).abs().max().item():.6f}"
        )

    def _fused(self, flow):
        from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron

        # Materialize any Lazy BN (patch embed) before fusion, as the pipeline's
        # warmup forward does.
        with torch.no_grad():
            flow(torch.randn(2, 1, 28, 28))
        for perceptron in flow.get_perceptrons():
            fuse_into_perceptron(perceptron, device="cpu")
        return flow

    def test_nf_fold_removes_all_normalization_and_preserves_output(self):
        _, flow = self._converted_flow("batch")
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            out_before = flow(x).clone()

        self._fused(flow)
        for p in flow.get_perceptrons():
            assert isinstance(p.normalization, nn.Identity), p.name

        with torch.no_grad():
            out_after = flow(x)
        assert torch.allclose(out_before, out_after, atol=1e-4), (
            f"max diff: {(out_before - out_after).abs().max().item():.6f}"
        )

    def test_frozen_stats_wrap_preserves_flow_output(self):
        """WeightQuantizationStep wraps perceptron BN in FrozenStatsNormalization
        before NF; the wrap must respect the mixer's channels-last 3D layout."""
        from mimarsinan.models.nn.layers import FrozenStatsNormalization

        _, flow = self._converted_flow("batch")
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            out_before = flow(x).clone()

        for p in flow.get_perceptrons():
            if not isinstance(p.normalization, nn.Identity):
                p.normalization = FrozenStatsNormalization(p.normalization)
        with torch.no_grad():
            out_after = flow(x)
        assert torch.allclose(out_before, out_after, atol=1e-5), (
            f"max diff: {(out_before - out_after).abs().max().item():.6f}"
        )

    def test_nf_folded_bn_mixer_is_mapping_equivalent_to_norm_free(self):
        """After NF, the BN'd mixer must present the exact same perceptron
        structure (shapes, activations, Identity norms) as the norm-free build."""
        _, bn_flow = self._converted_flow("batch")
        _, none_flow = self._converted_flow("none")
        self._fused(bn_flow)
        self._fused(none_flow)

        bn_ps = bn_flow.get_perceptrons()
        none_ps = none_flow.get_perceptrons()
        assert len(bn_ps) == len(none_ps)
        for p_bn, p_none in zip(bn_ps, none_ps):
            assert tuple(p_bn.layer.weight.shape) == tuple(p_none.layer.weight.shape)
            assert (p_bn.layer.bias is None) == (p_none.layer.bias is None)
            assert isinstance(p_bn.normalization, nn.Identity)
            assert isinstance(p_none.normalization, nn.Identity)
            assert p_bn.base_activation_name == p_none.base_activation_name
