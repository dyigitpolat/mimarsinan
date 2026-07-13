"""Graph rules: each fires on a synthetic converted-model trigger and stays silent otherwise."""

import torch
import torch.nn as nn

from mimarsinan.advisories.engine import evaluate_graph_advisories
from mimarsinan.mapping.mapping_utils import (
    InputMapper,
    ModelRepresentation,
    PerceptronMapper,
)
from mimarsinan.mapping.mappers.structural import ConcatMapper
from mimarsinan.models.deep_mlp import DeepMLP
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.converter import convert_torch_model

_INPUT_SHAPE = (1, 8, 8)
_NUM_CLASSES = 4


def _converted_deep_mlp(depth: int, seed: int = 0):
    torch.manual_seed(seed)
    model = DeepMLP(
        input_shape=_INPUT_SHAPE, num_classes=_NUM_CLASSES, depth=depth, width=8,
    ).eval()
    flow = convert_torch_model(
        model, input_shape=_INPUT_SHAPE, num_classes=_NUM_CLASSES,
    ).eval()
    return flow.get_mapper_repr()


def _lif_config(**over):
    config = {
        "spiking_mode": "lif",
        "simulation_steps": 4,
        "activation_quantization": True,
    }
    config.update(over)
    return config


def _ids(model_repr, config, channel_stats=None):
    return {
        a.id
        for a in evaluate_graph_advisories(
            model_repr, config, channel_stats=channel_stats
        )
    }


def _by_id(model_repr, config, advisory_id, channel_stats=None):
    fired = [
        a
        for a in evaluate_graph_advisories(
            model_repr, config, channel_stats=channel_stats
        )
        if a.id == advisory_id
    ]
    assert len(fired) == 1, f"{advisory_id} fired {len(fired)} times"
    return fired[0]


class TestStaircaseDepth:
    def test_fires_on_deep_chain_at_low_s(self):
        repr_ = _converted_deep_mlp(depth=7)
        advisory = _by_id(repr_, _lif_config(simulation_steps=4), "ADV-STAIRCASE-DEPTH")
        assert advisory.mandate_violation is True
        assert advisory.tentative is True
        assert "sync_deployment_exactness.md" in advisory.detail
        assert "lif_deployment_exactness.md" in advisory.detail

    def test_silent_at_high_s(self):
        repr_ = _converted_deep_mlp(depth=7)
        assert "ADV-STAIRCASE-DEPTH" not in _ids(repr_, _lif_config(simulation_steps=16))

    def test_silent_on_shallow_chain(self):
        repr_ = _converted_deep_mlp(depth=3)
        assert "ADV-STAIRCASE-DEPTH" not in _ids(repr_, _lif_config(simulation_steps=4))

    def test_silent_on_value_continuous_deployment(self):
        """Analytic ttfs without activation quantization deploys value-continuous."""
        repr_ = _converted_deep_mlp(depth=7)
        config = {
            "spiking_mode": "ttfs",
            "simulation_steps": 4,
            "activation_quantization": False,
        }
        assert "ADV-STAIRCASE-DEPTH" not in _ids(repr_, config)

    def test_fires_for_synchronized_ttfs_with_trained_composition_adjudication(self):
        """[WS-W] The rule fires on the sync cell class and its detail carries
        the measured post-QAT adjudication: the residual is AQ capacity and
        twin-referenced post-hoc folds are closed (do not re-derive)."""
        repr_ = _converted_deep_mlp(depth=7)
        config = {
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "synchronized",
            "simulation_steps": 8,
            "activation_quantization": True,
        }
        advisory = _by_id(repr_, config, "ADV-STAIRCASE-DEPTH")
        assert "AQ capacity" in advisory.detail
        assert "do not re-derive" in advisory.detail
        assert "lossless_refinement_ledger.md" in advisory.detail

    def test_trained_residual_band_constant_matches_the_detail(self):
        """The measured residual band is a named module constant and the
        advisory text quotes exactly it (no drifting duplicate literals)."""
        from mimarsinan.advisories.rules_graph import (
            STAIRCASE_TRAINED_RESIDUAL_BAND_PP,
        )

        low, high = STAIRCASE_TRAINED_RESIDUAL_BAND_PP
        assert 0.0 < low < high
        repr_ = _converted_deep_mlp(depth=7)
        advisory = _by_id(repr_, _lif_config(simulation_steps=8), "ADV-STAIRCASE-DEPTH")
        assert f"{low:.1f}" in advisory.detail and f"{high:.1f}" in advisory.detail


class TestNormfreeChain:
    def test_fires_on_norm_free_chain(self):
        repr_ = _converted_deep_mlp(depth=6)
        advisory = _by_id(repr_, _lif_config(), "ADV-NORMFREE-CHAIN")
        assert advisory.mandate_violation is True
        assert "mixer_column_scale_pathology.md" in advisory.detail

    def test_silent_on_short_chain(self):
        repr_ = _converted_deep_mlp(depth=3)
        assert "ADV-NORMFREE-CHAIN" not in _ids(repr_, _lif_config())

    def test_normalization_breaks_the_chain(self):
        class _NormalizedMLP(nn.Module):
            def __init__(self):
                super().__init__()
                width = 8
                self.flatten = nn.Flatten()
                layers = []
                in_features = 64
                for _ in range(7):
                    layers += [
                        nn.Linear(in_features, width),
                        nn.BatchNorm1d(width),
                        nn.ReLU(),
                    ]
                    in_features = width
                self.hidden = nn.Sequential(*layers)
                self.classifier = nn.Linear(width, _NUM_CLASSES)

            def forward(self, x):
                return self.classifier(self.hidden(self.flatten(x)))

        torch.manual_seed(0)
        flow = convert_torch_model(
            _NormalizedMLP().eval(),
            input_shape=_INPUT_SHAPE,
            num_classes=_NUM_CLASSES,
        ).eval()
        assert "ADV-NORMFREE-CHAIN" not in _ids(flow.get_mapper_repr(), _lif_config())


class TestScaleSpread:
    def _inflate_one_channel(self, model_repr, factor: float):
        perceptrons = model_repr.get_perceptrons()
        target = perceptrons[1]
        with torch.no_grad():
            target.layer.weight.data[0, :] *= factor
        return target

    def test_fires_on_inflated_channel_weight_proxy(self):
        repr_ = _converted_deep_mlp(depth=3)
        self._inflate_one_channel(repr_, 1000.0)
        advisory = _by_id(repr_, _lif_config(), "ADV-SCALE-SPREAD")
        assert advisory.mandate_violation is True
        assert "weight" in advisory.detail.lower()
        assert "mixer_column_scale_pathology.md" in advisory.detail
        assert "scale_migration" in " ".join(advisory.suggested_levers)

    def test_silent_on_balanced_channels(self):
        repr_ = _converted_deep_mlp(depth=3)
        assert "ADV-SCALE-SPREAD" not in _ids(repr_, _lif_config())

    def test_runtime_q99_stats_take_precedence(self):
        repr_ = _converted_deep_mlp(depth=3)
        perceptrons = repr_.get_perceptrons()
        stats = {
            id(p): [1.0] * int(p.layer.weight.shape[0]) for p in perceptrons
        }
        hot = perceptrons[1]
        stats[id(hot)] = [1000.0] + [1.0] * (int(hot.layer.weight.shape[0]) - 1)
        advisory = _by_id(
            repr_, _lif_config(), "ADV-SCALE-SPREAD", channel_stats=stats
        )
        assert "q99" in advisory.detail

    def test_balanced_q99_stats_keep_it_silent(self):
        repr_ = _converted_deep_mlp(depth=3)
        # The weight proxy would fire; healthy runtime stats override it.
        self._inflate_one_channel(repr_, 1000.0)
        stats = {
            id(p): [1.0] * int(p.layer.weight.shape[0])
            for p in repr_.get_perceptrons()
        }
        assert "ADV-SCALE-SPREAD" not in _ids(
            repr_, _lif_config(), channel_stats=stats
        )


class TestBiasGridDominance:
    def _wq_config(self, **over):
        config = _lif_config(
            weight_quantization=True,
            weight_bits=5,
            wq_two_scale_projection=False,
        )
        config.update(over)
        return config

    def _dominated_repr(self):
        repr_ = _converted_deep_mlp(depth=3)
        target = repr_.get_perceptrons()[1]
        with torch.no_grad():
            w_max = float(target.layer.weight.data.abs().max())
            target.layer.bias.data[:] = 100.0 * w_max
        return repr_

    def test_fires_on_bias_dominated_grid(self):
        advisory = _by_id(
            self._dominated_repr(), self._wq_config(), "ADV-BIAS-GRID-DOMINANCE"
        )
        assert advisory.mandate_violation is True
        assert "wq_cascade_crater_repair.md" in advisory.detail

    def test_silent_without_weight_quantization(self):
        assert "ADV-BIAS-GRID-DOMINANCE" not in _ids(
            self._dominated_repr(), self._wq_config(weight_quantization=False)
        )

    def test_silent_when_two_scale_projection_is_effective(self):
        """Two-scale on a platform with an on-chip bias register puts the bias
        on its own grid — the shared-grid starvation channel is closed."""
        assert "ADV-BIAS-GRID-DOMINANCE" not in _ids(
            self._dominated_repr(), self._wq_config(wq_two_scale_projection=True)
        )

    def test_fires_when_two_scale_is_requested_but_platform_has_no_bias_register(self):
        config = self._wq_config(
            wq_two_scale_projection=True,
            platform_constraints={"has_bias": False},
        )
        assert "ADV-BIAS-GRID-DOMINANCE" in _ids(self._dominated_repr(), config)

    def test_silent_on_healthy_ratios(self):
        assert "ADV-BIAS-GRID-DOMINANCE" not in _ids(
            _converted_deep_mlp(depth=3), self._wq_config()
        )


def _unbalanced_join_repr():
    """input -> enc -> {p1 -> p2, direct} -> concat -> p3: the concat joins
    branch depths 1 (enc) and 3 (p2) inside one neural segment."""
    torch.manual_seed(0)
    enc = Perceptron(8, 64, name="enc")
    enc.is_encoding_layer = True
    p1 = Perceptron(8, 8, name="p1")
    p2 = Perceptron(8, 8, name="p2")
    p3 = Perceptron(_NUM_CLASSES, 16, name="p3")

    inp = InputMapper((64,))
    enc_m = PerceptronMapper(inp, enc)
    p1_m = PerceptronMapper(enc_m, p1)
    p2_m = PerceptronMapper(p1_m, p2)
    join = ConcatMapper([enc_m, p2_m], dim=1)
    out = PerceptronMapper(join, p3)
    return ModelRepresentation(out)


def _balanced_join_repr():
    """Both concat branches meet at equal depth: no imbalance."""
    torch.manual_seed(0)
    enc = Perceptron(8, 64, name="enc")
    enc.is_encoding_layer = True
    a = Perceptron(8, 8, name="a")
    b = Perceptron(8, 8, name="b")
    p3 = Perceptron(_NUM_CLASSES, 16, name="p3")

    inp = InputMapper((64,))
    enc_m = PerceptronMapper(inp, enc)
    a_m = PerceptronMapper(enc_m, a)
    b_m = PerceptronMapper(enc_m, b)
    join = ConcatMapper([a_m, b_m], dim=1)
    out = PerceptronMapper(join, p3)
    return ModelRepresentation(out)


class TestFanInDepthImbalance:
    def test_fires_on_unequal_depth_join_with_relays_off(self):
        config = _lif_config(lif_depth_balancing_relays=False)
        advisory = _by_id(
            _unbalanced_join_repr(), config, "ADV-FANIN-DEPTH-IMBALANCE"
        )
        assert advisory.mandate_violation is True
        assert "lif_deployment_exactness.md" in advisory.detail
        assert "lif_depth_balancing_relays" in " ".join(advisory.suggested_levers)

    def test_silent_when_relays_are_on(self):
        config = _lif_config(lif_depth_balancing_relays=True)
        assert "ADV-FANIN-DEPTH-IMBALANCE" not in _ids(
            _unbalanced_join_repr(), config
        )

    def test_silent_on_balanced_join(self):
        config = _lif_config(lif_depth_balancing_relays=False)
        assert "ADV-FANIN-DEPTH-IMBALANCE" not in _ids(
            _balanced_join_repr(), config
        )

    def test_silent_for_non_lif_deployments(self):
        config = {
            "spiking_mode": "ttfs_quantized",
            "simulation_steps": 4,
            "lif_depth_balancing_relays": False,
        }
        assert "ADV-FANIN-DEPTH-IMBALANCE" not in _ids(
            _unbalanced_join_repr(), config
        )
