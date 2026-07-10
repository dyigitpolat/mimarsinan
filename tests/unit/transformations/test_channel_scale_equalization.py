"""Cross-layer channel-scale migration: exactness, clip invariant, adjacency honesty.

The mechanism under test is the M4 pass from
docs/research/findings/mixer_column_scale_pathology.md: for feature-adjacent
affine pairs A -> ReLU -> B, migrate per-channel scale s_c = q99_c / geomean
(clipped to [1/r, r]) as W_A <- S^-1 W_A, b_A <- S^-1 b_A, W_B <- W_B S. The
float function must be preserved exactly (ReLU positive homogeneity); the
weight-shared token-fc2 axes are NOT migratable (the exact escape there is a
per-channel theta, owned elsewhere) and must be left alone.
"""

import copy

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.leading_dim import Ensure2DMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.transformations.channel_scale_equalization import (
    DEFAULT_CLIP_RATIO,
    apply_scale_migration,
    equalize_channel_scales,
    find_migratable_pairs,
    migration_scales,
)
from mimarsinan.tuning.orchestration.install_resolution import (
    ChannelStatsAccumulator,
    collect_channel_stats,
)

INPUT_SHAPE = (1, 8, 8)
NUM_CLASSES = 4


# ---------------------------------------------------------------- fixtures

def _tiny_mixer_flow(seed=0):
    """Converted tier-0-shaped mixer with a deliberate per-channel spread on fc1."""
    torch.manual_seed(seed)
    model = TorchMLPMixerCore(
        input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
        patch_n_1=2, patch_m_1=2, patch_c_1=4, fc_w_1=8, fc_w_2=8,
        base_activation="ReLU", num_blocks=2,
    ).eval()
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.mul_(3.0)
                if mod.bias is not None:
                    mod.bias.copy_(torch.randn_like(mod.bias) * 0.5)
        for blk in model.mixer_blocks:
            spread = torch.linspace(0.5, 3.0, blk.fc1.weight.shape[0]).view(-1, 1)
            blk.fc1.weight.mul_(spread)
            blk.fc1.bias.mul_(spread.view(-1))
    return convert_torch_model(
        model, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
    ).eval()


class _ChainFlow(nn.Module):
    """Input -> perceptron chain mapper graph (the mixer-free minimal DAG)."""

    def __init__(self, perceptrons, in_features=4):
        super().__init__()
        self.perceptron_modules = nn.ModuleList(perceptrons)
        out = InputMapper((in_features,))
        out = Ensure2DMapper(out)
        for p in perceptrons:
            out = PerceptronMapper(out, p)
        self._repr = ModelRepresentation(out)

    def get_perceptrons(self):
        return self._repr.get_perceptrons()

    def get_mapper_repr(self):
        return self._repr

    def forward(self, x):
        return self._repr(x)


def _stats_by_id(flow, batches):
    stats = collect_channel_stats(
        flow, batches, "cpu",
        accumulator_factory=lambda p: ChannelStatsAccumulator(
            channel_axis=p.output_channel_axis,
        ),
    )
    return {id(p): acc.per_channel_q99() for p, acc in stats}


# ---------------------------------------------------------------- adjacency

class TestMixerAdjacency:
    def test_finds_exactly_the_five_migratable_pairs(self):
        flow = _tiny_mixer_flow()
        perceptrons = list(flow.get_perceptrons())
        assert len(perceptrons) == 9  # patch conv + 4 mixer cores x fc1/fc2

        pairs = find_migratable_pairs(flow.get_mapper_repr())
        producers = {id(p.producer): p for p in pairs}

        # tok.fc1 -> tok.fc2 and ch.fc1 -> ch.fc2 in every core, plus the last
        # ch.fc2 -> mean -> classifier (mean over patches is channel-homogeneous).
        expected_producer_idx = [1, 3, 5, 7, 8]
        assert sorted(producers) == sorted(
            id(perceptrons[i]) for i in expected_producer_idx
        )

        for fc1_idx, fc2_idx in ((1, 2), (3, 4), (5, 6), (7, 8)):
            pair = producers[id(perceptrons[fc1_idx])]
            assert pair.consumer_perceptrons == (perceptrons[fc2_idx],)
            assert pair.consumer_modules == ()

        classifier_pair = producers[id(perceptrons[8])]
        assert classifier_pair.consumer_perceptrons == ()
        assert len(classifier_pair.consumer_modules) == 1
        classifier = classifier_pair.consumer_modules[0]
        assert isinstance(classifier, nn.Linear)
        assert classifier.out_features == NUM_CLASSES

    def test_weight_shared_and_axis_flipped_hops_are_not_producers(self):
        """token-fc2 channels ARE patch positions consumed under weight sharing
        (per-channel theta is the exact escape, owned elsewhere); the patch stem
        and the mid ch-fc2 flip their channel axis before the next consumer."""
        flow = _tiny_mixer_flow()
        perceptrons = list(flow.get_perceptrons())
        producer_ids = {
            id(p.producer) for p in find_migratable_pairs(flow.get_mapper_repr())
        }
        for excluded_idx in (0, 2, 4, 6):
            assert id(perceptrons[excluded_idx]) not in producer_ids


class TestChainAdjacency:
    def test_output_layer_is_never_a_producer(self):
        torch.manual_seed(0)
        flow = _ChainFlow([Perceptron(8, 4), Perceptron(3, 8)])
        pairs = find_migratable_pairs(flow.get_mapper_repr())
        assert len(pairs) == 1
        assert pairs[0].producer is flow.get_perceptrons()[0]

    def test_gelu_activation_blocks_migration(self):
        torch.manual_seed(0)
        flow = _ChainFlow([
            Perceptron(8, 4, base_activation_name="GELU"),
            Perceptron(8, 8),
            Perceptron(3, 8),
        ])
        perceptrons = flow.get_perceptrons()
        pairs = find_migratable_pairs(flow.get_mapper_repr())
        assert [p.producer for p in pairs] == [perceptrons[1]]

    def test_consumer_with_input_wire_op_blocks_migration(self):
        torch.manual_seed(0)
        flow = _ChainFlow([Perceptron(8, 4), Perceptron(8, 8), Perceptron(3, 8)])
        perceptrons = flow.get_perceptrons()
        perceptrons[1].append_input_wire_op(nn.ReLU())
        pairs = find_migratable_pairs(flow.get_mapper_repr())
        assert [p.producer for p in pairs] == [perceptrons[1]]


# ---------------------------------------------------------------- scales

class TestMigrationScales:
    def test_geomean_normalization_and_dead_channels(self):
        # live geomean of {1, 4, 0.25} is exactly 1; dead channels pin s=1.
        s = migration_scales([1.0, 4.0, 0.0, 0.25])
        assert s is not None
        assert torch.allclose(
            s, torch.tensor([1.0, 4.0, 1.0, 0.25], dtype=s.dtype)
        )

    def test_clip_invariant(self):
        q99 = [0.008, 0.5, 2.0, 15.0]
        s = migration_scales(q99, clip_ratio=DEFAULT_CLIP_RATIO)
        assert s is not None
        assert float(s.max()) <= DEFAULT_CLIP_RATIO + 1e-12
        assert float(s.min()) >= 1.0 / DEFAULT_CLIP_RATIO - 1e-12

        tighter = migration_scales(q99, clip_ratio=2.0)
        assert tighter is not None
        assert float(tighter.max()) <= 2.0 + 1e-12
        assert float(tighter.min()) >= 0.5 - 1e-12

    def test_clip_ratio_below_one_rejected(self):
        with pytest.raises(ValueError):
            migration_scales([1.0, 2.0], clip_ratio=0.5)

    def test_all_dead_channels_yield_no_scales(self):
        assert migration_scales([0.0, 0.0]) is None


# ---------------------------------------------------------------- exactness

class TestExactness:
    def test_equalization_preserves_float_function_on_mixer(self):
        flow = _tiny_mixer_flow()
        torch.manual_seed(1)
        batches = [torch.rand(16, *INPUT_SHAPE) for _ in range(2)]
        stats = _stats_by_id(flow, batches)

        with torch.no_grad():
            reference = [flow(x) for x in batches]

        report = equalize_channel_scales(flow, stats, clip_ratio=4.0)
        assert len(report.migrated) >= 4  # at minimum the four fc1 hops

        with torch.no_grad():
            migrated = [flow(x) for x in batches]
        for ref, out in zip(reference, migrated):
            assert float((ref - out).abs().max()) <= 1e-5
            assert torch.equal(ref.argmax(-1), out.argmax(-1))

    def test_equalization_shrinks_live_channel_spread(self):
        flow = _tiny_mixer_flow()
        torch.manual_seed(1)
        batches = [torch.rand(16, *INPUT_SHAPE) for _ in range(2)]
        pre = _stats_by_id(flow, batches)
        report = equalize_channel_scales(flow, pre, clip_ratio=4.0)
        post = _stats_by_id(flow, batches)

        migrated_names = {hop.name for hop in report.migrated}
        for p in flow.get_perceptrons():
            if p.name not in migrated_names:
                continue
            pre_live = [q for q in pre[id(p)] if q > 0]
            post_live = [q for q in post[id(p)] if q > 0]
            pre_ratio = max(pre_live) / min(pre_live)
            post_ratio = max(post_live) / min(post_live)
            assert post_ratio <= pre_ratio + 1e-6

    def test_bn_attached_producer_scales_norm_affine_exactly(self):
        torch.manual_seed(0)
        p1 = Perceptron(8, 4, normalization=nn.BatchNorm1d(8))
        p2 = Perceptron(3, 8)
        flow = _ChainFlow([p1, p2])
        flow.train()
        with torch.no_grad():
            for _ in range(4):
                flow(torch.randn(16, 4))
        flow.eval()

        frozen = copy.deepcopy(flow)
        layer_weight_before = p1.layer.weight.detach().clone()
        norm_weight_before = p1.normalization.weight.detach().clone()

        q99 = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]
        stats = {id(p1): q99}
        report = equalize_channel_scales(flow, stats, clip_ratio=4.0)
        assert [hop.name for hop in report.migrated] == [p1.name]

        # BN realization: the layer is untouched, S^-1 lands on the norm affine.
        assert torch.equal(p1.layer.weight, layer_weight_before)
        assert not torch.equal(p1.normalization.weight, norm_weight_before)

        x = torch.randn(32, 4)
        with torch.no_grad():
            assert torch.allclose(frozen(x), flow(x), atol=1e-6)

        # Exact in TRAIN-mode BN too: batch stats of the pre-norm are untouched.
        frozen.train()
        flow.train()
        with torch.no_grad():
            assert torch.allclose(frozen(x), flow(x), atol=1e-6)

    def test_chained_pairs_compose_exactly(self):
        torch.manual_seed(0)
        p1, p2, p3 = Perceptron(8, 4), Perceptron(6, 8), Perceptron(3, 6)
        flow = _ChainFlow([p1, p2, p3]).eval()
        frozen = copy.deepcopy(flow)

        stats = {
            id(p1): [0.1, 0.4, 0.9, 1.5, 2.5, 4.0, 8.0, 16.0],
            id(p2): [0.2, 0.5, 1.0, 2.0, 4.0, 9.0],
        }
        report = equalize_channel_scales(flow, stats, clip_ratio=4.0)
        assert len(report.migrated) == 2

        x = torch.randn(32, 4)
        with torch.no_grad():
            assert torch.allclose(frozen(x), flow(x), atol=1e-6)


# ---------------------------------------------------------------- no-op path

class TestNoopAndFailLoud:
    def test_noop_when_spread_small(self):
        torch.manual_seed(0)
        p1, p2 = Perceptron(8, 4), Perceptron(3, 8)
        flow = _ChainFlow([p1, p2])
        weight_before = p1.layer.weight.detach().clone()
        consumer_before = p2.layer.weight.detach().clone()

        report = equalize_channel_scales(flow, {id(p1): [2.0] * 8})
        assert report.migrated == ()
        assert p1.name in report.skipped
        assert torch.equal(p1.layer.weight, weight_before)
        assert torch.equal(p2.layer.weight, consumer_before)

    def test_missing_stats_for_migratable_producer_fails_loud(self):
        torch.manual_seed(0)
        flow = _ChainFlow([Perceptron(8, 4), Perceptron(3, 8)])
        with pytest.raises(ValueError):
            equalize_channel_scales(flow, {})

    def test_stats_length_mismatch_fails_loud(self):
        torch.manual_seed(0)
        p1 = Perceptron(8, 4)
        flow = _ChainFlow([p1, Perceptron(3, 8)])
        with pytest.raises(ValueError, match="axis"):
            equalize_channel_scales(flow, {id(p1): [1.0, 4.0, 0.25]})

    def test_apply_returns_false_below_noop_tolerance(self):
        torch.manual_seed(0)
        p1, p2 = Perceptron(8, 4), Perceptron(3, 8)
        flow = _ChainFlow([p1, p2])
        (pair,) = find_migratable_pairs(flow.get_mapper_repr())
        assert apply_scale_migration(pair, torch.ones(8, dtype=torch.float64)) is False


# ------------------------------------------------- stats machinery reuse

class TestChannelStatsAxis:
    def test_declared_channel_axis_resolves_last_axis(self):
        acc = ChannelStatsAccumulator(channel_axis=-1)
        x = torch.zeros(2, 3, 5)
        for c in range(5):
            x[:, :, c] = float(c + 1)
        acc.output_transform(x)
        q99 = acc.per_channel_q99()
        assert q99 == [float(c + 1) for c in range(5)]

    def test_default_axis_stays_legacy_dim_one(self):
        acc = ChannelStatsAccumulator()
        x = torch.zeros(2, 3, 5)
        for n in range(3):
            x[:, n, :] = float(n + 1)
        acc.output_transform(x)
        q99 = acc.per_channel_q99()
        assert q99 == [float(n + 1) for n in range(3)]

    def test_collect_channel_stats_accumulator_factory(self):
        torch.manual_seed(0)
        flow = _tiny_mixer_flow()
        batches = [torch.rand(8, *INPUT_SHAPE)]
        stats = collect_channel_stats(
            flow, batches, "cpu",
            accumulator_factory=lambda p: ChannelStatsAccumulator(
                channel_axis=p.output_channel_axis,
            ),
        )
        for perceptron, acc in stats:
            assert len(acc.per_channel_q99()) == perceptron.output_channels
