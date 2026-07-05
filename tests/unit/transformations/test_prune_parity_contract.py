"""W-CAL-2: pruning holds in COMMITTED RAW PARAMETERS, not only via forward hooks.

The deployed segment executor never fires ``nn.Module`` pre-hooks, so any raw
parameter the hooks would have re-zeroed is LIVE on chip. Three enforcement
points: (a) ``PerceptronTransformer`` effective->raw writes re-commit the
layer's prune masks and fail loud on non-finite results (the Shift step's
degenerate-``u`` inversion wrote raw biases in the hundreds into pruned rows);
(b) the DFQ loop commits masks at entry and never writes pruned rows;
(c) ``SoftCoreMappingStep`` commits and then fail-loud-verifies mask.param == param.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)
from mimarsinan.transformations.pruning.committed_masks import (
    commit_layer_pruning,
    commit_norm_pruning,
    commit_perceptron_pruning,
    verify_committed_pruning,
)
from mimarsinan.tuning.tuners.pruning.pruning_tuner_enforce import (
    enforce_pruning_persistently,
    register_prune_buffers,
)

PRUNED_ROWS = (1, 3)


def _pruned_perceptron(out=4, inp=3, seed=0):
    """A BN perceptron with committed pruning and a DEGENERATE fold factor
    ``u = gamma/sigma`` on the pruned rows (the t0_18 poison precondition)."""
    torch.manual_seed(seed)
    p = Perceptron(out, inp, normalization=nn.BatchNorm1d(out))
    norm = p.normalization
    with torch.no_grad():
        p.layer.weight.copy_(torch.randn(out, inp))
        p.layer.bias.copy_(torch.randn(out) * 0.1)
        norm.running_mean.copy_(torch.randn(out) * 0.1)
        norm.running_var.copy_(torch.rand(out) + 0.5)
        norm.weight.copy_(torch.rand(out) + 0.5)
        norm.bias.copy_(torch.randn(out) * 0.1)
        for r in PRUNED_ROWS:
            norm.weight[r] = 1e-4
            norm.running_var[r] = 1e-8
    row_keep = torch.ones(out, dtype=torch.bool)
    row_keep[list(PRUNED_ROWS)] = False
    col_keep = torch.ones(inp, dtype=torch.bool)
    register_prune_buffers([p], [row_keep], [col_keep])
    enforce_pruning_persistently([p], [row_keep], [col_keep])
    return p, ~row_keep


# ── (a) the effective->raw inversion no longer poisons pruned rows ─────────────


class TestEffectiveBiasTransformGuard:
    def test_shift_transform_leaves_pruned_rows_exactly_zero(self):
        """Reproduces today's poison: pruned perceptron + shift-step transform
        used to write raw biases ~ delta/u into pruned rows; they must come out
        exactly zero."""
        p, pruned = _pruned_perceptron()
        PerceptronTransformer().apply_effective_bias_transform(
            p, lambda b: b + 0.125,
        )
        assert torch.equal(
            p.layer.bias.detach()[pruned], torch.zeros(int(pruned.sum())),
        ), "the +0.125 effective shift must not materialize in pruned raw biases"

    def test_live_rows_still_receive_the_effective_shift(self):
        p, pruned = _pruned_perceptron()
        eff_before = PerceptronTransformer().get_effective_bias(p).detach().clone()
        PerceptronTransformer().apply_effective_bias_transform(
            p, lambda b: b + 0.125,
        )
        eff_after = PerceptronTransformer().get_effective_bias(p).detach()
        live = ~pruned
        torch.testing.assert_close(
            eff_after[live], eff_before[live] + 0.125, rtol=0, atol=1e-5,
        )

    def test_pruned_rows_keep_zero_effective_bias(self):
        p, pruned = _pruned_perceptron()
        PerceptronTransformer().apply_effective_bias_transform(
            p, lambda b: b + 0.125,
        )
        eff = PerceptronTransformer().get_effective_bias(p).detach()
        torch.testing.assert_close(
            eff[pruned], torch.zeros(int(pruned.sum())), rtol=0, atol=1e-6,
        )

    def test_zero_fold_factor_live_row_is_handled_totally(self):
        """A LIVE row with an exactly-zero fold factor used to invert to inf
        (guarded by a raise); the total inversion now realizes the delta through
        the normalization beta with bounded, finite raw params (W2 fix A)."""
        p, _ = _pruned_perceptron()
        with torch.no_grad():
            p.normalization.weight[0] = 0.0  # unpruned row, u == 0
        raw_before = p.layer.bias.detach().clone()
        eff_before = PerceptronTransformer().get_effective_bias(p).detach().clone()
        PerceptronTransformer().apply_effective_bias_transform(
            p, lambda b: b + 0.125,
        )
        assert torch.isfinite(p.layer.bias.detach()).all()
        assert torch.equal(p.layer.bias.detach()[0], raw_before[0])
        eff = PerceptronTransformer().get_effective_bias(p).detach()
        torch.testing.assert_close(
            eff[0], eff_before[0] + 0.125, rtol=1e-5, atol=1e-6,
        )


class TestEffectiveWeightTransformGuard:
    def test_weight_transform_preserves_committed_masks(self):
        p, _ = _pruned_perceptron()
        prune_mask = p.layer.prune_mask
        PerceptronTransformer().apply_effective_weight_transform(
            p, lambda w: w + 1.0,
        )
        assert torch.equal(
            p.layer.weight.detach()[prune_mask],
            torch.zeros(int(prune_mask.sum())),
        )

    def test_zero_fold_factor_live_row_keeps_raw_weights(self):
        """A LIVE row with u == 0 used to invert to inf (guarded by a raise);
        the total inversion keeps its raw weights unchanged (its effective
        weight is ~0 and stays so; W2 fix A)."""
        p, _ = _pruned_perceptron()
        with torch.no_grad():
            p.normalization.weight[0] = 0.0
        w_before = p.layer.weight.detach().clone()
        PerceptronTransformer().apply_effective_weight_transform(
            p, lambda w: w + 1.0,
        )
        assert torch.isfinite(p.layer.weight.detach()).all()
        assert torch.equal(p.layer.weight.detach()[0], w_before[0])


# ── the committed-mask SSOT helpers (hooks + direct form share one mechanism) ──


class TestCommittedMaskHelpers:
    def test_commit_layer_pruning_zeroes_masked_entries(self):
        p, pruned = _pruned_perceptron()
        with torch.no_grad():
            p.layer.weight[pruned] = 7.0
            p.layer.bias[pruned] = 7.0
        commit_layer_pruning(p.layer)
        assert torch.equal(
            p.layer.weight.detach()[p.layer.prune_mask],
            torch.zeros(int(p.layer.prune_mask.sum())),
        )
        assert torch.equal(
            p.layer.bias.detach()[pruned], torch.zeros(int(pruned.sum())),
        )

    def test_commit_norm_pruning_zeroes_mean_and_beta(self):
        p, pruned = _pruned_perceptron()
        norm = p.normalization
        with torch.no_grad():
            norm.running_mean[pruned] = 7.0
            norm.bias[pruned] = 7.0
        commit_norm_pruning(norm)
        assert torch.equal(
            norm.running_mean.detach()[pruned], torch.zeros(int(pruned.sum())),
        )
        assert torch.equal(
            norm.bias.detach()[pruned], torch.zeros(int(pruned.sum())),
        )

    def test_commit_is_a_no_op_without_masks(self):
        p = Perceptron(4, 3, normalization=nn.BatchNorm1d(4))
        before_w = p.layer.weight.detach().clone()
        before_b = p.layer.bias.detach().clone()
        commit_perceptron_pruning(p)
        assert torch.equal(p.layer.weight.detach(), before_w)
        assert torch.equal(p.layer.bias.detach(), before_b)

    def test_pre_hooks_delegate_to_the_same_mechanism(self):
        from mimarsinan.tuning.tuners.pruning.pruning_enforce_hooks import (
            pruning_enforce_linear_pre_hook,
            pruning_enforce_norm_pre_hook,
        )

        p, pruned = _pruned_perceptron()
        with torch.no_grad():
            p.layer.bias[pruned] = 9.0
            p.normalization.running_mean[pruned] = 9.0
        pruning_enforce_linear_pre_hook(p.layer, ())
        pruning_enforce_norm_pre_hook(p.normalization, ())
        assert torch.equal(
            p.layer.bias.detach()[pruned], torch.zeros(int(pruned.sum())),
        )
        assert torch.equal(
            p.normalization.running_mean.detach()[pruned],
            torch.zeros(int(pruned.sum())),
        )


class TestVerifyCommittedPruning:
    def test_clean_model_passes(self):
        p, _ = _pruned_perceptron()
        verify_committed_pruning([p], where="test")

    def test_model_without_masks_passes(self):
        verify_committed_pruning(
            [Perceptron(4, 3, normalization=nn.BatchNorm1d(4))], where="test",
        )

    def test_poisoned_bias_fails_loud(self):
        p, pruned = _pruned_perceptron()
        with torch.no_grad():
            p.layer.bias[list(PRUNED_ROWS)[0]] = 117.0
        with pytest.raises(RuntimeError, match="prun"):
            verify_committed_pruning([p], where="test")

    def test_poisoned_weight_fails_loud(self):
        p, _ = _pruned_perceptron()
        with torch.no_grad():
            p.layer.weight[list(PRUNED_ROWS)[0], 0] = 3.0
        with pytest.raises(RuntimeError, match="prun"):
            verify_committed_pruning([p], where="test")

    def test_poisoned_norm_fails_loud(self):
        p, _ = _pruned_perceptron()
        with torch.no_grad():
            p.normalization.running_mean[list(PRUNED_ROWS)[0]] = 3.0
        with pytest.raises(RuntimeError, match="prun"):
            verify_committed_pruning([p], where="test")


# ── (b) the DFQ loop is mask-aware ─────────────────────────────────────────────


class _PrunedFakePerceptron:
    def __init__(self, out_features, pruned_rows):
        self.layer = nn.Linear(out_features, out_features)
        self.layer.bias.data.zero_()
        mask = torch.zeros(out_features, dtype=torch.bool)
        mask[list(pruned_rows)] = True
        self.layer.register_buffer("prune_bias_mask", mask)
        self.layer.register_buffer(
            "prune_mask",
            mask.unsqueeze(1) | torch.zeros(
                out_features, out_features, dtype=torch.bool
            ),
        )
        self.output_channel_axis = -1
        self.normalization = nn.Identity()


class _PrunedFakeModel:
    def __init__(self, out_features=4, pruned_rows=(1,), offset=0.5):
        self._perceptrons = [_PrunedFakePerceptron(out_features, pruned_rows)]
        self._offset = offset

    def get_perceptrons(self):
        return self._perceptrons

    def cascade_means(self):
        return {
            k: p.layer.bias.detach() + self._offset
            for k, p in enumerate(self._perceptrons)
        }


class TestDfqMaskAware:
    def test_entry_commit_zeroes_preexisting_poison(self):
        from mimarsinan.spiking.dfq_bias_correction import dfq_correct_biases

        model = _PrunedFakeModel(pruned_rows=(1,))
        p = model.get_perceptrons()[0]
        with torch.no_grad():
            p.layer.bias[1] = 117.0  # the shift-step poison, pre-DFQ
        dfq_correct_biases(
            model,
            {0: torch.ones(4)},
            model.cascade_means,
            bias_iters=0,
            eta=0.7,
        )
        assert float(p.layer.bias.detach()[1]) == 0.0

    def test_no_writes_into_pruned_rows(self):
        from mimarsinan.spiking.dfq_bias_correction import dfq_correct_biases

        model = _PrunedFakeModel(pruned_rows=(1,), offset=0.5)
        p = model.get_perceptrons()[0]
        dfq_correct_biases(
            model,
            {0: torch.ones(4)},
            model.cascade_means,
            bias_iters=30,
            eta=0.7,
        )
        bias = p.layer.bias.detach()
        assert float(bias[1]) == 0.0, "DFQ must never write structurally-dead rows"
        live = torch.tensor([0, 2, 3])
        torch.testing.assert_close(
            bias[live], torch.full((3,), 0.5), rtol=0, atol=1e-3,
        )


# ── the conv forward path enforces committed masks (hooks never fire there) ────


def _conv_mapper(cls_name, **kwargs):
    if cls_name == "conv2d":
        from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper

        return Conv2DPerceptronMapper(
            None, in_channels=2, out_channels=4, kernel_size=3, padding=1,
            use_batchnorm=False, **kwargs,
        )
    from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper

    return Conv1DPerceptronMapper(
        None, in_channels=2, out_channels=4, kernel_size=3, padding=1,
        use_batchnorm=False, **kwargs,
    )


def _register_row_masks(layer, pruned_row=1):
    mask = torch.zeros(layer.out_features, dtype=torch.bool)
    mask[pruned_row] = True
    layer.register_buffer("prune_bias_mask", mask.clone())
    layer.register_buffer(
        "prune_mask", mask.unsqueeze(1).expand(-1, layer.in_features).clone(),
    )


class TestConvForwardEnforcesCommittedMasks:
    """``Conv{1,2}DPerceptronMapper._forward_impl`` drives the shared perceptron
    through ``F.conv`` on raw weights WITHOUT calling ``layer.__call__`` — the
    pruning pre-hooks never fire there, so the conv path must commit the masks
    itself (otherwise recovery training regrows pruned rows into load-bearing
    capacity that only exists outside the pruning contract)."""

    @pytest.mark.parametrize("kind,x_shape", [
        ("conv2d", (2, 2, 5, 5)), ("conv1d", (2, 2, 5)),
    ])
    def test_forward_ignores_and_rezeros_poisoned_pruned_rows(self, kind, x_shape):
        torch.manual_seed(0)
        mapper = _conv_mapper(kind)
        _register_row_masks(mapper.perceptron.layer, pruned_row=1)
        clean = torch.zeros_like(mapper.perceptron.layer.weight.detach())
        with torch.no_grad():
            clean.copy_(mapper.perceptron.layer.weight)
            clean[1] = 0.0
            mapper.perceptron.layer.bias[1] = 0.0
        x = torch.randn(*x_shape)
        with torch.no_grad():
            y_clean = mapper.forward(x)
            # poison the pruned row (what unhooked optimizer steps do)
            mapper.perceptron.layer.weight[1] = 3.0
            mapper.perceptron.layer.bias[1] = 5.0
            y_poisoned = mapper.forward(x)
        assert torch.equal(y_poisoned, y_clean), (
            "the conv forward must not use pruned-row parameters"
        )
        assert torch.equal(mapper.perceptron.layer.weight.detach(), clean)
        assert float(mapper.perceptron.layer.bias.detach()[1]) == 0.0

    def test_forward_without_masks_is_untouched(self):
        torch.manual_seed(0)
        mapper = _conv_mapper("conv2d")
        before = mapper.perceptron.layer.weight.detach().clone()
        with torch.no_grad():
            mapper.forward(torch.randn(2, 2, 5, 5))
        assert torch.equal(mapper.perceptron.layer.weight.detach(), before)


# ── (c) soft-core-mapping-time commit + fail-loud verification ─────────────────


class TestSoftCoreMappingPruneContract:
    def _step(self):
        from conftest import MockPipeline
        from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
            SoftCoreMappingStep,
        )

        return SoftCoreMappingStep(MockPipeline(config={}))

    def test_commit_then_verify_roundtrip(self):
        step = self._step()
        p, pruned = _pruned_perceptron()
        with torch.no_grad():
            p.layer.bias[pruned] = 0.03  # benign optimizer drift
        model = SimpleNamespace(get_perceptrons=lambda: [p])
        step._commit_pruning_to_raw_params(model)
        step._verify_pruning_committed(model)
        assert torch.equal(
            p.layer.bias.detach()[pruned], torch.zeros(int(pruned.sum())),
        )

    def test_verify_without_commit_fails_loud_on_poison(self):
        step = self._step()
        p, pruned = _pruned_perceptron()
        with torch.no_grad():
            p.layer.bias[list(PRUNED_ROWS)[0]] = 205.0
        model = SimpleNamespace(get_perceptrons=lambda: [p])
        with pytest.raises(RuntimeError, match="prun"):
            step._verify_pruning_committed(model)


# ── the module-tree commit/verify pair (cache-boundary form of the SSOT) ───────


class TestModelTreeCommitVerify:
    """``commit_model_pruning`` / ``verify_model_pruning`` walk the module tree
    directly, so seams without ``get_perceptrons()`` (the pipeline cache) share
    the exact enforcement mechanism."""

    def _model(self):
        p, pruned = _pruned_perceptron()
        return nn.Sequential(p), pruned

    def test_commit_zeroes_layer_and_norm_entries(self):
        from mimarsinan.transformations.pruning.committed_masks import (
            commit_model_pruning,
        )

        model, pruned = self._model()
        p = model[0]
        with torch.no_grad():
            p.layer.weight[pruned] = 5.0
            p.layer.bias[pruned] = 5.0
            p.normalization.running_mean[pruned] = 5.0
            p.normalization.bias[pruned] = 5.0
        commit_model_pruning(model)
        assert torch.equal(
            p.layer.weight.detach()[p.layer.prune_mask],
            torch.zeros(int(p.layer.prune_mask.sum())),
        )
        assert torch.equal(
            p.layer.bias.detach()[pruned], torch.zeros(int(pruned.sum())),
        )
        assert torch.equal(
            p.normalization.running_mean.detach()[pruned],
            torch.zeros(int(pruned.sum())),
        )
        assert torch.equal(
            p.normalization.bias.detach()[pruned], torch.zeros(int(pruned.sum())),
        )

    def test_verify_passes_clean_and_names_the_module_on_poison(self):
        from mimarsinan.transformations.pruning.committed_masks import (
            verify_model_pruning,
        )

        model, pruned = self._model()
        verify_model_pruning(model, where="test")
        with torch.no_grad():
            model[0].layer.weight[list(PRUNED_ROWS)[0], 0] = 3.0
        with pytest.raises(RuntimeError, match="0.layer"):
            verify_model_pruning(model, where="test")

    def test_verify_is_a_no_op_without_masks(self):
        from mimarsinan.transformations.pruning.committed_masks import (
            verify_model_pruning,
        )

        verify_model_pruning(nn.Sequential(nn.Linear(4, 4)), where="test")
