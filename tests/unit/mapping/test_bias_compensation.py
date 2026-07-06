"""Characterization pins for bias_compensation: exact tensors in = exact tensors out.

These literals were captured from the pre-merge neg_shift_bias / ttfs_bias
modules; every comparison is bit-exact (torch.equal) so the merged module
cannot drift numerically. The synchronized floor-collapse depends on the
ttfs bake's +shift semantics — a double-shift here once cost -1.9pp.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.support.bias_compensation import (
    apply_additive_effective_bias_shift,
    apply_negative_shift_bias,
    apply_ttfs_quantization_bias_compensation,
    apply_ttfs_quantized_bias_shift,
    negative_shifts_from_min,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

_W = torch.tensor([[1.0, -2.0], [0.5, 4.0], [-3.0, 0.25]])
_B = torch.tensor([0.5, -1.5, 2.0])


def _perceptron(activation_scale=None, bias=True):
    p = Perceptron(3, 2, bias=bias, normalization=nn.Identity())
    p.layer.weight.data = _W.clone()
    if bias:
        p.layer.bias.data = _B.clone()
    if activation_scale is not None:
        p.activation_scale.data = torch.tensor(activation_scale)
    return p


class _Model:
    def __init__(self, perceptrons):
        self._perceptrons = perceptrons

    def get_perceptrons(self):
        return self._perceptrons


def _assert_bias_exact(perceptron, expected):
    assert torch.equal(perceptron.layer.bias.data, torch.tensor(expected))


class TestApplyAdditiveEffectiveBiasShift:
    def test_adds_shift_in_effective_domain_identity_scale(self):
        p = _perceptron()
        baked = apply_additive_effective_bias_shift(p, 0.125, baked_flag="_flag")
        assert baked is True
        _assert_bias_exact(p, [0.625, -1.375, 2.125])
        assert p._flag is True

    def test_effective_domain_rescales_through_activation_scale(self):
        p = _perceptron(0.5)
        apply_additive_effective_bias_shift(p, 0.125, baked_flag="_flag")
        _assert_bias_exact(p, [0.5625, -1.4375, 2.0625])

    def test_idempotent_per_flag(self):
        p = _perceptron()
        apply_additive_effective_bias_shift(p, 0.125, baked_flag="_flag")
        baked_again = apply_additive_effective_bias_shift(p, 0.125, baked_flag="_flag")
        assert baked_again is False
        _assert_bias_exact(p, [0.625, -1.375, 2.125])

    def test_bias_none_sets_flag_without_crash(self):
        p = _perceptron(bias=False)
        baked = apply_additive_effective_bias_shift(p, 0.125, baked_flag="_flag")
        assert baked is True and p.layer.bias is None and p._flag is True


class TestApplyNegativeShiftBias:
    def test_per_axon_shift_exact(self):
        p = _perceptron()
        apply_negative_shift_bias(p, torch.tensor([0.5, 0.25]))
        _assert_bias_exact(p, [0.5, -2.75, 3.4375])
        assert p._neg_shift_baked is True

    def test_scalar_shift_exact(self):
        p = _perceptron()
        apply_negative_shift_bias(p, 1.5)
        _assert_bias_exact(p, [2.0, -8.25, 6.125])

    def test_activation_scale_cancels_for_identity_norm(self):
        p = _perceptron(0.5)
        apply_negative_shift_bias(p, torch.tensor([0.5, 0.25]))
        _assert_bias_exact(p, [0.5, -2.75, 3.4375])

    def test_idempotent(self):
        p = _perceptron()
        apply_negative_shift_bias(p, torch.tensor([0.5, 0.25]))
        apply_negative_shift_bias(p, torch.tensor([0.5, 0.25]))
        _assert_bias_exact(p, [0.5, -2.75, 3.4375])

    def test_bias_none_sets_flag_without_crash(self):
        p = _perceptron(bias=False)
        apply_negative_shift_bias(p, torch.tensor([0.5, 0.25]))
        assert p.layer.bias is None and p._neg_shift_baked is True


class TestNegativeShiftsFromMin:
    def test_exact_shift_table(self):
        out = negative_shifts_from_min(
            {5: np.array([-0.4, 0.1, -1.2]), 7: np.array([0.0, 0.3, 0.5])}
        )
        assert set(out) == {5}
        assert out[5].dtype == np.float64
        np.testing.assert_array_equal(out[5], np.array([0.4, 0.0, 1.2]))


class TestApplyTtfsQuantizationBiasCompensation:
    def test_plus_half_step_shift_exact(self):
        model = _Model([_perceptron()])
        apply_ttfs_quantization_bias_compensation(model, 4)
        _assert_bias_exact(model.get_perceptrons()[0], [0.625, -1.375, 2.125])
        assert model.get_perceptrons()[0]._ttfs_shift_baked_into_bias is True

    def test_activation_scale_cancels_in_stored_bias_step(self):
        model = _Model([_perceptron(0.5)])
        apply_ttfs_quantization_bias_compensation(model, 4)
        _assert_bias_exact(model.get_perceptrons()[0], [0.5625, -1.4375, 2.0625])

    def test_idempotent_no_double_shift(self):
        model = _Model([_perceptron()])
        apply_ttfs_quantization_bias_compensation(model, 4)
        apply_ttfs_quantization_bias_compensation(model, 4)
        _assert_bias_exact(model.get_perceptrons()[0], [0.625, -1.375, 2.125])

    def test_encoding_layer_skipped_and_unflagged(self):
        encoder = _perceptron()
        encoder.is_encoding_layer = True
        apply_ttfs_quantization_bias_compensation(_Model([encoder]), 4)
        _assert_bias_exact(encoder, [0.5, -1.5, 2.0])
        assert not getattr(encoder, "_ttfs_shift_baked_into_bias", False)

    def test_alias_matches(self):
        model = _Model([_perceptron()])
        apply_ttfs_quantized_bias_shift(model, 4)
        _assert_bias_exact(model.get_perceptrons()[0], [0.625, -1.375, 2.125])


class TestApplySyncExactEntryHalfStep:
    """[5v B1] the sync exact-QAT ENTRY half-step: same +0.5/S bake math as the
    TTFS mapping-time compensation, folded BEFORE training so the ceil-kernel
    QAT enters through the half-step (entry 0.10 -> 0.85 measured with the
    deflated quantile) and can train it away (bias stays a live parameter)."""

    def test_plus_half_step_entry_fold_exact(self):
        from mimarsinan.mapping.support.bias_compensation import (
            apply_sync_exact_entry_half_step,
        )

        model = _Model([_perceptron()])
        folded = apply_sync_exact_entry_half_step(model, 4)
        assert folded == 1
        _assert_bias_exact(model.get_perceptrons()[0], [0.625, -1.375, 2.125])
        assert model.get_perceptrons()[0]._sync_entry_half_step_folded is True

    def test_idempotent_no_double_fold(self):
        from mimarsinan.mapping.support.bias_compensation import (
            apply_sync_exact_entry_half_step,
        )

        model = _Model([_perceptron()])
        apply_sync_exact_entry_half_step(model, 4)
        folded_again = apply_sync_exact_entry_half_step(model, 4)
        assert folded_again == 0
        _assert_bias_exact(model.get_perceptrons()[0], [0.625, -1.375, 2.125])

    def test_encoding_layer_skipped(self):
        from mimarsinan.mapping.support.bias_compensation import (
            apply_sync_exact_entry_half_step,
        )

        encoder = _perceptron()
        encoder.is_encoding_layer = True
        model = _Model([encoder])
        assert apply_sync_exact_entry_half_step(model, 4) == 0
        _assert_bias_exact(encoder, [0.5, -1.5, 2.0])

    def test_distinct_flag_from_the_mapping_time_bake(self):
        # The entry fold and the mapping-time bake are different decisions:
        # a model that got the entry fold must still be foldable by (or skip)
        # the mapping-time path on ITS marker alone.
        from mimarsinan.mapping.support.bias_compensation import (
            SYNC_ENTRY_HALF_STEP_FLAG,
            TTFS_COMP_BAKED_FLAG,
        )

        assert SYNC_ENTRY_HALF_STEP_FLAG != TTFS_COMP_BAKED_FLAG


class TestApplyLifHalfStepBiasCompensation:
    """[5v B3] the LIF analogue of the TTFS half-step bake: the deployed LIF
    rate grid is theta/T (floor), so folding +theta/(2T) per cycle turns the
    floor into nearest over the window AND head-starts every hop's first fire
    (t0_01: +13 pp value-domain measured; t0_05 control +1.2 pp)."""

    def test_plus_half_step_fold_exact(self):
        from mimarsinan.mapping.support.bias_compensation import (
            apply_lif_half_step_bias_compensation,
        )

        model = _Model([_perceptron()])
        folded = apply_lif_half_step_bias_compensation(model, 4)
        assert folded == 1
        # theta=1, T=4: shift/theta = 0.125 — identical math to the TTFS bake.
        _assert_bias_exact(model.get_perceptrons()[0], [0.625, -1.375, 2.125])
        assert model.get_perceptrons()[0]._lif_half_step_baked_into_bias is True

    def test_idempotent_no_double_fold(self):
        from mimarsinan.mapping.support.bias_compensation import (
            apply_lif_half_step_bias_compensation,
        )

        model = _Model([_perceptron()])
        apply_lif_half_step_bias_compensation(model, 4)
        assert apply_lif_half_step_bias_compensation(model, 4) == 0
        _assert_bias_exact(model.get_perceptrons()[0], [0.625, -1.375, 2.125])

    def test_encoding_layer_skipped(self):
        from mimarsinan.mapping.support.bias_compensation import (
            apply_lif_half_step_bias_compensation,
        )

        encoder = _perceptron()
        encoder.is_encoding_layer = True
        model = _Model([encoder])
        assert apply_lif_half_step_bias_compensation(model, 4) == 0
        _assert_bias_exact(encoder, [0.5, -1.5, 2.0])

    def test_uses_the_window_not_target_tq(self):
        # The LIF grid is theta/T (simulation window), NOT theta/Tq: at T=8
        # the fold halves relative to T=4.
        from mimarsinan.mapping.support.bias_compensation import (
            apply_lif_half_step_bias_compensation,
        )

        model = _Model([_perceptron()])
        apply_lif_half_step_bias_compensation(model, 8)
        _assert_bias_exact(model.get_perceptrons()[0], [0.5625, -1.4375, 2.0625])

    def test_fold_is_a_trainable_parameter_write(self):
        # [fbb2 finding] the fold must precede training: injected at mapping it
        # breaks the LIF bit-exact train<->deploy identity (t0_01 parity gate
        # refused at 0.9336; t0_05 slipped 1.0000 -> 0.9883). The bake writes
        # through to the RAW bias parameter so the QAT owns it from entry.
        from mimarsinan.mapping.support.bias_compensation import (
            apply_lif_half_step_bias_compensation,
        )

        model = _Model([_perceptron()])
        p = model.get_perceptrons()[0]
        assert p.layer.bias.requires_grad
        apply_lif_half_step_bias_compensation(model, 4)
        assert p.layer.bias.requires_grad


class TestHalfStepEntryFoldUnification:
    """The LIF and sync entry folds share one bake core (identical shift math);
    only the idempotency marker differs."""

    def test_lif_sync_and_core_write_identical_bias(self):
        from mimarsinan.mapping.support.bias_compensation import (
            apply_half_step_entry_fold,
            apply_lif_half_step_bias_compensation,
            apply_sync_exact_entry_half_step,
        )

        lif_model = _Model([_perceptron()])
        sync_model = _Model([_perceptron()])
        core_model = _Model([_perceptron()])
        assert apply_lif_half_step_bias_compensation(lif_model, 4) == 1
        assert apply_sync_exact_entry_half_step(sync_model, 4) == 1
        assert apply_half_step_entry_fold(core_model, 4, baked_flag="_probe") == 1
        lif_b = lif_model.get_perceptrons()[0].layer.bias.data
        sync_b = sync_model.get_perceptrons()[0].layer.bias.data
        core_b = core_model.get_perceptrons()[0].layer.bias.data
        assert torch.equal(lif_b, sync_b)
        assert torch.equal(lif_b, core_b)

    def test_markers_are_distinct(self):
        from mimarsinan.mapping.support.bias_compensation import (
            LIF_HALF_STEP_FLAG,
            SYNC_ENTRY_HALF_STEP_FLAG,
        )

        assert LIF_HALF_STEP_FLAG != SYNC_ENTRY_HALF_STEP_FLAG


class TestLifHalfStepEntryFoldGate:
    """[5v B3 relocation] the LIF half-step fold enters the weight-quant QAT
    (before quantization) so the QAT reconciles it — NOT at soft-core mapping,
    where the post-QAT injection broke the parity identity (t0_01 0.9336)."""

    def _run_gate(self, model, cfg):
        from types import SimpleNamespace

        from conftest import MockPipeline
        from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import (
            WeightQuantizationStep,
        )

        pipeline = MockPipeline(config=cfg)
        fake_step = SimpleNamespace(pipeline=pipeline)
        WeightQuantizationStep._apply_lif_half_step_entry_fold(fake_step, model)

    def _lif_cfg(self, *, knob=True, act_q=True):
        from conftest import default_config

        cfg = default_config()
        cfg["spiking_mode"] = "lif"
        cfg["simulation_steps"] = 4
        cfg["activation_quantization"] = act_q
        if knob:
            cfg["lif_half_step_bias"] = True
        return cfg

    def _model(self):
        from conftest import make_tiny_supermodel

        return make_tiny_supermodel()

    def test_applied_for_lif_with_knob(self):
        model = self._model()
        self._run_gate(model, self._lif_cfg())
        non_encoders = [
            p for p in model.get_perceptrons()
            if not getattr(p, "is_encoding_layer", False)
        ]
        assert non_encoders
        assert all(
            getattr(p, "_lif_half_step_baked_into_bias", False)
            for p in non_encoders
        )

    def test_knob_off_is_bit_identical(self):
        model = self._model()
        before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
        self._run_gate(model, self._lif_cfg(knob=False))
        for p, b in zip(model.get_perceptrons(), before):
            assert torch.equal(p.layer.bias.detach(), b)

    def test_non_lif_mode_never_folds(self):
        model = self._model()
        cfg = self._lif_cfg()
        cfg["spiking_mode"] = "ttfs_quantized"
        before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
        self._run_gate(model, cfg)
        for p, b in zip(model.get_perceptrons(), before):
            assert torch.equal(p.layer.bias.detach(), b)

    def test_no_activation_quantization_never_folds(self):
        model = self._model()
        before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
        self._run_gate(model, self._lif_cfg(act_q=False))
        for p, b in zip(model.get_perceptrons(), before):
            assert torch.equal(p.layer.bias.detach(), b)

    def test_mapping_step_no_longer_folds_lif(self):
        from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
            SoftCoreMappingStep,
        )

        assert not hasattr(
            SoftCoreMappingStep, "_apply_lif_half_step_bias_compensation"
        )
