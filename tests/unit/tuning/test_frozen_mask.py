"""Tests for the frozen-mask behaviour of PerceptronTransformTuner._mix_params.

The legacy behaviour redrew the random mask on every call, which meant every
training step saw a different stochastic realisation of "which weights are
transformed vs pristine". Combined with the rate-aware
NormalizationAwarePerceptronQuantization transform (which now produces a
coherent ``rate * q(w) + (1 - rate) * w`` perturbation on its own), the
redrawn mask was pure noise -- it made training chase a moving-target loss
and made validation measure a different realisation than training had just
optimised.

The fix freezes the mask per probe cycle: a fresh mask is generated the
first time a given perceptron-parameter pair is encountered during a given
``_mixed_transform(rate)`` closure, and then reused across every
training-step forward pass and every validation call within that cycle.
When ``_update_and_evaluate`` is called with a new rate, a fresh closure
(with a fresh cache) is created, so the mask is regenerated per probe.

These tests lock in:
  - consistency across repeated calls within the same cycle,
  - independence across cycles,
  - deterministic endpoints (rate == 0 -> all-False, rate == 1 -> all-True),
  - behaviour on scalar (0-d) params.
"""

import copy

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)
from mimarsinan.tuning.tuners.perceptron_transform_tuner import (
    PerceptronTransformTuner,
)


def _make_pipeline(tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = 0.05
    return MockPipeline(config=cfg, working_directory=str(tmp_path))


def _make_tuner(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    model = make_tiny_supermodel()
    am = AdaptationManager()
    return NormalizationAwarePerceptronQuantizationTuner(
        pipeline,
        model,
        quantization_bits=8,
        target_accuracy=0.9,
        lr=0.001,
        adaptation_manager=am,
    )


class TestMaskFrozenWithinCycle:
    """Within a single ``_mixed_transform(rate)`` closure, repeated
    invocations on the same perceptron slot (the real usage pattern --
    ``PerceptronTransformTrainer`` pre-allocates a persistent ``temp_p``
    per slot and refreshes its params from ``aux_model`` before each
    call) must produce the same effective weights (i.e. the same mask
    is reused)."""

    def test_same_slot_same_cycle_yields_identical_result(self, tmp_path):
        """The trainer mutates ``temp_p`` in place each step -- between
        two successive invocations the perceptron object identity is
        preserved and the params are refreshed to the original FP
        state. The cache must keep the mask stable across those
        successive invocations."""
        tuner = _make_tuner(tmp_path)

        perceptrons = list(tuner.model.get_perceptrons())
        assert perceptrons, "test supermodel must have at least one perceptron"

        base = copy.deepcopy(perceptrons[0]).to(tuner._device)
        slot = copy.deepcopy(base).to(tuner._device)

        transform = tuner._mixed_transform(0.25)
        transform(slot)
        first_out = {k: v.clone() for k, v in slot.state_dict().items()}

        # Refresh slot from the original FP state (as the trainer does
        # from aux_model) and transform again.
        slot.load_state_dict(base.state_dict())
        transform(slot)
        second_out = {k: v.clone() for k, v in slot.state_dict().items()}

        for k in first_out:
            assert torch.allclose(first_out[k], second_out[k], atol=0.0), (
                f"Parameter {k} changed between calls within the same "
                f"cycle for the same slot: mask is not frozen."
            )

    def test_multiple_training_steps_see_same_model(self, tmp_path):
        """Simulate 5 successive 'training steps' on the same slot (the
        trainer calls the transformation once per step on a persistent
        ``temp_p``, refreshed from aux_model each step). All must yield
        bit-identical post-transform perceptron state."""
        tuner = _make_tuner(tmp_path)
        perceptrons = list(tuner.model.get_perceptrons())
        base = copy.deepcopy(perceptrons[0]).to(tuner._device)
        slot = copy.deepcopy(base).to(tuner._device)

        transform = tuner._mixed_transform(0.3)

        results = []
        for _ in range(5):
            slot.load_state_dict(base.state_dict())
            transform(slot)
            results.append({k: v.clone() for k, v in slot.state_dict().items()})

        ref = results[0]
        for i, r in enumerate(results[1:], start=1):
            for k in ref:
                assert torch.allclose(ref[k], r[k], atol=0.0), (
                    f"Training step {i + 1} produced different values for "
                    f"parameter '{k}' than step 1: mask is being redrawn."
                )


class TestMaskRefreshedAcrossCycles:
    """A new _mixed_transform(rate) closure must produce a fresh mask, so
    the cache is bound to the closure not to the tuner."""

    def test_different_rate_yields_different_mask(self, tmp_path):
        torch.manual_seed(0)
        tuner = _make_tuner(tmp_path)
        perceptrons = list(tuner.model.get_perceptrons())

        base = copy.deepcopy(perceptrons[0]).to(tuner._device)

        transform_low = tuner._mixed_transform(0.3)
        p_low = copy.deepcopy(base).to(tuner._device)
        transform_low(p_low)

        transform_high = tuner._mixed_transform(0.7)
        p_high = copy.deepcopy(base).to(tuner._device)
        transform_high(p_high)

        # Parameters should differ between cycles for at least weight /
        # bias (scalar scales might coincide by chance, so we don't check
        # every key).
        assert not torch.allclose(
            p_low.layer.weight.data, p_high.layer.weight.data, atol=1e-9
        ), "Different rates must produce different post-transform weights."

    def test_same_rate_different_closure_may_differ(self, tmp_path):
        """Different closures at the same rate are statistically independent:
        with a large enough weight tensor they should almost never coincide
        element-wise (they both redraw the mask)."""
        torch.manual_seed(123)
        tuner = _make_tuner(tmp_path)
        perceptrons = list(tuner.model.get_perceptrons())
        base = copy.deepcopy(perceptrons[0]).to(tuner._device)

        transform_a = tuner._mixed_transform(0.4)
        p_a = copy.deepcopy(base).to(tuner._device)
        transform_a(p_a)

        transform_b = tuner._mixed_transform(0.4)
        p_b = copy.deepcopy(base).to(tuner._device)
        transform_b(p_b)

        # Large-ish tensors at rate=0.4: probability of identical masks is
        # effectively zero. We only require *some* element to differ.
        assert not torch.equal(p_a.layer.weight.data, p_b.layer.weight.data), (
            "Two independent closures at the same rate produced bit-identical "
            "weights. The cache is probably shared across closures, not per-"
            "closure."
        )


class TestEndpointsAreDeterministic:
    """Rate == 0 must be identity; rate == 1 must match the fully
    transformed perceptron, regardless of randomness."""

    def test_rate_zero_is_identity(self, tmp_path):
        torch.manual_seed(42)
        tuner = _make_tuner(tmp_path)
        perceptrons = list(tuner.model.get_perceptrons())
        base = copy.deepcopy(perceptrons[0]).to(tuner._device)

        transform = tuner._mixed_transform(0.0)
        p = copy.deepcopy(base).to(tuner._device)
        transform(p)

        for k, v in base.state_dict().items():
            assert torch.allclose(p.state_dict()[k], v, atol=1e-6), (
                f"rate=0.0 changed parameter '{k}' -- must be a true identity."
            )

    def test_rate_one_matches_unmixed_new_transform(self, tmp_path):
        """At rate=1.0 the mask is all-True, so _mix_params returns
        new_param and the result should match applying
        _get_new_perceptron_transform(1.0) alone."""
        torch.manual_seed(7)
        tuner = _make_tuner(tmp_path)
        perceptrons = list(tuner.model.get_perceptrons())
        base = copy.deepcopy(perceptrons[0]).to(tuner._device)

        p_mixed = copy.deepcopy(base).to(tuner._device)
        tuner._mixed_transform(1.0)(p_mixed)

        p_direct = copy.deepcopy(base).to(tuner._device)
        tuner._get_new_perceptron_transform(1.0)(p_direct)

        for k in p_mixed.state_dict():
            assert torch.allclose(
                p_mixed.state_dict()[k], p_direct.state_dict()[k], atol=1e-6
            ), (
                f"At rate=1.0, _mixed_transform and _get_new_perceptron_transform "
                f"disagree on parameter '{k}'."
            )


class TestScalarParamMasking:
    """The cache must handle 0-d (scalar) parameters without shape errors."""

    def test_scalar_param_is_masked_once_per_cycle(self, tmp_path):
        """Perceptron has several 0-d scalar Parameters (parameter_scale,
        activation_scale, input_activation_scale, scale_factor). Repeated
        transform calls on the same slot within one cycle must keep each
        scalar stable."""
        tuner = _make_tuner(tmp_path)
        perceptrons = list(tuner.model.get_perceptrons())
        base = copy.deepcopy(perceptrons[0]).to(tuner._device)
        slot = copy.deepcopy(base).to(tuner._device)

        transform = tuner._mixed_transform(0.5)

        transform(slot)
        first = {k: v.clone() for k, v in slot.state_dict().items()}

        slot.load_state_dict(base.state_dict())
        transform(slot)
        second = {k: v.clone() for k, v in slot.state_dict().items()}

        scalar_names = [
            "parameter_scale",
            "activation_scale",
            "input_activation_scale",
            "scale_factor",
        ]
        for name in scalar_names:
            if name in first:
                assert torch.allclose(first[name], second[name], atol=0.0), (
                    f"Scalar param '{name}' changed between calls within "
                    "the same cycle on the same slot."
                )
