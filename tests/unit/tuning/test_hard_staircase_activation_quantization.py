"""Post-DSQ contract for activation quantisation in ``AdaptationManager``.

Phase C2's differentiable soft quantiser (DSQ) has been dropped.  Any
non-zero ``quantization_rate`` -- whether the global scalar or a
per-perceptron override -- must install the *hard*
``QuantizeDecorator`` (backed by ``StaircaseFunction``) in the
perceptron's activation chain.  There is no more rate-dependent soft
quantiser; the ``rate`` is now a binary "install or don't" flag at
the decorator level, and the cycle-by-cycle rollout is driven by the
concrete ``ActivationQuantizationTuner`` via
``set_per_perceptron_rate``.

These tests pin:

1. ``get_rate_adjusted_quantization_decorator`` no longer constructs
   a ``DSQQuantizeDecorator`` for any quant_rate in (0, 1).
2. ``update_activation`` installs the hard ``QuantizeDecorator`` for
   every ``quant_rate != 0`` -- including small rates (e.g. 0.1) that
   used to land on DSQ.
3. Per-perceptron overrides correctly select which perceptrons get
   the hard quantiser when the global scalar is 0.
4. ``quant_rate == 0`` installs *no* quantiser in the activation
   chain (the ``StaircaseFunction``/``QuantizeDecorator`` is absent).
"""

from __future__ import annotations

import pytest

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.models.decorators import (
    NestedDecoration,
    QuantizeDecorator,
    RateAdjustedDecorator,
    ShiftDecorator,
)
from mimarsinan.tuning.adaptation_manager import AdaptationManager


def _flatten_decorators(activation) -> list:
    """Walk the perceptron's ``TransformedActivation`` decorator list and
    flatten any nested ``NestedDecoration`` wrappers so we can look for
    decorator instances by type without caring about depth."""
    decorators = list(activation.decorators)
    out: list = []
    while decorators:
        d = decorators.pop(0)
        if isinstance(d, NestedDecoration):
            decorators = list(d.decorators) + decorators
        else:
            out.append(d)
    return out


def _make_manager_and_perceptron(tmp_path):
    pipeline = MockPipeline(config=default_config(), working_directory=str(tmp_path))
    pipeline.config["target_tq"] = 8
    model = make_tiny_supermodel()
    am = AdaptationManager()
    p = model.get_perceptrons()[0]
    return pipeline, am, p


class TestNoDSQInActivationChain:
    """Sentinel: DSQ must not appear in any activation chain produced by
    ``AdaptationManager.update_activation`` under the new hard-staircase
    rollout contract."""

    @pytest.mark.parametrize("rate", [0.1, 0.25, 0.5, 0.75, 1.0])
    def test_any_nonzero_rate_installs_hard_quantize_decorator(self, tmp_path, rate):
        pipeline, am, p = _make_manager_and_perceptron(tmp_path)
        am.quantization_rate = rate
        am.update_activation(pipeline.config, p)
        flat = _flatten_decorators(p.activation)
        types = {type(d) for d in flat}
        assert QuantizeDecorator in types, (
            f"at quant_rate={rate} the hard QuantizeDecorator must be "
            f"installed (post-DSQ); got decorators {types}"
        )

    @pytest.mark.parametrize("rate", [0.0, 0.1, 0.5, 1.0])
    def test_dsq_decorator_is_never_installed(self, tmp_path, rate):
        pipeline, am, p = _make_manager_and_perceptron(tmp_path)
        am.quantization_rate = rate
        am.update_activation(pipeline.config, p)
        flat = _flatten_decorators(p.activation)
        for d in flat:
            assert type(d).__name__ != "DSQQuantizeDecorator", (
                "DSQ was deleted; no activation chain should produce a "
                f"DSQQuantizeDecorator (got one at quant_rate={rate})"
            )

    def test_zero_rate_installs_no_quantiser(self, tmp_path):
        pipeline, am, p = _make_manager_and_perceptron(tmp_path)
        am.quantization_rate = 0.0
        am.update_activation(pipeline.config, p)
        flat = _flatten_decorators(p.activation)
        for d in flat:
            assert not isinstance(d, QuantizeDecorator), (
                "quant_rate=0 must not install a quantiser in the "
                "activation chain"
            )
            assert type(d).__name__ != "DSQQuantizeDecorator"


class TestRateAdjustedWrapperNotUsedForQuantization:
    """The legacy rate-mixed ``RateAdjustedDecorator(QuantizeDecorator, ...)``
    chain must not reappear.  The hard quantiser is installed unconditionally
    once the rate is non-zero -- there is no rate-based blending anymore."""

    @pytest.mark.parametrize("rate", [0.1, 0.5, 1.0])
    def test_no_rate_adjusted_wrapping_the_quantizer(self, tmp_path, rate):
        pipeline, am, p = _make_manager_and_perceptron(tmp_path)
        am.quantization_rate = rate
        am.update_activation(pipeline.config, p)
        flat = _flatten_decorators(p.activation)
        for d in flat:
            if isinstance(d, RateAdjustedDecorator):
                assert not isinstance(d.decorator, QuantizeDecorator), (
                    "RateAdjustedDecorator must not wrap the quantiser "
                    f"(found at quant_rate={rate})"
                )


class TestShiftIsStillNestedWithQuantizer:
    """The shift-then-quantize ordering is preserved post-DSQ: the shift
    decorator that undoes the clamp/shift bias still lives inside the
    ``NestedDecoration`` alongside the hard ``QuantizeDecorator``."""

    def test_nested_holds_shift_and_quantize(self, tmp_path):
        pipeline, am, p = _make_manager_and_perceptron(tmp_path)
        am.quantization_rate = 1.0
        am.update_activation(pipeline.config, p)

        top_level = list(p.activation.decorators)
        nested = [d for d in top_level if isinstance(d, NestedDecoration)]
        assert nested, (
            "a NestedDecoration wrapping ShiftDecorator + QuantizeDecorator "
            "must be present at non-zero quant_rate"
        )
        inner_types = [type(d) for d in nested[0].decorators]
        assert QuantizeDecorator in inner_types
        assert ShiftDecorator in inner_types


class TestPerPerceptronOverridesGateQuantization:
    """Per-perceptron overrides (Phase B1) must correctly gate WHICH
    perceptrons receive the hard quantiser when the global scalar is
    zero.  This is how ``ActivationQuantizationTuner`` will roll out
    quantisation one layer at a time."""

    def test_override_installs_quantizer_only_on_targeted_perceptron(
        self, tmp_path
    ):
        pipeline = MockPipeline(
            config=default_config(), working_directory=str(tmp_path)
        )
        pipeline.config["target_tq"] = 8
        model = make_tiny_supermodel()
        am = AdaptationManager()

        perceptrons = list(model.get_perceptrons())
        # Give them distinct stable names so per-perceptron overrides
        # have a unique key.  The tiny test model defaults to
        # ``.name == "Perceptron"`` for every perceptron; the real
        # pipeline uses the original ``nn.Module`` attribute name.
        for i, p in enumerate(perceptrons):
            p.name = f"layer_{i}"

        # Global rate off; only layer_0 is quantised.
        am.quantization_rate = 0.0
        am.set_per_perceptron_rate("quantization_rate", "layer_0", 1.0)
        for p in perceptrons:
            am.update_activation(pipeline.config, p)

        flat_0 = _flatten_decorators(perceptrons[0].activation)
        flat_1 = _flatten_decorators(perceptrons[1].activation)
        types_0 = {type(d) for d in flat_0}
        types_1 = {type(d) for d in flat_1}

        assert QuantizeDecorator in types_0, (
            "layer_0 (override=1.0) must receive the hard quantiser"
        )
        assert QuantizeDecorator not in types_1, (
            "layer_1 (no override, global=0) must NOT receive the quantiser"
        )

    def test_override_can_disable_quantization_for_one_perceptron(
        self, tmp_path
    ):
        """Inverse: global scalar = 1.0 but one perceptron overrides
        to 0.0 -- only that perceptron should be left unquantised.
        Required for the rollback path, which un-selects a perceptron
        when its cycle fails."""
        pipeline = MockPipeline(
            config=default_config(), working_directory=str(tmp_path)
        )
        pipeline.config["target_tq"] = 8
        model = make_tiny_supermodel()
        am = AdaptationManager()

        perceptrons = list(model.get_perceptrons())
        for i, p in enumerate(perceptrons):
            p.name = f"layer_{i}"

        am.quantization_rate = 1.0
        am.set_per_perceptron_rate("quantization_rate", "layer_1", 0.0)
        for p in perceptrons:
            am.update_activation(pipeline.config, p)

        flat_0 = _flatten_decorators(perceptrons[0].activation)
        flat_1 = _flatten_decorators(perceptrons[1].activation)
        types_0 = {type(d) for d in flat_0}
        types_1 = {type(d) for d in flat_1}
        assert QuantizeDecorator in types_0
        assert QuantizeDecorator not in types_1


class TestGetRateAdjustedQuantizationDecoratorReturnsHardOnly:
    """Direct test of the factory: it must *always* return a chain
    whose inner quantiser is the hard ``QuantizeDecorator`` -- never
    DSQ -- regardless of ``quant_rate``."""

    @pytest.mark.parametrize("rate", [0.01, 0.25, 0.5, 0.99, 1.0])
    def test_factory_always_returns_hard_quantizer(self, tmp_path, rate):
        pipeline, am, p = _make_manager_and_perceptron(tmp_path)
        nested = am.get_rate_adjusted_quantization_decorator(
            pipeline.config, p, quant_rate=rate, shift_rate=0.0
        )
        assert isinstance(nested, NestedDecoration)
        inner_types = [type(d) for d in nested.decorators]
        assert QuantizeDecorator in inner_types, (
            f"factory at rate={rate} did not produce a hard quantiser; "
            f"got inner decorators {inner_types}"
        )
        for d in nested.decorators:
            assert type(d).__name__ != "DSQQuantizeDecorator"
