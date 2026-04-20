"""Phase D3: Dead-code deletion contracts.

The refactor plan calls for removing:

  * ``mimarsinan.tuning.tolerance_calibration`` (and its test file)
  * Noise/scale decorators that were only ever used by now-dead tuners:
    ``NoisyDropout``, ``NoiseDecorator``, ``ScaleDecorator``, ``AnyDecorator``
  * The stochastic-mixing adjustment strategies that were superseded
    by Phase C1/C2: ``RandomMaskAdjustmentStrategy``,
    ``NestedAdjustmentStrategy``
  * ``AdaptationManager`` fields that no tuner writes to anymore:
    ``noise_rate``, ``scale_rate``, and the implicit noise-regularization
    branch inside ``update_activation``.
  * The corresponding ``NoiseTuner`` (never instantiated by any pipeline
    step).

These tests are purely negative: each asserts that a specific symbol
is **not** importable (or not present as an attribute).  They are the
rollback breakpoint if someone ever tries to re-introduce the dead
paths; deleting the production code is the fix for the failures.
"""

from __future__ import annotations

import importlib

import pytest


class TestToleranceCalibrationGone:
    def test_module_not_importable(self):
        with pytest.raises(ImportError):
            importlib.import_module("mimarsinan.tuning.tolerance_calibration")


class TestDeadDecoratorsGone:
    @pytest.mark.parametrize(
        "name",
        [
            "NoisyDropout",
            "NoiseDecorator",
            "ScaleDecorator",
            "AnyDecorator",
            "RandomMaskAdjustmentStrategy",
            "NestedAdjustmentStrategy",
        ],
    )
    def test_not_in_models_decorators(self, name):
        mod = importlib.import_module("mimarsinan.models.decorators")
        assert not hasattr(mod, name), (
            f"{name!r} is supposed to be deleted in Phase D3 but "
            f"still lives in mimarsinan.models.decorators"
        )

    @pytest.mark.parametrize(
        "name",
        [
            "NoisyDropout",
            "NoiseDecorator",
            "ScaleDecorator",
            "AnyDecorator",
            "RandomMaskAdjustmentStrategy",
            "NestedAdjustmentStrategy",
        ],
    )
    def test_not_reexported_via_layers(self, name):
        mod = importlib.import_module("mimarsinan.models.layers")
        assert not hasattr(mod, name), (
            f"{name!r} is supposed to be deleted in Phase D3 but is "
            f"still re-exported from mimarsinan.models.layers"
        )

    @pytest.mark.parametrize(
        "name",
        ["NoisyDropout", "ScaleDecorator", "NoiseDecorator"],
    )
    def test_not_reexported_via_models_package(self, name):
        mod = importlib.import_module("mimarsinan.models")
        assert not hasattr(mod, name)


class TestAdaptationManagerFieldsGone:
    def test_noise_rate_field_removed(self):
        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        am = AdaptationManager()
        assert not hasattr(am, "noise_rate"), (
            "noise_rate is no longer driven by any tuner; Phase D3 "
            "removes the field"
        )

    def test_scale_rate_field_removed(self):
        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        am = AdaptationManager()
        assert not hasattr(am, "scale_rate"), (
            "scale_rate has had no live writer for releases; Phase D3 "
            "removes the field"
        )

    def test_known_rate_fields_set_does_not_contain_dead_rates(self):
        import mimarsinan.tuning.adaptation_manager as am_mod

        assert "noise_rate" not in am_mod._KNOWN_RATE_FIELDS
        assert "scale_rate" not in am_mod._KNOWN_RATE_FIELDS

    def test_update_activation_has_no_noise_branch(self):
        """``update_activation`` must stop installing ``NoisyDropout``
        as a perceptron regularizer -- the branch is dead code."""
        import inspect

        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        src = inspect.getsource(AdaptationManager.update_activation)
        assert "NoisyDropout" not in src
        assert "noise_rate" not in src


class TestNoiseTunerGone:
    def test_noise_tuner_not_importable(self):
        with pytest.raises(ImportError):
            importlib.import_module("mimarsinan.tuning.tuners.noise_tuner")

    def test_noise_tuner_not_reexported(self):
        mod = importlib.import_module("mimarsinan.tuning.tuners")
        assert not hasattr(mod, "NoiseTuner")
