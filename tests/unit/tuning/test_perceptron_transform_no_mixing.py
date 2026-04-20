"""Phase C1 verification: PerceptronTransformTuner no longer mixes.

The legacy implementation interpolated FP and quantized weights via a
random Bernoulli mask with probability ``rate``.  With LSQ + STE the
quantizer is fully differentiable, so the mixing is replaced by a
deterministic "apply-the-new-transform" step.  These tests pin that
behaviour so any future refactor that re-introduces stochastic mixing
(or reverts to the random-mask pattern) fails loudly.
"""

from __future__ import annotations

import torch

from mimarsinan.tuning.tuners.perceptron_transform_tuner import (
    PerceptronTransformTuner,
)


class _TrackingTuner:
    """Stand-in for a PerceptronTransformTuner subclass.

    We don't need a real pipeline / trainer to exercise the mixing
    logic -- the only thing under test is the mixing helper itself.
    """

    def __init__(self):
        self._device = "cpu"
        self.calls = []

    def _get_previous_perceptron_transform(self, rate):
        def transform(p):
            self.calls.append(("prev", float(rate)))
        return transform

    def _get_new_perceptron_transform(self, rate):
        def transform(p):
            self.calls.append(("new", float(rate)))
        return transform


def _make_combined_tuner():
    return type(
        "_T",
        (PerceptronTransformTuner,),
        {
            "__init__": _TrackingTuner.__init__,
            "_get_previous_perceptron_transform": _TrackingTuner._get_previous_perceptron_transform,
            "_get_new_perceptron_transform": _TrackingTuner._get_new_perceptron_transform,
        },
    )()


class TestNoStochasticMixing:
    def test_does_not_call_previous_transform_at_full_rate(self):
        """The legacy code always evaluated ``_get_previous_perceptron_transform``
        because the random-mask mixer needed a 'prev' tensor to blend
        against.  After Phase C1 only the 'new' branch is invoked."""
        t = _make_combined_tuner()
        dummy = torch.nn.Linear(2, 2)
        t._mixed_perceptron_transform(dummy, rate=1.0)
        assert ("prev", 1.0) not in t.calls
        assert ("new", 1.0) in t.calls

    def test_does_not_call_previous_transform_at_partial_rate(self):
        t = _make_combined_tuner()
        dummy = torch.nn.Linear(2, 2)
        t._mixed_perceptron_transform(dummy, rate=0.3)
        assert all(tag != "prev" for tag, _ in t.calls)
        assert ("new", 0.3) in t.calls

    def test_zero_rate_skips_transform_entirely(self):
        t = _make_combined_tuner()
        dummy = torch.nn.Linear(2, 2)
        t._mixed_perceptron_transform(dummy, rate=0.0)
        assert t.calls == [], (
            f"rate=0 must be a no-op; got calls={t.calls}"
        )

    def test_no_random_mask_in_source(self):
        """Sentinel against re-introducing a random-mask mixer."""
        import inspect

        source = inspect.getsource(
            PerceptronTransformTuner._mixed_perceptron_transform
        )
        assert "torch.rand" not in source
        assert "random_mask" not in source
