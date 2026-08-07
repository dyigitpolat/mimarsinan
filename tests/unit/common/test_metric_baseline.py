"""The baseline-agreement contract: a measured number must be checked against its record.

The incident this exists for: a model that normalizes its input INTERNALLY was
fed through an additional ``torchvision`` ``Normalize``. The harness reported
60.83% against a recorded 92.30% and nothing raised -- the number was plausible,
self-consistent and wrong. That is the same failure family as the NaN-batch
corruption ``data_handling.batch_integrity`` guards: a rare fault that does not
crash, it emits a number.
"""

import math

import pytest
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from mimarsinan.common.measurement import (
    BASELINE_FLOOR,
    BASELINE_SIGMAS,
    BaselineMismatchError,
    MetricTolerance,
    assert_matches_recorded_baseline,
)
from mimarsinan.data_handling.preprocessing import resolve_preprocessing


class TestMetricToleranceIsTheOnlyDefinitionOfMatching:
    def test_identical_values_match_under_a_zero_tolerance(self):
        assert MetricTolerance().matches(0.923, 0.923)

    def test_a_difference_never_matches_under_a_zero_tolerance(self):
        assert not MetricTolerance().matches(0.923, 0.9230001)

    def test_absolute_tolerance_matches_exactly_at_its_boundary(self):
        # Dyadic values: the boundary is the boundary, not a rounding artifact.
        tolerance = MetricTolerance(abs_tol=0.125)
        assert tolerance.matches(0.5, 0.375)
        assert tolerance.matches(0.5, 0.625)

    def test_absolute_tolerance_rejects_one_ulp_beyond_its_boundary(self):
        tolerance = MetricTolerance(abs_tol=0.125)
        assert not tolerance.matches(0.5, math.nextafter(0.375, 0.0))
        assert not tolerance.matches(0.5, math.nextafter(0.625, 1.0))

    def test_relative_tolerance_scales_with_the_larger_magnitude(self):
        tolerance = MetricTolerance(rel_tol=0.25)
        assert tolerance.matches(4.0, 3.0)
        assert not tolerance.matches(4.0, math.nextafter(3.0, 0.0))

    def test_the_verdict_does_not_depend_on_argument_order(self):
        tolerance = MetricTolerance(abs_tol=0.01, rel_tol=0.25)
        for a, b in ((4.0, 3.0), (0.5, 0.375), (1.0, 0.0)):
            assert tolerance.matches(a, b) == tolerance.matches(b, a)

    def test_a_bound_is_reported_so_a_caller_can_state_it(self):
        bound = MetricTolerance(abs_tol=0.01, rel_tol=0.5).bound(0.8, 0.4)
        assert bound == pytest.approx(0.41)

    def test_a_negative_tolerance_is_refused(self):
        with pytest.raises(ValueError):
            MetricTolerance(abs_tol=-0.01)
        with pytest.raises(ValueError):
            MetricTolerance(rel_tol=-0.01)

    def test_a_non_finite_tolerance_is_refused(self):
        with pytest.raises(ValueError):
            MetricTolerance(abs_tol=float("inf"))
        with pytest.raises(ValueError):
            MetricTolerance(rel_tol=float("nan"))


class TestSampledProportionTolerance:
    """A tolerance quoted without the size of the sample it judges is a guess."""

    def test_it_is_the_documented_sigma_multiple_of_the_standard_error_plus_the_floor(self):
        expected, n = 0.923, 250
        standard_error = math.sqrt(expected * (1.0 - expected) / n)
        tolerance = MetricTolerance.for_sampled_proportion(expected, n)
        assert tolerance.rel_tol == 0.0
        assert tolerance.abs_tol == pytest.approx(
            BASELINE_FLOOR + BASELINE_SIGMAS * standard_error
        )

    def test_it_tightens_as_the_sample_grows(self):
        small = MetricTolerance.for_sampled_proportion(0.9, 100)
        large = MetricTolerance.for_sampled_proportion(0.9, 10_000)
        assert large.abs_tol < small.abs_tol

    def test_it_is_honoured_at_its_boundary(self):
        tolerance = MetricTolerance.for_sampled_proportion(0.9, 100)
        assert tolerance.matches(0.9, 0.9 - tolerance.abs_tol * (1 - 1e-9))
        assert not tolerance.matches(0.9, 0.9 - tolerance.abs_tol * (1 + 1e-9))

    def test_a_measurement_on_no_samples_is_not_a_measurement(self):
        with pytest.raises(ValueError):
            MetricTolerance.for_sampled_proportion(0.9, 0)

    def test_a_proportion_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError):
            MetricTolerance.for_sampled_proportion(1.5, 100)


class TestAssertMatchesRecordedBaseline:
    def _assert(self, measured, expected, **kwargs):
        return assert_matches_recorded_baseline(
            measured,
            expected=expected,
            tolerance=kwargs.pop("tolerance", MetricTolerance(abs_tol=0.05)),
            observable=kwargs.pop("observable", "the model's validation accuracy"),
            recorded_as=kwargs.pop("recorded_as", "weight set 'w1'"),
            **kwargs,
        )

    def test_a_matching_baseline_passes_and_returns_the_measured_value(self):
        assert self._assert(0.9210, 0.9230) == pytest.approx(0.9210)

    def test_a_mismatch_raises_with_both_numbers_in_the_message(self):
        with pytest.raises(BaselineMismatchError) as err:
            self._assert(0.6083, 0.9230)
        message = str(err.value)
        assert "0.6083" in message
        assert "0.9230" in message

    def test_the_message_states_the_gap_and_the_tolerance_it_broke(self):
        with pytest.raises(BaselineMismatchError) as err:
            self._assert(0.6083, 0.9230)
        message = str(err.value)
        assert "-0.3147" in message
        assert "0.05" in message

    def test_the_message_names_the_observable_and_the_record(self):
        with pytest.raises(BaselineMismatchError) as err:
            self._assert(0.6083, 0.9230, observable="preloaded top-1", recorded_as="set 'v1'")
        message = str(err.value)
        assert "preloaded top-1" in message
        assert "set 'v1'" in message

    def test_preprocessing_is_named_as_a_likely_cause(self):
        with pytest.raises(BaselineMismatchError) as err:
            self._assert(0.6083, 0.9230)
        assert "preprocessing" in str(err.value)
        assert "normalizes" in str(err.value)

    def test_caller_supplied_causes_are_named_before_the_generic_ones(self):
        with pytest.raises(BaselineMismatchError) as err:
            self._assert(0.6083, 0.9230, causes=("this run resized to 32 not 224",))
        message = str(err.value)
        assert message.index("resized to 32") < message.index("preprocessing")

    def test_a_non_finite_measurement_raises_rather_than_being_compared(self):
        with pytest.raises(BaselineMismatchError) as err:
            self._assert(float("nan"), 0.9230, tolerance=MetricTolerance(abs_tol=1.0))
        assert "0.9230" in str(err.value)

    def test_a_measurement_above_the_record_is_a_mismatch_too(self):
        with pytest.raises(BaselineMismatchError) as err:
            self._assert(0.9990, 0.9230)
        assert "+0.0760" in str(err.value)

    def test_the_failure_is_not_a_value_or_type_error_a_caller_may_swallow(self):
        assert not issubclass(BaselineMismatchError, (ValueError, TypeError))


# --------------------------------------------------------------------------
# The incident, reproduced: an evaluation harness that double-normalizes.
# --------------------------------------------------------------------------

_RECORDED_ACCURACY = 0.923
"""The number the record carries -- external to this run, as a record must be."""

_MODEL_MEAN, _MODEL_STD = 0.5, 0.25
_DATASET_MEAN, _DATASET_STD = 0.5, 0.5
_SAMPLES = 260
_MISLABELLED_EVERY = 13


class _InternallyNormalizingClassifier(nn.Module):
    """Normalizes its OWN input, then thresholds the mean -- the incident's model shape."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(_MODEL_MEAN))
        self.register_buffer("std", torch.tensor(_MODEL_STD))

    def forward(self, x):
        z = (x - self.mean) / self.std
        score = z.flatten(1).mean(dim=1, keepdim=True)
        return torch.cat([torch.zeros_like(score), score], dim=1)


class _ConstantIntensityImages(torch.utils.data.Dataset):
    """Deterministic 1x2x2 images of one intensity each, labelled ``intensity > 0.5``."""

    def __init__(self, transform):
        step = 0.96 / (_SAMPLES - 1)
        self.intensities = [0.02 + i * step for i in range(_SAMPLES)]
        self.transform = transform

    def __len__(self):
        return _SAMPLES

    def __getitem__(self, index):
        intensity = self.intensities[index]
        label = int(intensity > 0.5)
        if index % _MISLABELLED_EVERY == 0:
            label = 1 - label
        image = torch.full((1, 2, 2), intensity, dtype=torch.float32)
        return self.transform(image), label


def _measure_accuracy(model, transform) -> tuple[float, int]:
    """A minimal evaluation harness: preprocess, forward, count, report a number."""
    loader = torch.utils.data.DataLoader(
        _ConstantIntensityImages(transform), batch_size=25, num_workers=0
    )
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            correct += int(model(x).argmax(dim=1).eq(y).sum())
            total += int(y.shape[0])
    return correct / total, total


def _recorded_preprocessing():
    return transforms.Compose([])


def _double_normalizing_preprocessing():
    spec = resolve_preprocessing(
        {"normalize": {"mean": [_DATASET_MEAN], "std": [_DATASET_STD]}}
    )
    assert spec is not None
    return spec.compose([])


class TestDoubleNormalizationIsCaughtNotReported:
    """The regression: the harness that double-normalizes must fail, not return a number."""

    def test_the_recorded_preprocessing_reproduces_the_recorded_baseline(self):
        measured, samples = _measure_accuracy(
            _InternallyNormalizingClassifier(), _recorded_preprocessing()
        )
        assert measured == pytest.approx(_RECORDED_ACCURACY, abs=1e-3)
        assert assert_matches_recorded_baseline(
            measured,
            expected=_RECORDED_ACCURACY,
            tolerance=MetricTolerance.for_sampled_proportion(
                _RECORDED_ACCURACY, samples
            ),
            observable="the classifier's accuracy",
            recorded_as="the recorded baseline",
        ) == pytest.approx(measured)

    def test_the_wrong_number_is_plausible_which_is_why_nothing_noticed(self):
        wrong, _ = _measure_accuracy(
            _InternallyNormalizingClassifier(), _double_normalizing_preprocessing()
        )
        assert 0.5 < wrong < _RECORDED_ACCURACY
        assert math.isfinite(wrong)

    def test_the_baseline_contract_catches_the_double_normalized_harness(self):
        wrong, samples = _measure_accuracy(
            _InternallyNormalizingClassifier(), _double_normalizing_preprocessing()
        )
        with pytest.raises(BaselineMismatchError) as err:
            assert_matches_recorded_baseline(
                wrong,
                expected=_RECORDED_ACCURACY,
                tolerance=MetricTolerance.for_sampled_proportion(
                    _RECORDED_ACCURACY, samples
                ),
                observable="the classifier's accuracy",
                recorded_as="the recorded baseline",
            )
        message = str(err.value)
        assert f"{wrong:.6f}" in message
        assert f"{_RECORDED_ACCURACY:.6f}" in message
        assert "preprocessing" in message
