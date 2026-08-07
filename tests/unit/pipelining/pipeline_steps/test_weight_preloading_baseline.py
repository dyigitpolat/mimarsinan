"""WeightPreloadingStep checks the number it reports against the record it loaded.

A preload that silently produces a plausible-but-wrong accuracy is the incident
this wiring exists for: the recorded expectation is right there in the weight
set, so measuring without asserting against it is a choice to not look.
"""

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, default_config

from mimarsinan.common.measurement import BaselineMismatchError
from mimarsinan.common.pretrained import PretrainedWeightSet
from mimarsinan.data_handling.data_provider import ClassificationMode, DataProvider
from mimarsinan.pipelining.pipeline_steps.config.weight_preloading_step import (
    WeightPreloadingStep,
)

_INPUT_SHAPE = (1, 2, 2)
_NUM_CLASSES = 2
_SIZE = 64
_RECORDED_ACCURACY = 0.9375


class _IntensityDataset(torch.utils.data.Dataset):
    """Constant-intensity images labelled ``intensity > 0.5``; no RNG anywhere."""

    def __len__(self):
        return _SIZE

    def __getitem__(self, index):
        intensity = (index + 0.5) / _SIZE
        image = torch.full(_INPUT_SHAPE, intensity, dtype=torch.float32)
        return image, int(intensity > 0.5)


class _IntensityProvider(DataProvider):
    def __init__(self, datasets_path=""):
        super().__init__(datasets_path, seed=0)
        self._dataset = _IntensityDataset()

    def _get_training_dataset(self):
        return self._dataset

    def _get_validation_dataset(self):
        return self._dataset

    def _get_test_dataset(self):
        return self._dataset

    def get_prediction_mode(self):
        return ClassificationMode(_NUM_CLASSES)

    def get_input_shape(self):
        return _INPUT_SHAPE

    def get_output_shape(self):
        return _NUM_CLASSES

    def get_training_batch_size(self):
        return _SIZE


class _IntensityProviderFactory:
    def __init__(self):
        self._provider = None

    def create(self):
        if self._provider is None:
            self._provider = _IntensityProvider()
        return self._provider


def _threshold_model(bias: float) -> nn.Module:
    """Predicts class 1 iff the image mean exceeds ``-bias``."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, _NUM_CLASSES))
    linear = model[1]
    assert isinstance(linear, nn.Linear)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[1].fill_(0.25)
        linear.bias.zero_()
        linear.bias[1] = bias
    return model


def _checkpoint(tmp_path, bias: float):
    path = tmp_path / f"weights_{bias}.pt"
    torch.save(_threshold_model(bias).state_dict(), path)
    return path


def _weight_set(**overrides) -> PretrainedWeightSet:
    facts = dict(
        id="tiny_v1",
        label="Tiny (V1)",
        task="image classification",
        dataset="constant-intensity images",
        input_shape=_INPUT_SHAPE,
        num_classes=_NUM_CLASSES,
        source="checkpoint",
        expected_accuracy=_RECORDED_ACCURACY,
        preprocessing={"mean": [0.5], "std": [0.25]},
    )
    facts.update(overrides)
    return PretrainedWeightSet(**facts)  # type: ignore[arg-type]


def _make_step(tmp_path, *, bias: float, weight_set: PretrainedWeightSet):
    config = default_config()
    config.update(
        {
            "input_shape": _INPUT_SHAPE,
            "num_classes": _NUM_CLASSES,
            "num_workers": 0,
            "finetune_epochs": 0,
            "weight_source": str(_checkpoint(tmp_path, bias)),
            "preload_weights": True,
            "pretrained_weight_sets": [weight_set.as_dict()],
            "pretrained_weight_set": weight_set.id,
        }
    )
    pipeline = MockPipeline(
        config=config,
        working_directory=str(tmp_path / "cache"),
        data_provider_factory=_IntensityProviderFactory(),
    )
    pipeline.seed("model", _threshold_model(0.0))
    pipeline.seed("model_builder", object())

    step = WeightPreloadingStep(pipeline)
    step.name = "WeightPreloading"
    pipeline.prepare_step(step)
    return step


class TestWeightPreloadingAssertsItsRecordedBaseline:
    def test_a_load_that_reproduces_the_record_passes(self, tmp_path):
        step = _make_step(tmp_path, bias=-0.5, weight_set=_weight_set())
        step.run()
        assert step.trainer is not None
        assert step.trainer.validate() == pytest.approx(1.0)

    def test_a_load_that_contradicts_the_record_fails_loud(self, tmp_path):
        step = _make_step(tmp_path, bias=-0.8, weight_set=_weight_set())
        with pytest.raises(BaselineMismatchError) as err:
            step.run()
        message = str(err.value)
        assert f"{_RECORDED_ACCURACY:.6f}" in message
        assert "0.703125" in message
        assert "preprocessing" in message

    def test_the_contradicting_number_was_plausible_on_its_own(self, tmp_path):
        """Nothing about 0.70 looks broken -- that is why the record must be asserted."""
        step = _make_step(tmp_path, bias=-0.8, weight_set=_weight_set())
        with pytest.raises(BaselineMismatchError):
            step.run()
        assert step.trainer is not None
        measured = step.trainer.validate()
        assert 0.5 < measured < _RECORDED_ACCURACY

    def test_a_set_with_no_recorded_accuracy_disarms_the_check(self, tmp_path, capsys):
        step = _make_step(
            tmp_path, bias=-0.8, weight_set=_weight_set(expected_accuracy=None)
        )
        step.run()
        assert "no expected accuracy" in capsys.readouterr().out

    def test_an_adapted_workload_disarms_the_check(self, tmp_path, capsys):
        """The record describes a model this deploy does not run once the head is
        rebuilt, so it is not an expectation and must not be asserted."""
        step = _make_step(
            tmp_path,
            bias=-0.8,
            weight_set=_weight_set(num_classes=1000, adapts_num_classes=True),
        )
        step.run()
        assert "1000" in capsys.readouterr().out

    def test_the_disarmed_reason_is_printed_not_swallowed(self, tmp_path, capsys):
        step = _make_step(
            tmp_path, bias=-0.5, weight_set=_weight_set(expected_accuracy=None)
        )
        step.run()
        out = capsys.readouterr().out
        assert "Recorded baseline not asserted" in out
