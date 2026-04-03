"""Tests for the commit guard in ActivationAdaptationTuner._after_run().

When the final accuracy after committing to ReLU falls below the target
adjuster's floor, the tuner should restore the pre-commit state and return
the pre-commit metric instead of the catastrophically low one.
"""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.tuners.activation_adaptation_tuner import (
    ActivationAdaptationTuner,
)
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.perceptron_mixer.perceptron import make_activation
from mimarsinan.models.layers import TransformedActivation


def _model_with_gelu():
    model = make_tiny_supermodel()
    for p in model.get_perceptrons():
        p.base_activation = make_activation("GELU")
        p.base_activation_name = "GELU"
        p.set_activation(TransformedActivation(p.base_activation, []))
    return model


class TestCommitGuard:
    @pytest.fixture
    def setup(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = _model_with_gelu()
        am = AdaptationManager()
        tuner = ActivationAdaptationTuner(pipeline, model, 0.9, 0.001, am)
        tuner._rollback_tolerance = 0.05
        return tuner, model

    def test_commit_proceeds_when_above_floor(self, setup):
        """Normal case: test accuracy is above floor, commit succeeds."""
        tuner, model = setup

        original_test = tuner.trainer.test

        def mock_test():
            return tuner.target_adjuster.floor + 0.05

        tuner.trainer.test = mock_test

        result = tuner._after_run()
        assert result >= tuner.target_adjuster.floor
        assert tuner._committed_metric >= tuner.target_adjuster.floor

    def test_commit_rolls_back_when_below_floor(self, setup):
        """If test accuracy after commit is below floor, pre-commit state is restored."""
        tuner, model = setup

        pre_commit_acc = 0.75
        catastrophic_post_acc = tuner.target_adjuster.floor * 0.5

        call_count = [0]

        def mock_validate_n_batches(n):
            return pre_commit_acc

        def mock_test():
            call_count[0] += 1
            return catastrophic_post_acc

        tuner.trainer.validate_n_batches = mock_validate_n_batches
        tuner.trainer.test = mock_test

        result = tuner._after_run()

        assert result == pre_commit_acc, (
            f"Should return pre-commit accuracy ({pre_commit_acc}), "
            f"not the catastrophic post-commit one ({catastrophic_post_acc})"
        )
        assert tuner._committed_metric == pre_commit_acc

    def test_committed_metric_set_on_success(self, setup):
        """On successful commit, _committed_metric equals the test accuracy."""
        tuner, model = setup
        good_acc = tuner.target_adjuster.floor + 0.1

        tuner.trainer.test = lambda: good_acc
        tuner._after_run()

        assert tuner._committed_metric == good_acc

    def test_validate_returns_committed_metric(self, setup):
        """After _after_run(), validate() returns the cached committed metric."""
        tuner, model = setup
        good_acc = tuner.target_adjuster.floor + 0.1

        tuner.trainer.test = lambda: good_acc
        tuner._after_run()

        assert tuner.validate() == good_acc
        assert tuner.validate() == good_acc

    def test_floor_derived_from_target(self, setup):
        """The floor should be target * floor_ratio, as defined by AdaptationTargetAdjuster."""
        tuner, _ = setup
        expected_floor = 0.9 * 0.90
        assert tuner.target_adjuster.floor == pytest.approx(expected_floor, rel=0.01)
