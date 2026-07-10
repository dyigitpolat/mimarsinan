"""Reported metrics are fp32: the metric_grade_eval seam and its consumers."""

import pytest
import torch
import torch.nn as nn

from mimarsinan.model_training import basic_trainer_eval
from mimarsinan.model_training.basic_trainer_eval import (
    metric_grade_eval,
    test as eval_test,
    validate,
    validate_correctness_on_indices,
    validate_n_batches,
)


class _AutocastRecordingModel(nn.Module):
    """Records the ambient autocast state and output dtype of every forward."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.autocast_states: list[bool] = []
        self.output_dtypes: list[torch.dtype] = []

    def forward(self, x):
        self.autocast_states.append(torch.is_autocast_enabled("cpu"))
        out = self.linear(x)
        self.output_dtypes.append(out.dtype)
        return out


class _FakeTrainer:
    def __init__(self, model, batches):
        self.model = model
        self.device = "cpu"
        self.test_loader = batches
        self._gpu_val_cache = batches
        self._gpu_val_cursor = 0
        self.reports = []

    def _report(self, name, value):
        self.reports.append((name, value))

    def _validation_metric_name(self, name):
        return name

    def iter_validation_batches(self, n_batches):
        for i in range(int(n_batches)):
            yield self._gpu_val_cache[i % len(self._gpu_val_cache)]

    def next_validation_batch(self):
        return self._gpu_val_cache[0]


def _trainer():
    x = torch.randn(6, 4)
    y = torch.randint(0, 2, (6,))
    return _FakeTrainer(_AutocastRecordingModel(), [(x, y)])


class TestMetricGradeEval:
    def test_disables_ambient_cpu_autocast(self):
        with torch.autocast("cpu"):
            assert torch.is_autocast_enabled("cpu")
            with metric_grade_eval("cpu"):
                assert not torch.is_autocast_enabled("cpu")
            assert torch.is_autocast_enabled("cpu")

    def test_noop_outside_autocast_capable_devices(self):
        with metric_grade_eval("meta"):
            pass


class TestEvalPathsAreMetricGrade:
    """Every reported-metric path measures in fp32 even inside an ambient
    autocast region (the training-loop convention)."""

    @pytest.mark.parametrize(
        "invoke",
        [
            pytest.param(lambda t: eval_test(t), id="test"),
            pytest.param(lambda t: validate(t), id="validate"),
            pytest.param(lambda t: validate_n_batches(t, 2), id="validate_n_batches"),
            pytest.param(
                lambda t: validate_correctness_on_indices(t, [0]),
                id="validate_correctness_on_indices",
            ),
        ],
    )
    def test_forward_runs_fp32_under_ambient_autocast(self, invoke):
        trainer = _trainer()
        with torch.autocast("cpu"):
            invoke(trainer)
        assert trainer.model.autocast_states, "forward never ran"
        assert not any(trainer.model.autocast_states)
        assert all(d == torch.float32 for d in trainer.model.output_dtypes)

    def test_fp16_eval_autocast_seam_is_gone(self):
        assert not hasattr(basic_trainer_eval, "_eval_autocast")


class TestSharedSeamReuse:
    def test_mbh_ledger_reuses_metric_grade_eval(self):
        from mimarsinan.tuning.orchestration import mbh_ledger

        assert mbh_ledger.metric_grade_eval is metric_grade_eval
        assert not hasattr(mbh_ledger, "_autocast_disabled")
