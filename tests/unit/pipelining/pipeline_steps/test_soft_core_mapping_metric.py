from __future__ import annotations

import pytest

from conftest import MockPipeline
from mimarsinan.pipelining.pipeline_steps.soft_core_mapping_step import (
    SoftCoreMappingStep,
)


class _ExplodingTrainer:
    def validate(self):
        raise AssertionError("trainer.validate should not be called")

    def test(self):
        raise AssertionError("trainer.test should not be called")


class _Trainer:
    def __init__(self, validate_value=0.7, test_value=0.9):
        self.validate_value = validate_value
        self.test_value = test_value

    def validate(self):
        return self.validate_value

    def test(self):
        return self.test_value


def test_lif_soft_core_spiking_metric_is_pipeline_metric():
    """When the spiking sim ran, its result is the SCM step's metric.

    Reporting the FP-trainer's ``test()`` here previously caused a
    printed/reported mismatch (e.g. printed 0.876, reported 0.9687) and
    broke cross-step comparison against HCM / nevresim, which are also
    spiking-sim numbers.
    """
    step = SoftCoreMappingStep(MockPipeline())
    step.trainer = _ExplodingTrainer()
    step._soft_core_spiking_metric = 0.8125

    assert step.validate() == pytest.approx(0.8125)
    assert step.pipeline_metric() == pytest.approx(0.8125)


def test_ttfs_shifted_soft_core_mapping_metric_prefers_spiking_result():
    step = SoftCoreMappingStep(MockPipeline())
    step.trainer = _ExplodingTrainer()
    step._soft_core_spiking_metric = 0.8125

    assert step.validate() == pytest.approx(0.8125)
    assert step.pipeline_metric() == pytest.approx(0.8125)


def test_soft_core_metric_falls_back_when_sim_did_not_run():
    """If the spiking sim didn't set a metric (e.g. it crashed non-fatally),
    fall through to the FP trainer so the pipeline still gets a number."""
    step = SoftCoreMappingStep(MockPipeline())
    step.trainer = _Trainer(validate_value=0.75, test_value=0.875)
    step._soft_core_spiking_metric = None

    assert step.validate() == pytest.approx(0.75)
    assert step.pipeline_metric() == pytest.approx(0.875)


def test_soft_core_mapping_does_not_require_scaled_simulation_length():
    """``scaled_simulation_length`` is obsolete — every spiking path reads
    ``simulation_steps`` directly. Guard against it being reintroduced as
    a pipeline cache entry, which would re-create the SCM/HCM/Sim
    divergence it used to mask."""
    step = SoftCoreMappingStep(MockPipeline())

    assert "scaled_simulation_length" not in step.requires
