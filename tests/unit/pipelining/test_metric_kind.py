"""Honest metric semantics: measured vs carried tagging and gate verdicts."""

from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    METRIC_MEASURED,
    PipelineStep,
)
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep


class _FakeTrainer:
    def test(self):
        return 0.9

    def validate(self):
        return 0.9


class _FakePipeline:
    def get_target_metric(self):
        return 0.5


def _bare(cls):
    return cls([], [], [], [], _FakePipeline())


class TestKindResolution:
    def test_base_step_defaults_to_measured_validate(self):
        step = _bare(PipelineStep)
        assert step.pipeline_metric_kind() == METRIC_MEASURED

    def test_step_with_trainer_is_measured(self):
        step = _bare(PipelineStep)
        step.trainer = _FakeTrainer()
        assert step.pipeline_metric_kind() == METRIC_MEASURED

    def test_trainer_step_without_trainer_is_carried(self):
        step = _bare(TrainerPipelineStep)
        assert step.trainer is None
        assert step.pipeline_metric_kind() == METRIC_CARRIED
        assert step.validate() == 0.5

    def test_tuner_step_without_tuner_is_carried(self):
        step = _bare(TunerPipelineStep)
        assert step.pipeline_metric_kind() == METRIC_CARRIED

    def test_tuner_step_with_tuner_trainer_is_measured(self):
        step = _bare(TunerPipelineStep)

        class _FakeTuner:
            trainer = _FakeTrainer()

            def validate(self):
                return 0.8

        step.tuner = _FakeTuner()
        assert step.pipeline_metric_kind() == METRIC_MEASURED

    def test_kind_resolution_never_measures(self):
        """pipeline_metric_kind is pure inspection: a trainer whose test()
        raises must not be invoked."""
        step = _bare(PipelineStep)

        class _Explosive:
            def test(self):
                raise RuntimeError("kind resolution must not measure")

        step.trainer = _Explosive()
        assert step.pipeline_metric_kind() == METRIC_MEASURED


class TestGateStepsDeclareCarried:
    def test_every_verdict_gate_declares_carried(self):
        from mimarsinan.pipelining.pipeline_steps.mapping.core_quantization_verification_step import (
            CoreQuantizationVerificationStep,
        )
        from mimarsinan.pipelining.pipeline_steps.verification.simulation_step import (
            SimulationStep,
        )

        for cls in (CoreQuantizationVerificationStep, SimulationStep):
            step = cls.__new__(cls)
            assert cls.validate_metric_kind(step) == METRIC_CARRIED, cls.__name__

    def test_config_steps_declare_carried(self):
        from mimarsinan.pipelining.pipeline_steps.config.model_building_step import (
            ModelBuildingStep,
        )
        from mimarsinan.pipelining.pipeline_steps.config.model_configuration_step import (
            ModelConfigurationStep,
        )

        for cls in (ModelBuildingStep, ModelConfigurationStep):
            step = cls.__new__(cls)
            assert cls.validate_metric_kind(step) == METRIC_CARRIED, cls.__name__


class TestVerdict:
    def test_default_verdict_is_none(self):
        step = _bare(PipelineStep)
        assert step.step_verdict() is None

    def test_recorded_verdict_surfaces(self):
        step = _bare(PipelineStep)
        step._verdict = {"status": "pass", "rule": "x", "detail": {}}
        assert step.step_verdict()["status"] == "pass"
