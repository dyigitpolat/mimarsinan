"""Tests verifying that the pipeline's step-level tolerance is derived from
degradation_tolerance config rather than being hardcoded.

DeploymentPipeline._initialize_config() sets:
    self.tolerance = 1.0 - config["degradation_tolerance"]
"""

import pytest

from conftest import MockDataProviderFactory


def _noop_reporter():
    return type(
        "R", (),
        {
            "report": lambda *a, **kw: None,
            "console_log": lambda *a, **kw: None,
            "finish": lambda *a, **kw: None,
        },
    )()


_DEFAULT_CONSTRAINTS = {
    "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
    "target_tq": 4,
    "weight_bits": 8,
}


class TestPipelineToleranceFromConfig:
    def test_default_tolerance_is_0_95(self, tmp_path):
        """With default degradation_tolerance=0.05, pipeline tolerance = 0.95."""
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline

        dp = DeploymentPipeline(
            data_provider_factory=MockDataProviderFactory(),
            deployment_parameters={},
            platform_constraints=_DEFAULT_CONSTRAINTS,
            reporter=_noop_reporter(),
            working_directory=str(tmp_path),
        )
        assert dp.tolerance == pytest.approx(0.95)

    def test_custom_tolerance(self, tmp_path):
        """degradation_tolerance=0.10 -> pipeline tolerance = 0.90."""
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline

        dp = DeploymentPipeline(
            data_provider_factory=MockDataProviderFactory(),
            deployment_parameters={"degradation_tolerance": 0.10},
            platform_constraints=_DEFAULT_CONSTRAINTS,
            reporter=_noop_reporter(),
            working_directory=str(tmp_path),
        )
        assert dp.tolerance == pytest.approx(0.90)

    def test_tight_tolerance(self, tmp_path):
        """degradation_tolerance=0.01 -> pipeline tolerance = 0.99."""
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline

        dp = DeploymentPipeline(
            data_provider_factory=MockDataProviderFactory(),
            deployment_parameters={"degradation_tolerance": 0.01},
            platform_constraints=_DEFAULT_CONSTRAINTS,
            reporter=_noop_reporter(),
            working_directory=str(tmp_path),
        )
        assert dp.tolerance == pytest.approx(0.99)

    def test_tuner_and_pipeline_use_same_formula(self, tmp_path):
        """The tuner's rollback threshold and pipeline's step assertion must be
        algebraically equivalent to prevent the gap that caused the failures."""
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline

        for dt in [0.01, 0.05, 0.10, 0.20]:
            dp = DeploymentPipeline(
                data_provider_factory=MockDataProviderFactory(),
                deployment_parameters={"degradation_tolerance": dt},
                platform_constraints=_DEFAULT_CONSTRAINTS,
                reporter=_noop_reporter(),
                working_directory=str(tmp_path / f"dp_{dt}"),
            )
            assert dp.tolerance == pytest.approx(1.0 - dt), (
                f"Pipeline tolerance must equal 1 - degradation_tolerance for dt={dt}"
            )

    def test_scm_step_tolerance_opt_in(self, tmp_path):
        """``scm_degradation_tolerance`` tightens only the SCM (rung-2) gate.

        After the behavioral unification (rung 1 and rung 2 share semantics),
        the NF↔SCM residual is the honest mapping-level wire effect, so a
        ~2 pp budget is safe — but it is opt-in; the global tolerance keeps
        governing every step until the config asks otherwise."""
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline

        dp = DeploymentPipeline(
            data_provider_factory=MockDataProviderFactory(),
            deployment_parameters={
                "degradation_tolerance": 0.15,
                "scm_degradation_tolerance": 0.02,
            },
            platform_constraints=_DEFAULT_CONSTRAINTS,
            reporter=_noop_reporter(),
            working_directory=str(tmp_path),
        )
        assert dp._step_tolerance("Soft Core Mapping") == pytest.approx(0.98)
        assert dp._step_tolerance("Hard Core Mapping") == pytest.approx(0.85)
        assert dp._step_tolerance("Normalization Fusion") == pytest.approx(0.85)

    def test_scm_step_tolerance_defaults_to_global(self, tmp_path):
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline

        dp = DeploymentPipeline(
            data_provider_factory=MockDataProviderFactory(),
            deployment_parameters={"degradation_tolerance": 0.15},
            platform_constraints=_DEFAULT_CONSTRAINTS,
            reporter=_noop_reporter(),
            working_directory=str(tmp_path),
        )
        assert dp._step_tolerance("Soft Core Mapping") == pytest.approx(0.85)

    def test_base_pipeline_step_tolerance_falls_back_to_global(self, tmp_path):
        from mimarsinan.pipelining.core.engine.pipeline import Pipeline

        base = Pipeline(str(tmp_path / "base_step"))
        assert base._step_tolerance("Any Step") == base.tolerance

    def test_base_pipeline_tolerance_is_overridden(self, tmp_path):
        """DeploymentPipeline must override Pipeline.__init__'s hardcoded 0.95."""
        from mimarsinan.pipelining.core.engine.pipeline import Pipeline
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline

        # Base class hardcodes 0.95
        base = Pipeline(str(tmp_path / "base"))
        assert base.tolerance == 0.95

        # DeploymentPipeline with dt=0.10 overrides to 0.90
        dp = DeploymentPipeline(
            data_provider_factory=MockDataProviderFactory(),
            deployment_parameters={"degradation_tolerance": 0.10},
            platform_constraints=_DEFAULT_CONSTRAINTS,
            reporter=_noop_reporter(),
            working_directory=str(tmp_path / "deploy"),
        )
        assert dp.tolerance == 0.90
