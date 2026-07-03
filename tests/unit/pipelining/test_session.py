"""PipelineSession is the single composition root for config -> running pipeline."""

import json
from pathlib import Path

import pytest

from conftest import MockDataProviderFactory

from mimarsinan.pipelining.session import (
    ParsedDeploymentConfig,
    PipelineSession,
    parse_deployment_config,
)


def _config(tmp_path, **overrides):
    cfg = {
        "experiment_name": "sess_test",
        "data_provider_name": "MNIST_DataProvider",
        "generated_files_path": str(tmp_path),
        "deployment_parameters": {
            "model_type": "mlp_mixer",
            "spiking_mode": "lif",
            "model_config_mode": "user",
            "hw_config_mode": "fixed",
            "model_config": {},
        },
        "platform_constraints": {},
    }
    cfg.update(overrides)
    return cfg


class TestParseDeploymentConfig:
    def test_working_directory_naming_and_config_persistence(self, tmp_path):
        cfg = _config(tmp_path)
        cfg["_internal_marker"] = "not persisted"
        parsed = parse_deployment_config(
            cfg, data_provider_factory=MockDataProviderFactory()
        )
        assert parsed.working_directory == f"{tmp_path}/sess_test_phased_deployment_run"
        saved = json.loads(
            (Path(parsed.working_directory) / "_RUN_CONFIG" / "config.json").read_text()
        )
        assert "_internal_marker" not in saved
        assert saved["experiment_name"] == "sess_test"

    def test_explicit_working_directory_wins(self, tmp_path):
        cfg = _config(tmp_path, _working_directory=str(tmp_path / "custom"))
        parsed = parse_deployment_config(
            cfg, data_provider_factory=MockDataProviderFactory()
        )
        assert parsed.working_directory == str(tmp_path / "custom")

    def test_hw_search_merges_search_space_without_mutating_input(self, tmp_path):
        cfg = _config(tmp_path)
        cfg["deployment_parameters"]["hw_config_mode"] = "search"
        cfg["platform_constraints"] = {
            "max_axons": 256,
            "search_space": {"core_counts": [60, 120]},
        }
        parsed = parse_deployment_config(
            cfg, data_provider_factory=MockDataProviderFactory()
        )
        assert "search_space" not in parsed.platform_constraints
        assert parsed.deployment_parameters["arch_search"]["core_counts"] == [60, 120]
        assert "search_space" in cfg["platform_constraints"], "input must not be mutated"


class TestPipelineSession:
    def _session(self, tmp_path, **overrides):
        return PipelineSession.from_config(
            _config(tmp_path, **overrides),
            data_provider_factory=MockDataProviderFactory(),
        )

    def test_builds_pipeline_with_resolved_plan(self, tmp_path):
        session = self._session(tmp_path)
        assert session.pipeline.plan.spiking_mode == "lif"
        assert session.pipeline.steps, "steps must be assembled"

    def test_target_metric_override_applies(self, tmp_path):
        session = self._session(tmp_path, target_metric_override=0.42)
        assert session.pipeline.get_target_metric() == 0.42

    def test_attach_gui_wires_reporter_and_hooks(self, tmp_path):
        session = self._session(tmp_path)

        class _Gui:
            reporter = object()
            def on_step_start(self, *a): ...
            def on_step_end(self, *a): ...

        gui = _Gui()
        session.attach_gui(gui)
        assert gui.reporter in session.pipeline.reporter._reporters
        assert session.reporter in session.pipeline.reporter._reporters
        assert gui.on_step_start in session.pipeline.pre_step_hooks
        assert gui.on_step_end in session.pipeline.post_step_hooks

    def test_run_dispatches_to_run_or_run_from(self, tmp_path, monkeypatch):
        session = self._session(tmp_path, stop_step="Pretraining")
        calls = []
        monkeypatch.setattr(
            session.pipeline, "run", lambda *, stop_step: calls.append(("run", stop_step))
        )
        session.run()
        assert calls == [("run", "Pretraining")]

        session2 = self._session(tmp_path, start_step="Pretraining")
        calls2 = []
        monkeypatch.setattr(
            session2.pipeline, "get_resolved_start_step", lambda name: name
        )
        monkeypatch.setattr(
            session2.pipeline,
            "run_from",
            lambda *, step_name, stop_step: calls2.append((step_name, stop_step)),
        )
        session2.run()
        assert calls2 == [("Pretraining", None)]

    def test_finish_degrades_on_reporter_failure(self, tmp_path):
        class _ExplodingReporter:
            def report(self, *a, **k): ...
            def console_log(self, *a, **k): ...
            def finish(self):
                raise RuntimeError("flush failed")

        session = PipelineSession.from_config(
            _config(tmp_path),
            data_provider_factory=MockDataProviderFactory(),
            reporter=_ExplodingReporter(),
        )
        session.finish()
