"""Headless wiring: PipelineSession surfaces config + post-pretrain advisories."""

from conftest import MockDataProviderFactory

from mimarsinan.advisories.surfacing import ADVISORY_EVENT_KIND
from mimarsinan.pipelining.session import PipelineSession


class RecordingReporter:
    prefix = ""

    def __init__(self):
        self.events = []

    def report(self, metric_name, metric_value, step=None):
        pass

    def console_log(self, metric_name, metric_value):
        pass

    def event(self, kind, payload):
        self.events.append((kind, payload))

    def finish(self):
        pass


def _config(tmp_path, deployment_overrides=None, **top):
    cfg = {
        "experiment_name": "advisory_sess_test",
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
    cfg["deployment_parameters"].update(deployment_overrides or {})
    cfg.update(top)
    return cfg


def _session(tmp_path, deployment_overrides=None, **top):
    reporter = RecordingReporter()
    session = PipelineSession.from_config(
        _config(tmp_path, deployment_overrides, **top),
        data_provider_factory=MockDataProviderFactory(),
        reporter=reporter,
    )
    return session, reporter


def _advisory_ids(reporter):
    return [
        payload["id"]
        for kind, payload in reporter.events
        if kind == ADVISORY_EVENT_KIND
    ]


class TestConfigAdvisorySurfacing:
    def test_cascaded_config_prints_and_emits(self, tmp_path, capsys):
        session, reporter = _session(
            tmp_path, {"spiking_mode": "ttfs_cycle_based"}
        )
        session.surface_config_advisories()
        out = capsys.readouterr().out
        assert "[ADVISORY][UNSUPPORTED] ADV-CASC-UNSUPPORTED" in out
        assert "not fully supported" in out
        assert "ADV-CASC-UNSUPPORTED" in _advisory_ids(reporter)
        payload = dict(reporter.events[-1][1])
        assert payload["context"] == "config"

    def test_clean_lif_config_stays_silent(self, tmp_path, capsys):
        session, reporter = _session(tmp_path)
        session.surface_config_advisories()
        assert "[ADVISORY]" not in capsys.readouterr().out
        assert _advisory_ids(reporter) == []


class TestPostPretrainAdvisoryHook:
    def test_hook_is_registered(self, tmp_path):
        session, _ = _session(tmp_path)
        assert (
            session._post_pretraining_advisory_hook
            in session.pipeline.post_step_hooks
        )

    def test_fires_below_the_declared_acceptance_target(self, tmp_path, capsys):
        session, reporter = _session(tmp_path, target_metric_override=0.9)
        session.pipeline.set_target_metric(0.5)
        session._post_pretraining_advisory_hook("Pretraining", None)
        assert "ADV-ENVELOPE-GATE" in _advisory_ids(reporter)
        assert "[ADVISORY][RISK] ADV-ENVELOPE-GATE" in capsys.readouterr().out

    def test_silent_at_or_above_the_target(self, tmp_path):
        session, reporter = _session(tmp_path, target_metric_override=0.9)
        session.pipeline.set_target_metric(0.95)
        session._post_pretraining_advisory_hook("Pretraining", None)
        assert _advisory_ids(reporter) == []

    def test_ignores_other_steps(self, tmp_path):
        session, reporter = _session(tmp_path, target_metric_override=0.9)
        session.pipeline.set_target_metric(0.1)
        session._post_pretraining_advisory_hook("Torch Mapping", None)
        assert _advisory_ids(reporter) == []

    def test_silent_without_a_declared_target(self, tmp_path):
        session, reporter = _session(tmp_path)
        session.pipeline.set_target_metric(0.1)
        session._post_pretraining_advisory_hook("Pretraining", None)
        assert _advisory_ids(reporter) == []
