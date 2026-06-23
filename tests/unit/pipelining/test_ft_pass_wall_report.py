"""A5b: TunerPipelineStep persists the per-FT-pass max wall (AC5) as a reported
metric, so it lands in the already-persisted steps.json for the AC5 verdict."""

from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep


class _RecordingReporter:
    def __init__(self):
        self.reports = []

    def report(self, name, value):
        self.reports.append((name, value))


class _Pipeline:
    def __init__(self):
        self.reporter = _RecordingReporter()


def _step():
    return TunerPipelineStep([], [], [], [], _Pipeline())


def test_reports_max_ft_pass_wall_when_tuner_exposes_it():
    step = _step()
    step.tuner = type("T", (), {"max_ft_pass_wall_s": 42.5})()
    step._report_ft_pass_wall()
    assert ("max_ft_pass_wall_s", 42.5) in step.pipeline.reporter.reports


def test_no_report_when_tuner_lacks_the_metric():
    step = _step()
    step.tuner = object()  # a tuner family without the AC5 instrumentation
    step._report_ft_pass_wall()
    assert step.pipeline.reporter.reports == []


def test_no_report_when_no_tuner():
    step = _step()
    step.tuner = None
    step._report_ft_pass_wall()
    assert step.pipeline.reporter.reports == []
