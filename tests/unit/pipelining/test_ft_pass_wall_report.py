"""A5b: TunerPipelineStep persists the per-FT-pass max wall (AC5) as a reported
metric, so it lands in the already-persisted steps.json for the AC5 verdict.

W3-S1: the same seam also persists the run-directory adaptation artifacts —
the ``ft_pass_walls.json`` accumulator and one ``retention_ledger.json`` entry
per tuner-hosting step — at commit time (``_commit_tuner_entries``), covering
BOTH the ``run_tuner`` steps and the direct-construction steps."""

import json

from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.orchestration import endpoint_steps


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


# --------------------------------------------------------------------------- #
# W3-S1: commit-time persistence of ft_pass_walls.json + retention_ledger.json.
# --------------------------------------------------------------------------- #


class _RunPipeline(_Pipeline):
    """Pipeline duck with the surfaces the instrumentation seam reads."""

    def __init__(self, working_directory=None):
        super().__init__()
        self.cache = {}
        self.config = {}
        if working_directory is not None:
            self.working_directory = working_directory
        self.committed = {}

    def update_entry(self, step, key, value, strategy):
        self.committed[key] = value

    def add_entry(self, step, key, value, strategy="basic"):
        self.cache[f"{step.name}.{key}"] = value


class _WallsTuner:
    """A tuner double with the AC5 bundle and a zero-draw exit estimate."""

    def __init__(self, passes, exit_metric=0.9):
        self._passes = passes
        self._exit = exit_metric

    def ft_pass_wall_metrics(self):
        walls = [w for _, w in self._passes]
        return {
            "max_ft_pass_wall_s": max(walls, default=0.0),
            "passes": [{"label": lbl, "wall_s": w} for lbl, w in self._passes],
        }

    def exit_metric_estimate(self):
        return self._exit


class _NoWallsTuner:
    """A tuner family without the AC5 instrumentation, but with an exit read."""

    def __init__(self, exit_metric):
        self._exit = exit_metric

    def exit_metric_estimate(self):
        return self._exit


def _committing_step(pipeline, name="Tuner Step"):
    step = TunerPipelineStep([], [], ["model", "adaptation_manager"], [], pipeline)
    step.name = name
    step._updated_entries = set()
    return step


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class TestCommitPersistsAdaptationArtifacts:
    def test_walls_accumulate_with_step_qualified_labels(self, tmp_path):
        pipeline = _RunPipeline(str(tmp_path))
        step_a = _committing_step(pipeline, "LIF Adaptation")
        step_a.tuner = _WallsTuner([("recover", 3.0)])
        step_a._commit_tuner_entries(object(), object())
        step_b = _committing_step(pipeline, "Weight Quantization")
        step_b.tuner = _WallsTuner([("recover", 8.5)])
        step_b._commit_tuner_entries(object(), object())

        data = _read_json(tmp_path / "ft_pass_walls.json")
        assert [p["label"] for p in data["passes"]] == [
            "LIF Adaptation/recover",
            "Weight Quantization/recover",
        ]
        assert data["max_ft_pass_wall_s"] == 8.5

    def test_retention_entry_uses_step_entry_and_exit_estimate(self, tmp_path):
        pipeline = _RunPipeline(str(tmp_path))
        step = _committing_step(pipeline, "Weight Quantization")
        step.pipeline_previous_metric = 0.95
        step._endpoint_steps_consumed_before = 0
        step.tuner = _WallsTuner([("recover", 1.0)], exit_metric=0.93)
        step._commit_tuner_entries(object(), object())

        entries = _read_json(tmp_path / "retention_ledger.json")["entries"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["step"] == "Weight Quantization"
        assert entry["entry_metric"] == 0.95
        assert entry["exit_metric"] == 0.93
        assert round(entry["retention_delta"], 6) == -0.02
        assert entry["endpoint_steps_consumed_before"] == 0
        assert entry["endpoint_steps_consumed_after"] == 0
        assert entry["armed_recovery"] is False

    def test_run_snapshots_consumed_before_and_derives_armed(self, tmp_path):
        pipeline = _RunPipeline(str(tmp_path))
        endpoint_steps.consume(pipeline, 500)  # a previous step's consumption

        class _ConsumingStep(TunerPipelineStep):
            def process(self):
                # The tuner's armed endpoint stage consumes ledger steps.
                endpoint_steps.consume(self.pipeline, 1200)
                self.tuner = _NoWallsTuner(exit_metric=0.91)
                self._commit_tuner_entries(object(), object())

        # [TS5] the base now promises its adaptation-ledger artifact, so a step
        # double driven through run() must carry that promise too.
        step = _ConsumingStep(
            [], TunerPipelineStep.PROMISES, ["model", "adaptation_manager"], [],
            pipeline,
        )
        step.name = "LIF Adaptation"
        step.pipeline_previous_metric = 0.9
        step.run()

        entry = _read_json(tmp_path / "retention_ledger.json")["entries"][0]
        assert entry["endpoint_steps_consumed_before"] == 500
        assert entry["endpoint_steps_consumed_after"] == 1700
        assert entry["armed_recovery"] is True
        assert "LIF Adaptation.adaptation_ledger" in pipeline.cache

    def test_ledger_entries_append_across_steps_in_order(self, tmp_path):
        pipeline = _RunPipeline(str(tmp_path))
        for name in ("A", "B"):
            step = _committing_step(pipeline, name)
            step.tuner = _NoWallsTuner(exit_metric=0.9)
            step._commit_tuner_entries(object(), object())
        entries = _read_json(tmp_path / "retention_ledger.json")["entries"]
        assert [e["step"] for e in entries] == ["A", "B"]

    def test_tuner_without_walls_still_writes_the_ledger(self, tmp_path):
        pipeline = _RunPipeline(str(tmp_path))
        step = _committing_step(pipeline)
        step.tuner = _NoWallsTuner(exit_metric=0.9)
        step._commit_tuner_entries(object(), object())
        assert not (tmp_path / "ft_pass_walls.json").exists()
        assert (tmp_path / "retention_ledger.json").exists()

    def test_no_tuner_writes_nothing(self, tmp_path):
        pipeline = _RunPipeline(str(tmp_path))
        step = _committing_step(pipeline)
        step.tuner = None
        step._commit_tuner_entries(object(), object())
        assert list(tmp_path.iterdir()) == []

    def test_no_working_directory_writes_nothing(self):
        # A bare pipeline double (unit-test contexts) has nowhere to write.
        pipeline = _RunPipeline(working_directory=None)
        step = _committing_step(pipeline)
        step.tuner = _WallsTuner([("recover", 1.0)])
        step._commit_tuner_entries(object(), object())  # must not raise

    def test_bare_double_without_exit_estimate_skips_the_ledger(self, tmp_path):
        # A non-TunerBase double (no exit_metric_estimate) has no exit read.
        pipeline = _RunPipeline(str(tmp_path))
        step = _committing_step(pipeline)
        step.tuner = object()
        step._commit_tuner_entries(object(), object())
        assert not (tmp_path / "retention_ledger.json").exists()

    def test_never_measured_tuner_skips_the_ledger(self, tmp_path):
        # A TunerBase family whose run never measured anything (estimate None)
        # writes no entry rather than a fabricated exit metric.
        pipeline = _RunPipeline(str(tmp_path))
        step = _committing_step(pipeline)
        step.tuner = _NoWallsTuner(exit_metric=None)
        step._commit_tuner_entries(object(), object())
        assert not (tmp_path / "retention_ledger.json").exists()
