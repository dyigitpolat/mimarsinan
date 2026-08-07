"""[IMC G-A] Every Hard Core Mapping run drops a crossbar-utilization record.

Ratchet: the mvm-semantics mapping run must print the summary, emit the
``crossbar_utilization`` reporter event, and serialize the flat ``to_dict()``
row as JSON into the run directory — mirroring the WeightProgrammingReport
seam. These assertions only ever tighten.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import MockPipeline, default_config, make_tiny_ir_graph

from mimarsinan.mapping.crossbar_utilization import (
    UTILIZATION_RECORD_FILENAME,
    CrossbarUtilizationReport,
    write_utilization_record,
)
from mimarsinan.pipelining.pipeline_steps.mapping.hard_core_mapping_step import (
    HardCoreMappingStep,
)

_PLATFORM = {
    "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
    "weight_bits": 8,
}


class _Cache(dict):
    def add(self, key, obj, strategy="basic"):
        self[key] = obj


class _RecordingReporter:
    def __init__(self):
        self.events = []

    def report(self, *args, **kwargs): ...

    def console_log(self, *args, **kwargs): ...

    def event(self, kind, payload):
        self.events.append((kind, payload))

    def finish(self): ...


def _mvm_pipeline(tmp_path):
    config = default_config()
    config["core_semantics"] = "mvm"
    pipeline = MockPipeline(config=config, working_directory=str(tmp_path))
    pipeline.cache = _Cache()
    pipeline.reporter = _RecordingReporter()
    return pipeline


def _run_mapping_step(pipeline, monkeypatch, platform=_PLATFORM):
    import mimarsinan.pipelining.pipeline_steps.mapping.hard_core_mapping_step as hcm

    monkeypatch.setattr(hcm, "run_spike_count_certificate_gate", lambda *a, **k: None)
    monkeypatch.setattr(hcm, "run_value_twin_certificate_gate", lambda *a, **k: None)
    monkeypatch.setattr(hcm, "run_value_mapping_metric", lambda *a, **k: 1.0)

    pipeline.seed("model", object())
    pipeline.seed("ir_graph", make_tiny_ir_graph())
    pipeline.seed("platform_constraints_resolved", platform)
    step = HardCoreMappingStep(pipeline)
    pipeline.prepare_step(step)
    step.run()
    return step


class TestUtilizationRecordWriter:
    def _report(self):
        return CrossbarUtilizationReport.from_hybrid_mapping(
            _FakeMapping(), weight_bits=4
        )

    def test_record_lands_as_json_with_the_flat_row_keys(self, tmp_path):
        report = self._report()
        path = write_utilization_record(report, str(tmp_path))
        assert Path(path) == tmp_path / UTILIZATION_RECORD_FILENAME
        record = json.loads(Path(path).read_text())
        assert record == report.to_dict()

    def test_writer_fails_loud_on_a_missing_run_directory(self, tmp_path):
        with pytest.raises(OSError):
            write_utilization_record(self._report(), str(tmp_path / "absent"))


class _FakeHardCore:
    def __init__(self):
        self.axons_per_core = 16
        self.neurons_per_core = 16
        self.available_axons = 4
        self.available_neurons = 8
        self.unusable_space = 3


class _FakeSegment:
    def __init__(self):
        self.cores = [_FakeHardCore()]


class _FakeStage:
    def __init__(self):
        self.kind = "neural"
        self.hard_core_mapping = _FakeSegment()


class _FakeMapping:
    def __init__(self):
        self.stages = [_FakeStage()]


class TestMvmRunEmitsUtilization:
    def test_mvm_mapping_run_drops_the_utilization_record(
        self, tmp_path, monkeypatch
    ):
        pipeline = _mvm_pipeline(tmp_path)
        _run_mapping_step(pipeline, monkeypatch)

        record_path = tmp_path / UTILIZATION_RECORD_FILENAME
        assert record_path.exists(), "every mvm run must drop a utilization record"
        record = json.loads(record_path.read_text())

        mapping = pipeline.cache["hybrid_mapping"]
        expected = CrossbarUtilizationReport.from_hybrid_mapping(
            mapping, weight_bits=_PLATFORM["weight_bits"]
        ).to_dict()
        assert set(record.keys()) == set(expected.keys())
        assert record == expected
        assert record["cores_allocated"] >= 1, "a packed program allocates cores"

    def test_reporter_event_carries_the_flat_record(self, tmp_path, monkeypatch):
        pipeline = _mvm_pipeline(tmp_path)
        _run_mapping_step(pipeline, monkeypatch)

        events = dict(pipeline.reporter.events)
        assert "crossbar_utilization" in events
        record = json.loads((tmp_path / UTILIZATION_RECORD_FILENAME).read_text())
        assert events["crossbar_utilization"] == record

    def test_weight_programming_event_still_emitted(self, tmp_path, monkeypatch):
        # The new emission mirrors — never displaces — the WPR seam.
        pipeline = _mvm_pipeline(tmp_path)
        _run_mapping_step(pipeline, monkeypatch)
        assert "weight_programming" in dict(pipeline.reporter.events)

    def test_undeclared_platform_width_leaves_programming_bits_null(
        self, tmp_path, monkeypatch
    ):
        platform = {"cores": _PLATFORM["cores"]}
        pipeline = _mvm_pipeline(tmp_path)
        _run_mapping_step(pipeline, monkeypatch, platform=platform)
        record = json.loads((tmp_path / UTILIZATION_RECORD_FILENAME).read_text())
        assert record["weight_bits"] is None
        assert record["programming_bits"] is None

    def test_summary_line_is_printed(self, tmp_path, monkeypatch, capsys):
        pipeline = _mvm_pipeline(tmp_path)
        _run_mapping_step(pipeline, monkeypatch)
        assert "[Crossbar]" in capsys.readouterr().out
