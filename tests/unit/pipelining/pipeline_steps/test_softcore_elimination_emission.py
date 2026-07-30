"""[W6b] Every pruning-enabled run drops a softcore-elimination record.

Ratchet, mirroring the crossbar-utilization seam: the soft-core mapping step
must print the one-line summary, emit the ``softcore_elimination`` reporter
event, and serialize BOTH the JSON record and the drop-in markdown table into
the run directory. These assertions only ever tighten.

The seam is the soft-core one on purpose: it is the only point that holds the
mapping at its full pre-elimination geometry AND the per-arm kill sets. One
step later ``prune_ir_graph`` has compacted owned matrices and discarded the
weaker arms, so neither the denominator nor the C1 columns is recoverable.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import MockPipeline, default_config, make_tiny_ir_graph

from mimarsinan.mapping.softcore_elimination import (
    SOFTCORE_ELIMINATION_RECORD_FILENAME,
    SOFTCORE_ELIMINATION_TABLE_FILENAME,
)
from mimarsinan.pipelining.pipeline_steps.mapping import (
    soft_core_mapping_ir_pruning as mod,
)


class _RecordingReporter:
    def __init__(self):
        self.events = []

    def report(self, *args, **kwargs): ...

    def console_log(self, *args, **kwargs): ...

    def event(self, kind, payload):
        self.events.append((kind, payload))

    def finish(self): ...


class _StepStub:
    def __init__(self, pipeline):
        self.pipeline = pipeline


class _ModelStub:
    def get_perceptrons(self):
        return []


def _run(tmp_path, *, pruning=True, config_extra=None):
    config = default_config()
    config["pruning"] = pruning
    config.update(config_extra or {})
    pipeline = MockPipeline(config=config, working_directory=str(tmp_path))
    pipeline.reporter = _RecordingReporter()
    step = _StepStub(pipeline)
    mod.apply_ir_pruning_if_enabled(
        step, _ModelStub(), make_tiny_ir_graph(), "test_phase"
    )
    return pipeline


class TestSoftcoreEliminationEmission:
    def test_pruning_run_serializes_the_json_record(self, tmp_path):
        _run(tmp_path)
        path = Path(tmp_path) / SOFTCORE_ELIMINATION_RECORD_FILENAME
        assert path.exists(), "a pruning run must drop softcore_elimination.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "deployed_arm", "geometry", "arms", "groups", "layers", "total",
            "storage", "storage_total", "per_arm",
        ):
            assert key in record, f"record is missing {key!r}"
        assert record["total"]["cells"] > 0
        assert record["total"]["surviving"] <= record["total"]["cells"]
        assert 0.0 <= record["total"]["eliminated_fraction"] <= 1.0

    def test_pruning_run_serializes_the_markdown_table(self, tmp_path):
        _run(tmp_path)
        path = Path(tmp_path) / SOFTCORE_ELIMINATION_TABLE_FILENAME
        assert path.exists(), "a pruning run must drop softcore_elimination.md"
        text = path.read_text(encoding="utf-8")
        assert "| softcore group |" in text
        assert "Physical weight storage" in text

    def test_pruning_run_emits_the_reporter_event(self, tmp_path):
        pipeline = _run(tmp_path)
        kinds = [kind for kind, _ in pipeline.reporter.events]
        assert "softcore_elimination" in kinds
        payload = dict(pipeline.reporter.events)["softcore_elimination"]
        assert payload["deployed_arm"] == "cascade"
        assert payload["arms"] == ["masked", "closure", "cascade"]

    def test_pruning_run_prints_the_one_line_summary(self, tmp_path, capsys):
        _run(tmp_path)
        lines = [
            line for line in capsys.readouterr().out.splitlines()
            if "[SoftcoreElimination]" in line
        ]
        assert len(lines) == 1, lines
        assert "cells=" in lines[0] and "bank_cells=" in lines[0]

    def test_the_arm_columns_share_the_ledger_arms(self, tmp_path):
        _run(tmp_path)
        record = json.loads(
            (Path(tmp_path) / SOFTCORE_ELIMINATION_RECORD_FILENAME)
            .read_text(encoding="utf-8")
        )
        denominators = {
            view["total"]["cells"] for view in record["per_arm"].values()
        }
        assert len(denominators) == 1, "arm columns must be comparable"
        fractions = [
            record["per_arm"][arm]["total"]["eliminated_fraction"]
            for arm in record["arms"]
        ]
        assert fractions == sorted(fractions), (
            "a stronger arm can never reclaim less"
        )

    def test_pruning_disabled_emits_nothing(self, tmp_path):
        pipeline = _run(tmp_path, pruning=False)
        assert not (Path(tmp_path) / SOFTCORE_ELIMINATION_RECORD_FILENAME).exists()
        assert not (Path(tmp_path) / SOFTCORE_ELIMINATION_TABLE_FILENAME).exists()
        assert pipeline.reporter.events == []
