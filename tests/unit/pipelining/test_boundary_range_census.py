"""[U-BRANGE] T-5, the clamp census present: Boundary Quantization prints, per boundary, the clamped fraction, the
clamped values' share of the squared magnitude and the largest value over the scale; its report carries the same
fields for every boundary, and the sealed deployment record carries the report as ``boundary_ranges`` wherever the
run grids its boundaries (a record without it keeps every byte).

W38 group 6's record could place its loss only at the step, because the step printed no census: a boundary that
clamped the same two positions on every calibration image left no trace in any record.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest
import torch

from mimarsinan.deployment_record.schema import DEPLOYMENT_RECORD_FILENAME, DeploymentRecord, load_deployment_record
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import BoundaryQuantizationStep
from unit.pipelining.test_boundary_range_rule import HOT, _calibrated_form
from unit.pipelining.test_declared_grid_steps import _boundary_step, _flow
from spiking_points import declaring_fire_while_integrating

REPORT = "boundary_quantization_report"
CENSUS_FIELDS = ("clamped_fraction", "clamped_share", "largest_over_scale")


def _report(monkeypatch, **config):
    flow = _flow()
    pipeline = _boundary_step(monkeypatch, flow, **config)
    assert REPORT in pipeline.entries, "Boundary Quantization reports no range census"
    return flow, pipeline.entries[REPORT]


class TestTheStepLineAndReport:
    def test_the_report_carries_the_census_of_every_boundary(self, monkeypatch):
        flow, report = _report(monkeypatch)
        rows = report["boundaries"]
        assert len(rows) == len(flow.get_perceptrons()) == 2  # both perceptrons are segment entries here
        for row in rows:
            for field in CENSUS_FIELDS + ("scale", "low_span", "maximum", "squared_error", "name"):
                assert field in row, field
            assert 0.0 <= row["clamped_fraction"] <= 1.0 and 0.0 <= row["clamped_share"] <= 1.0
            assert row["largest_over_scale"] > 0.0
        assert report["activation_bits"] == 5 and "least squared error" in report["range_rule"]

    def test_the_step_prints_the_census_line_and_one_line_per_boundary(self, monkeypatch, capsys):
        _report(monkeypatch)
        out = capsys.readouterr().out
        summary = [line for line in out.splitlines() if "installed 5-bit value-grid quantizers" in line]
        assert len(summary) == 1
        assert "least squared error" in summary[0]
        for words in ("clamped fraction max", "clamped share of squared magnitude max", "largest value over scale max"):
            assert words in summary[0], words
        per_boundary = [line for line in out.splitlines() if line.startswith("[BoundaryQuantizationStep] boundary ")]
        assert len(per_boundary) == 2
        for line in per_boundary:
            for words in ("scale", "clamped", "share of squared magnitude", "largest value over scale"):
                assert words in line, (words, line)

    def test_a_declared_grid_is_marked_in_the_report(self, monkeypatch, tmp_path):
        from unit.mapping.declared_programs import write_state

        declaration = write_state(tmp_path / "s.json", {
            "_0": {"entry": {"scale": 6.0, "offset": 3.0, "rounding": "floor", "codes": [0, 15]}}})
        _flow_, report = _report(monkeypatch, quantizer_state=declaration)
        declared = {row["name"]: row["declared"] for row in report["boundaries"]}
        assert sorted(declared.values()) == [False, True]
        row = next(r for r in report["boundaries"] if r["declared"])
        assert row["deployed_scale"] == 6.0

    def test_the_miniature_s_systematic_value_is_kept_or_clamped_at_the_error_minimum(self):
        """W38 group 6's boundary 1 read its class-token values above the scale; after the rule, the census reads none
        above it, or a clamp the error minimum chose (the census's error at most both ends of the search)."""
        _form, quantizer, seams, census = _calibrated_form()
        assert census is not None, "the calibration returns no census"
        row = census[0]
        hot = max(float(s[:, 0, HOT].abs().max()) for s in seams)
        assert row.largest_over_scale == pytest.approx(hot / quantizer.grid.scale, rel=1e-6)
        assert row.clamped == 0 or (row.squared_error <= row.low_span_error and row.squared_error <= row.maximum_error)
        assert row.clamped_share < 0.5

    def test_the_census_is_the_direct_count_and_share_at_the_scale(self):
        """The census read directly from the seam's values: the entries above the scale, and their share of the
        squared magnitude, over both calibration batches."""
        _form, quantizer, seams, census = _calibrated_form()
        assert census is not None, "the calibration returns no census"
        row = census[0]
        values = torch.cat([s.reshape(-1).abs().double() for s in seams])
        above = values > row.scale
        assert row.entries == values.numel()
        assert row.clamped == int(above.sum())
        assert row.clamped_fraction == pytest.approx(float(above.double().mean()), rel=1e-9)
        assert row.clamped_share == pytest.approx(float((values[above] ** 2).sum() / (values ** 2).sum()), rel=1e-6)


class TestTheRecord:
    def test_the_record_schema_has_the_boundary_ranges_field(self):
        assert "boundary_ranges" in DeploymentRecord.__dataclass_fields__

    def test_a_record_without_boundary_ranges_emits_no_such_field(self):
        from unit.deployment_record.record_fixtures import make_full_record

        assert "boundary_ranges" in DeploymentRecord.__dataclass_fields__
        record = make_full_record()
        assert record.boundary_ranges is None and "boundary_ranges" not in record.to_dict()

    def test_boundary_ranges_round_trip_through_json(self, monkeypatch):
        from unit.deployment_record.record_fixtures import make_full_record

        assert "boundary_ranges" in DeploymentRecord.__dataclass_fields__
        _flow_, report = _report(monkeypatch)
        record = replace(make_full_record(), boundary_ranges=report)
        data = json.loads(json.dumps(record.to_dict()))
        assert data["boundary_ranges"]["boundaries"] == json.loads(json.dumps(report["boundaries"]))
        assert DeploymentRecord.from_dict(data) == record

    def test_the_seal_carries_the_attached_fragment(self, monkeypatch):
        from unit.deployment_record.record_fixtures import PlanView, make_provenance
        from unit.deployment_record.test_schema_seal_matrix import make_builder

        _flow_, report = _report(monkeypatch)
        builder = make_builder()
        builder.attach("boundary_ranges", dict(report), make_provenance("measured", "Boundary Quantization"))
        record = builder.seal(PlanView(), [])
        assert record.boundary_ranges["boundaries"] == report["boundaries"]


def _record_pipeline(tmp_path, *, gridded: bool, report=None):
    from conftest import MockPipeline, default_config
    from unit.pipelining.pipeline_steps import test_deployment_record_step as step_fixtures

    config = declaring_fire_while_integrating(default_config())  # [composed head, rule 12] U-LIF2 W16 (no census)
    config.update(enable_sanafe_simulation=False, enable_nevresim_simulation=False, spike_count_parity_samples=0)
    if gridded:
        config.update(core_semantics="mvm", activation_quantization=True)
    pipeline = MockPipeline(config=config, working_directory=str(tmp_path / "run"))
    os.makedirs(pipeline.working_directory, exist_ok=True)
    pipeline.set_target_metric(step_fixtures.TARGET_METRIC)
    pipeline.seed("hard_core_mapping", step_fixtures._mapping())
    pipeline.seed("deployment_record_scm", step_fixtures._scm_entry())
    pipeline.seed("deployment_record_hcm", step_fixtures._hcm_entry())
    pipeline.seed("platform_constraints_resolved", dict(step_fixtures._PLATFORM))
    if report is not None:
        pipeline.seed(REPORT, report)
    return pipeline


class TestTheRecordStep:
    @pytest.mark.parametrize("gridded", [False, True])
    def test_it_requires_the_report_exactly_where_boundary_quantization_applies(self, tmp_path, gridded):
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_step import DeploymentRecordStep

        pipeline = _record_pipeline(tmp_path, gridded=gridded)
        assert BoundaryQuantizationStep.applies_to(DeploymentPlan.of(pipeline)) is gridded
        assert (REPORT in DeploymentRecordStep(pipeline).requires) is gridded

    def test_the_sealed_record_carries_the_report(self, tmp_path, monkeypatch):
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_step import DeploymentRecordStep

        _flow_, report = _report(monkeypatch)
        pipeline = _record_pipeline(tmp_path, gridded=True, report=report)
        step = DeploymentRecordStep(pipeline)
        pipeline.prepare_step(step)
        step.run()
        record = load_deployment_record(os.path.join(pipeline.working_directory, DEPLOYMENT_RECORD_FILENAME))
        assert record.boundary_ranges is not None
        assert record.boundary_ranges["boundaries"] == json.loads(json.dumps(report["boundaries"]))
        assert record.provenance["boundary_ranges"].step == "Boundary Quantization"

    def test_a_run_without_gridded_boundaries_seals_no_boundary_ranges(self, tmp_path):
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_step import DeploymentRecordStep

        pipeline = _record_pipeline(tmp_path, gridded=False)
        step = DeploymentRecordStep(pipeline)
        pipeline.prepare_step(step)
        step.run()
        with open(os.path.join(pipeline.working_directory, DEPLOYMENT_RECORD_FILENAME), encoding="utf-8") as handle:
            assert "boundary_ranges" not in json.load(handle)


def test_the_census_fields_are_numbers_a_json_record_holds(monkeypatch):
    _flow_, report = _report(monkeypatch)
    assert json.loads(json.dumps(report)) == report
    assert all(isinstance(row[field], float) for row in report["boundaries"] for field in CENSUS_FIELDS)
    assert torch.isfinite(torch.tensor([row["largest_over_scale"] for row in report["boundaries"]])).all()
