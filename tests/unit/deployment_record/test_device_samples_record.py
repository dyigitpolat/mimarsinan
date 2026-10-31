"""[U-HV] The sealed record carries each ODIN step's sample read as ``device_samples``; nothing moves without one.

A run whose plan runs an ODIN step (the device run, ``enable_odin_fpga_simulation``; the HACC export,
``enable_odin_hacc_export``) seals ``device_samples``: one row per step, read from that step's ``sample_read`` entry
(its backend, the declaring key ``odin_sample_split``, the split and the per-split sample indices). The record step
requires those entries only where the plan runs the step; the seal requires the fragment where it does, so a record
can never seal without saying which split its device samples came from; and a record without the fragment keeps
every byte, because ``to_dict`` emits the field only when it is set (``test_record_dict_unchanged.py`` pins the
digest of such a record).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from conftest import MockPipeline, default_config
from mimarsinan.deployment_record.schema import (
    DEPLOYMENT_RECORD_FILENAME,
    DeploymentRecord,
    load_deployment_record,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import seal_view_for
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_step import DeploymentRecordStep

from unit.deployment_record.record_fixtures import PlanView, make_full_record, make_provenance
from unit.deployment_record.test_schema_seal_matrix import make_builder
from unit.pipelining.pipeline_steps import test_deployment_record_step as step_fixtures

HACC_ENTRY = "odin_hacc_deployment_bundle"
FPGA_ENTRY = "odin_fpga_deployment_results"
HACC_READ = {"backend": "odin_hacc", "declared_by": "odin_sample_split", "split": "validation",
             "indices": [0, 1, 2]}
FPGA_READ = {"backend": "odin_fpga", "declared_by": "odin_sample_split", "split": "validation", "indices": [0]}


class TestTheSchema:
    def test_a_record_without_device_samples_emits_no_such_field(self):
        record = make_full_record()
        assert record.device_samples is None
        assert "device_samples" not in record.to_dict()

    def test_device_samples_round_trip_through_json(self):
        record = replace(make_full_record(), device_samples=(dict(FPGA_READ), dict(HACC_READ)))
        data = json.loads(json.dumps(record.to_dict()))
        assert data["device_samples"] == [FPGA_READ, HACC_READ]
        assert DeploymentRecord.from_dict(data) == record


class TestTheSeal:
    def test_the_seal_carries_the_attached_fragment_and_its_provenance(self):
        builder = make_builder()
        builder.attach("device_samples", (dict(HACC_READ),), make_provenance("declared", "Deployment Record"))
        record = builder.seal(PlanView(device_samples=True), [])
        assert record.device_samples == (HACC_READ,)
        assert record.provenance["device_samples"].kind == "declared"

    def test_a_plan_with_an_odin_step_refuses_to_seal_without_the_fragment(self):
        with pytest.raises(ValueError, match="device_samples"):
            make_builder().seal(PlanView(device_samples=True), [])

    def test_a_plan_without_one_seals_no_fragment(self):
        assert make_builder().seal(PlanView(), []).device_samples is None


def _record_step(tmp_path, *, hacc: bool, fpga: bool, with_reads: bool = True):
    config = default_config()
    config.update(enable_sanafe_simulation=False, enable_nevresim_simulation=False, spike_count_parity_samples=0,
                  enable_odin_hacc_export=hacc, enable_odin_fpga_simulation=fpga)
    pipeline = MockPipeline(config=config, working_directory=str(tmp_path / "run"))
    os.makedirs(pipeline.working_directory, exist_ok=True)
    pipeline.set_target_metric(step_fixtures.TARGET_METRIC)
    pipeline.seed("hard_core_mapping", step_fixtures._mapping())
    pipeline.seed("deployment_record_scm", step_fixtures._scm_entry())
    pipeline.seed("deployment_record_hcm", step_fixtures._hcm_entry())
    pipeline.seed("platform_constraints_resolved", dict(step_fixtures._PLATFORM))
    if hacc:
        pipeline.seed(HACC_ENTRY, {"name": "fixture", "samples": 3,
                                   **({"sample_read": dict(HACC_READ)} if with_reads else {})})
    if fpga:
        pipeline.seed(FPGA_ENTRY, {"backend": "odin_fpga", "samples": 1,
                                   **({"sample_read": dict(FPGA_READ)} if with_reads else {})})
    step = DeploymentRecordStep(pipeline)
    pipeline.prepare_step(step)
    return pipeline, step


class TestTheRecordStep:
    @pytest.mark.parametrize("hacc, fpga", [(False, False), (True, False), (False, True), (True, True)])
    def test_it_requires_each_odin_entry_only_where_the_plan_runs_the_step(self, tmp_path, hacc, fpga):
        _pipeline, step = _record_step(tmp_path, hacc=hacc, fpga=fpga)
        assert (HACC_ENTRY in step.requires, FPGA_ENTRY in step.requires) == (hacc, fpga)

    def test_the_sealed_record_carries_each_step_s_read(self, tmp_path):
        pipeline, step = _record_step(tmp_path, hacc=True, fpga=True)
        step.run()
        record = load_deployment_record(os.path.join(pipeline.working_directory, DEPLOYMENT_RECORD_FILENAME))
        assert record.device_samples == (FPGA_READ, HACC_READ)
        assert record.provenance["device_samples"].kind == "declared"

    def test_a_run_without_an_odin_step_seals_no_device_samples(self, tmp_path):
        pipeline, step = _record_step(tmp_path, hacc=False, fpga=False)
        step.run()
        with open(os.path.join(pipeline.working_directory, DEPLOYMENT_RECORD_FILENAME), encoding="utf-8") as handle:
            assert "device_samples" not in json.load(handle)

    @pytest.mark.parametrize("hacc, fpga, entry", [(True, False, HACC_ENTRY), (False, True, FPGA_ENTRY)])
    def test_a_cached_entry_without_its_sample_read_is_refused_by_name(self, tmp_path, hacc, fpga, entry):
        """An entry cached before the declared split (no ``sample_read``) cannot say which split its samples came
        from, so the record refuses it by name instead of sealing without the split or failing anonymously."""
        from mimarsinan.common.refusal import ToolchainRefusal

        pipeline, step = _record_step(tmp_path, hacc=hacc, fpga=fpga, with_reads=False)
        with pytest.raises(ToolchainRefusal, match=f"{entry}.*sample_read"):
            step.run()
        assert not os.path.exists(os.path.join(pipeline.working_directory, DEPLOYMENT_RECORD_FILENAME))

    @pytest.mark.parametrize("hacc, fpga", [(False, False), (True, False), (False, True)])
    def test_the_seal_view_says_where_an_odin_step_runs(self, tmp_path, hacc, fpga):
        pipeline, _step = _record_step(tmp_path, hacc=hacc, fpga=fpga)
        assert seal_view_for(pipeline, DeploymentPlan.of(pipeline)).device_samples == (hacc or fpga)
