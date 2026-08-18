"""TS5: the additive-optional adaptation-ledger summary on the deployment record.

Totals only — the per-event detail stays in the step artifacts. The field
follows the ``invocations`` / ``partition`` additive precedent: it defaults to
None, sits LAST, carries no format-version bump, and a record JSON written
before it existed still loads.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from mimarsinan.deployment_record.build import DeploymentRecordBuilder
from mimarsinan.deployment_record.schema import (
    DEPLOYMENT_RECORD_FORMAT_VERSION,
    AdaptationSummaryRecord,
    DeploymentRecord,
    load_deployment_record,
    save_deployment_record,
)
from unit.deployment_record.record_fixtures import (
    PlanView,
    make_adaptation_summary,
    make_full_record,
    make_provenance,
)
from unit.deployment_record.test_schema_seal_matrix import make_builder

FIELD = "adaptation_ledger"


class TestFragmentShape:
    def test_carries_totals_stalls_and_completion_only(self):
        fields = [f.name for f in dataclasses.fields(AdaptationSummaryRecord)]
        assert fields == [
            "proposed", "accepted", "rejected", "retries", "recovery_steps",
            "probe_evals", "endpoint_steps", "total_steps", "stalls_by_path",
            "completed_via",
        ]

    def test_round_trips_through_json(self):
        fragment = make_adaptation_summary()
        wire = json.loads(json.dumps(fragment.to_dict()))
        assert AdaptationSummaryRecord.from_dict(wire) == fragment

    def test_unknown_field_rejected(self):
        wire = make_adaptation_summary().to_dict()
        wire["field_from_the_future"] = 1
        with pytest.raises(ValueError, match="unknown fields.*field_from_the_future"):
            AdaptationSummaryRecord.from_dict(wire)

    def test_it_answers_which_path_completed_each_adaptation_run(self):
        fragment = make_adaptation_summary()
        assert fragment.completed_via["LIF Adaptation"] == "reached_full_rate"
        assert fragment.stalls_by_path["epsilon_floor"] == 2


class TestAdditiveDiscipline:
    def test_the_field_is_last_and_defaults_to_none(self):
        fields = dataclasses.fields(DeploymentRecord)
        assert fields[-1].name == FIELD
        assert fields[-1].default is None

    def test_a_record_without_the_fragment_still_seals_and_round_trips(self):
        record = make_builder().seal(PlanView(), ["Hard Core Mapping"])
        assert getattr(record, FIELD) is None
        wire = record.to_dict()
        assert wire[FIELD] is None
        assert DeploymentRecord.from_dict(wire) == record

    def test_a_legacy_record_json_without_the_key_loads(self):
        wire = make_full_record().to_dict()
        del wire[FIELD]
        loaded = DeploymentRecord.from_dict(wire)
        assert getattr(loaded, FIELD) is None
        # Everything else survives the load unchanged.
        assert loaded.schedule == make_full_record().schedule

    def test_the_format_version_did_not_move(self):
        assert DEPLOYMENT_RECORD_FORMAT_VERSION == 1
        assert make_full_record().to_dict()["format_version"] == 1

    def test_a_record_carrying_the_fragment_round_trips_on_disk(self, tmp_path):
        record = make_full_record()
        assert getattr(record, FIELD) is not None
        path = save_deployment_record(record, str(tmp_path))
        assert load_deployment_record(path) == record
        assert json.loads(open(path).read())[FIELD]["completed_via"]


class TestSealMatrix:
    def test_the_fragment_is_attachable_with_provenance(self):
        builder = make_builder()
        builder.attach(FIELD, make_adaptation_summary(), make_provenance())
        record = builder.seal(PlanView(), [])
        assert getattr(record, FIELD) == make_adaptation_summary()
        assert FIELD in record.provenance

    def test_it_is_never_required_by_the_matrix(self):
        # Optional by construction: the seal must not demand it for any plan.
        record = make_builder().seal(
            PlanView(tuner_hosting_step_names=frozenset()), []
        )
        assert getattr(record, FIELD) is None

    def test_a_wrong_type_is_refused(self):
        with pytest.raises(TypeError, match=f"'{FIELD}' requires"):
            DeploymentRecordBuilder().attach(
                FIELD, make_full_record().accuracy, make_provenance()
            )


class TestRunDirRollup:
    """The assembler folds the per-step artifacts into the record's totals."""

    def _seal_artifacts(self, tmp_path, artifacts):
        for step, payload in artifacts.items():
            (tmp_path / f"{step}.adaptation_ledger.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

    def _payload(self, *, proposed, completed_via, stalls=None, endpoint_steps=0):
        return {
            "proposals": [], "verdicts": [], "refinements": [], "escalations": [],
            "recoveries": [], "endpoints": [],
            "totals": {
                "proposed": proposed, "accepted": proposed, "rejected": 0,
                "retries": 0, "recovery_steps": 10, "probe_evals": 2 * proposed,
                "endpoint_steps": endpoint_steps,
                "total_steps": 10 + endpoint_steps,
            },
            "stalls_by_path": stalls or {},
            "completed_via": completed_via,
        }

    def test_no_artifacts_yields_no_fragment(self, tmp_path):
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_adaptation import (  # noqa: E501
            adaptation_summary_from_run_dir,
        )

        assert adaptation_summary_from_run_dir(str(tmp_path)) is None

    def test_totals_and_stalls_sum_across_the_steps(self, tmp_path):
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_adaptation import (  # noqa: E501
            adaptation_summary_from_run_dir,
        )

        self._seal_artifacts(tmp_path, {
            "LIF Adaptation": self._payload(
                proposed=3, completed_via="reached_full_rate",
                stalls={"epsilon_floor": 1}, endpoint_steps=100,
            ),
            "Weight Quantization": self._payload(
                proposed=2, completed_via="epsilon_floor",
                stalls={"epsilon_floor": 1, "forced_full_rate": 1},
            ),
        })
        summary = adaptation_summary_from_run_dir(str(tmp_path))
        assert summary is not None
        assert summary.proposed == 5
        assert summary.accepted == 5
        assert summary.recovery_steps == 20
        assert summary.probe_evals == 10
        assert summary.endpoint_steps == 100
        assert summary.total_steps == 120
        assert summary.stalls_by_path == {"epsilon_floor": 2, "forced_full_rate": 1}
        assert summary.completed_via == {
            "LIF Adaptation": "reached_full_rate",
            "Weight Quantization": "epsilon_floor",
        }

    def test_a_step_that_never_completed_is_absent_from_completed_via(self, tmp_path):
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_adaptation import (  # noqa: E501
            adaptation_summary_from_run_dir,
        )

        self._seal_artifacts(tmp_path, {
            "Clamp Adaptation": self._payload(proposed=0, completed_via=None),
        })
        summary = adaptation_summary_from_run_dir(str(tmp_path))
        assert summary is not None
        assert summary.completed_via == {}
        assert summary.proposed == 0

    def test_the_assembler_attaches_the_fragment_it_read(self, tmp_path):
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_adaptation import (  # noqa: E501
            attach_adaptation_fragments,
        )

        self._seal_artifacts(tmp_path, {
            "LIF Adaptation": self._payload(
                proposed=2, completed_via="reached_full_rate"
            ),
        })
        builder = make_builder()
        walls = attach_adaptation_fragments(
            builder, str(tmp_path), tuner_steps_resolved=False,
            step_name="Deployment Record",
        )
        assert walls is False  # no ft_pass_walls.json in this run dir
        record = builder.seal(PlanView(), [])
        assert record.adaptation is None
        summary = getattr(record, FIELD)
        assert summary is not None
        assert summary.completed_via == {"LIF Adaptation": "reached_full_rate"}
        assert FIELD in record.provenance

    def test_a_run_with_no_adaptation_artifacts_attaches_neither(self, tmp_path):
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_adaptation import (  # noqa: E501
            attach_adaptation_fragments,
        )

        builder = make_builder()
        assert attach_adaptation_fragments(
            builder, str(tmp_path), tuner_steps_resolved=False,
            step_name="Deployment Record",
        ) is False
        record = builder.seal(PlanView(), [])
        assert record.adaptation is None
        assert getattr(record, FIELD) is None

    def test_the_rollup_is_json_round_trippable(self, tmp_path):
        from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_adaptation import (  # noqa: E501
            adaptation_summary_from_run_dir,
        )

        self._seal_artifacts(tmp_path, {
            "LIF Adaptation": self._payload(proposed=1, completed_via="round_budget"),
        })
        summary = adaptation_summary_from_run_dir(str(tmp_path))
        assert summary is not None
        wire = json.loads(json.dumps(summary.to_dict()))
        assert AdaptationSummaryRecord.from_dict(wire) == summary
