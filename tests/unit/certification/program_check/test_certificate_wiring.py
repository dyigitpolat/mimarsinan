"""[W15] The static check in the deployment certificate: Hard Core Mapping's gate reads the run's declaration, writes
the full report beside the run, and hands the deployment record two certificate records (Theorem 1, Proposition 2)
beside the sampled run's; a check that cannot run is recorded failed by name."""

from __future__ import annotations

import json
from types import SimpleNamespace

from mimarsinan.deployment_record.schema import CertificateRecord
from mimarsinan.pipelining.core.gates import static_program_gate as gate

from unit.certification.program_check import programs as P

CONFIG = {"spiking_family": "lif", "spiking_variant": "streamed", "simulation_steps": P.T}


def _pipeline(tmp_path):
    return SimpleNamespace(config=dict(CONFIG), working_directory=str(tmp_path))


def test_the_gate_reports_both_theorems_and_writes_the_report(tmp_path):
    program, ir = P.streamed_multipass()
    report, records = gate.run_static_program_check(_pipeline(tmp_path), ir, program, cores=P.V.cores(3),
                                                    compile_wall_s=0.25)
    assert report.semantics == "streamed_lif" and report.established("theorem1")
    assert [CertificateRecord.from_dict(r).name for r in records] == [
        "static_program_check_theorem1", "static_program_check_proposition2"]
    saved = json.loads((tmp_path / gate.REPORT_FILENAME).read_text())
    assert saved["theorem1"].startswith("established on the program") and saved["compile_wall_s"] == 0.25


def test_a_check_that_cannot_run_is_recorded_failed_by_name(tmp_path, monkeypatch):
    import mimarsinan.certification.program_check as pc

    def broken(*args, **kwargs):
        raise KeyError("no such field")

    monkeypatch.setattr(pc, "check_deployment", broken)
    program, ir = P.streamed_multipass()
    report, records = gate.run_static_program_check(_pipeline(tmp_path), ir, program, cores=P.V.cores(3))
    assert report is None and [r["passed"] for r in records] == [False, False]
    assert "KeyError" in json.loads(records[0]["detail"])["verdict"]


def test_the_record_fragment_carries_the_static_records_beside_the_sampled_ones(monkeypatch, tmp_path):
    from mimarsinan.pipelining.pipeline_steps.mapping import deployment_record_emission as emission

    monkeypatch.setattr(emission, "compute_partition_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(emission, "utilization_record_from_mapping", lambda *args, **kwargs: SimpleNamespace(to_dict=dict))
    program, ir = P.streamed_multipass()
    _, records = gate.run_static_program_check(_pipeline(tmp_path), ir, program, cores=P.V.cores(3))
    entries = {}
    step = SimpleNamespace(pipeline=SimpleNamespace(config={"simulation_steps": P.T, "cores": []}),
                           add_entry=lambda key, value, kind: entries.__setitem__(key, value))
    emission.emit_deployment_record_hcm(
        step, program, platform_constraints={"weight_bits": 8, "cores": P.V.cores(3)},
        scm_fragment={"reuse_plan": {"params_reloaded": 0}, "relay_cores_inserted": 0},
        programming=SimpleNamespace(neural_stages=0, programming_events=0, params_programmed=0, params_unique=0),
        crossbar_report=None, spike_gate_result=None, static_certificates=records, accuracy=0.5,
        observes_values=False, model=None, ir_graph=ir, input_shape=(P.V.IN,), num_classes=4,
        encoding_placement="subsume")
    certificates = [CertificateRecord.from_dict(c) for c in entries["deployment_record_hcm"]["certificates"]]
    assert [c.name for c in certificates] == ["static_program_check_theorem1", "static_program_check_proposition2"]
