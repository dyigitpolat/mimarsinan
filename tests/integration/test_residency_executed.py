"""Acceptance 14 (runs once S0's resolver carries ``residency_plan``): the variants t0_44r and t0_53r (t0_44 and
t0_53 with ``residency_plan: {step: floor}`` and a core budget forcing Q >= 2) run through ``PipelineSession``; every
pipeline gate and certificate gives the verdict the U1 run of the same variant gives, and the sealed record's
``identity.platform`` carries the key. Walls and peak RSS are printed.

The variants are built from the tier-0 templates here (the RA tree records the same declarations as
``RA/10_wave1/g5_residency_planner/variants/``). Both runs of ``t0_53r`` declare nevresim off, as they did when
nevresim read its T-cycle input train at the global cycle and refused the U1 twin (its second pass reads a carried
slice at latency 1). Since W33 every backend reads a stage input in core-local time and nevresim admits the twin
(``test_t0_53rs_u1_twin_and_its_residency_steps_are_admitted_on_nevresim``); turning nevresim back on in
``t0_53r``'s gate set is the integration run's change, since these pipeline runs read the data set's test split.
Both runs of ``t0_53r`` also bound their processes to the capped scopes the runs are admitted under (TasksMax 150):
``num_workers: 0`` (no persistent DataLoader workers; a data-loading setting, the deployed function unchanged) and one
Lava wave worker (``MIMARSINAN_LOIHI_WAVE_WORKERS``; the default, ``os.cpu_count()``, forks a Lava runtime per core of
a wave).

The verdicts are compared as each run prints them (``gate_verdicts``): every printed gate line, its verdict PASS, FAIL
or SKIP included, must match its U1 twin's, and each variant's own gates must have run (``RAN``). The records are
compared too (``certificate_verdicts``). [fix round M1] A value-domain record without a readout law holds no sampled
certificate row, only the static program check's two records (W15, ``STATIC_RECORDS``), which read outside scope
there: the value domain without a readout law runs the fp64 float path, not the integer law. So on t0_44r the records
compare those two verdicts and the printed lines carry the sampled ones. A static record passes only where its theorem
is established (W15's rule), so the runs are certified when every sampled certificate passes and every static record
is established, outside scope with its stated reason, or conditional with the conditions that stay with the sampled
run (``uncertified``), never failed.
"""

from __future__ import annotations

import copy
import json
import re
import resource
import time
from pathlib import Path

import pytest

from integration.parity_harness import ensure_nevresim_path
from mimarsinan.certification.program_check.report import ESTABLISHED_ON_PROGRAM
from mimarsinan.deployment_record.schema.record import DEPLOYMENT_RECORD_FILENAME, load_deployment_record
from mimarsinan.pipelining.session import PipelineSession

TEMPLATES = Path(__file__).resolve().parents[2] / "templates" / "tier_0"
#: variant -> (template, core-count scale, deployment overrides): the budgets the G5b unit tests certify (Q >= 2, every
#: unit within C); t0_53r runs without nevresim (module docstring). [fix_c] t0_44r runs at x3.4
#: (40 + 40 cores), below its 42 + 42 footprint: at x4 the declared pool holds the unscheduled pack, so the fit rule
#: makes the floor that pack (one pass per segment, no composition plan); its loaders run in process, as CL3's
#: harness runs them (a worker's queue timed out under a loaded host)
VARIANTS = {"t0_44r": ("t0_44", 3.4, {"num_workers": 0}),
            "t0_53r": ("t0_53", 1.34, {"enable_nevresim_simulation": False, "num_workers": 0})}
#: Per variant, gates that must have run and passed (value parity and the value twin on t0_44r; the chip
#: certificates on t0_53r), so the comparison is never one of two empty sets.
RAN = {"t0_44r": {("ValueParityGate", "", "PASS"), ("ValueTwinCertificate", "", "PASS")},
       "t0_53r": {("SpikeCountCertificate", "[loihi/counts-export]", "PASS"),
                  ("SpikeCountCertificate", "[sanafe/exact]", "PASS"), ("Loihi spike parity", "", "PASS")}}

_TAG = re.compile(r"^\[([^\]]+)\]")
_VERDICT = re.compile(r"\b(PASS|FAIL|SKIP)")
_KIND = re.compile(r"\[\w+/[\w-]+\]|streaming-twin|cell \S+|\([^)]*\bvs\b[^)]*\)")


def gate_verdicts(text: str) -> list:
    """Every gate and certificate line a run prints, as (gate, what it compares, verdict): the tag (or ``Loihi spike
    parity``), the backend bracket, streaming twin, cell or "a vs b" pair, and PASS, FAIL or SKIP. The numbers two
    programs of one function legitimately differ in (passes, windows compared) are not part of a verdict."""
    out = []
    for line in text.splitlines():
        verdict, tag = _VERDICT.search(line), _TAG.match(line)
        if verdict is None or (tag is None and not line.startswith("Loihi spike parity")):
            continue
        kind = _KIND.search(line)
        out.append((tag.group(1) if tag else "Loihi spike parity", kind.group(0) if kind else "", verdict.group(1)))
    return out


#: [W15] The static program check's records: backend ``static``, one per theorem, in every deployment record.
STATIC_BACKEND = "static"
STATIC_RECORDS = {"static_program_check_theorem1", "static_program_check_proposition2"}
#: A static verdict's class by the prefix the check writes (``program_check.report``, ``check.py``); every class but
#: ``failed`` is one a certified run may record, each but ``established`` with what it names.
_STATIC_PREFIXES = ((ESTABLISHED_ON_PROGRAM, "established"), ("outside_scope: ", "outside_scope"),
                    ("conditional: ", "conditional"), ("not claimed: ", "not_claimed"), ("failed", "failed"))
NAMED_STATIC = ("outside_scope", "conditional", "not_claimed")
_OPEN = " stay with the sampled run"


def static_verdict(certificate) -> tuple:
    """A static record's verdict (its ``detail``) as ``(class, what it names)``: ``("established", "")``; an
    outside-scope or not-claimed verdict with its stated reason; a conditional one with the conditions that stay with
    the sampled run; ``("failed", what failed)``; ``("unread", the verdict)`` for a text no prefix reads."""
    verdict = str(json.loads(certificate.detail or "{}").get("verdict", ""))
    for prefix, kind in _STATIC_PREFIXES:
        if not verdict.startswith(prefix):
            continue
        rest = verdict[len(prefix):]
        if kind == "established":
            return kind, ""
        if kind == "conditional":
            if _OPEN not in rest:
                break
            rest = rest.split(_OPEN, 1)[0]
        return kind, rest.strip(" :")
    return "unread", verdict


def certificate_verdicts(record) -> dict:
    """Every certificate of the record keyed ``(name, backend)``: a sampled certificate by whether it passed, a static
    record by ``static_verdict``."""
    return {(c.name, c.backend): static_verdict(c) if c.backend == STATIC_BACKEND else c.passed
            for c in record.accuracy.certificates}


def uncertified(record) -> list:
    """What keeps the record's certificates from certifying its run (empty when nothing does): a sampled certificate
    that did not pass; a static record whose verdict is failed or unread, or names no reason or condition; a static
    record whose ``passed`` is not its theorem being established (W15's rule)."""
    out = []
    for c in record.accuracy.certificates:
        if c.backend != STATIC_BACKEND:
            if not c.passed:
                out.append(f"{c.name} [{c.backend}] did not pass")
            continue
        kind, named = static_verdict(c)
        if bool(c.passed) != (kind == "established"):
            out.append(f"{c.name}: passed {c.passed} on a verdict of class {kind}")
        if kind not in ("established", *NAMED_STATIC) or (kind != "established" and not named):
            out.append(f"{c.name}: {kind} {named!r}")
    return out


def variant_config(name: str, residency_plan, out_dir: Path) -> dict:
    template, scale, overrides = VARIANTS[name]
    cfg = json.loads(next(TEMPLATES.glob(f"{template}_*.json")).read_text())
    cfg = copy.deepcopy(cfg)
    cfg["deployment_parameters"].update(overrides)
    platform = cfg["platform_constraints"]
    platform["cores"] = [{**ct, "count": max(1, int(ct["count"] * scale))} for ct in platform["cores"]]
    if residency_plan is not None:
        platform["residency_plan"] = residency_plan
    cfg["experiment_name"] = name if residency_plan is not None else f"{name}_u1"
    cfg["generated_files_path"] = str(out_dir)
    return cfg


def _run(cfg: dict):
    ensure_nevresim_path()                       # the pipeline's nevresim step reads the driver path (run.py's init)
    start = time.time()
    session = PipelineSession.from_config(cfg)
    try:
        session.run()
    finally:
        session.finish()
    run_dir = Path(session.pipeline.working_directory)
    record = load_deployment_record(str(run_dir / DEPLOYMENT_RECORD_FILENAME))
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[{cfg['experiment_name']}] wall {time.time() - start:.1f} s, peak RSS {peak_mb:.0f} MB")
    return record


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_the_variant_gives_the_verdicts_of_its_u1_twin(name, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("MIMARSINAN_LOIHI_WAVE_WORKERS", "1")
    u1 = _run(variant_config(name, None, tmp_path / "u1"))
    u1_out = capfd.readouterr().out
    executed = _run(variant_config(name, {"step": "floor"}, tmp_path / "residency"))
    executed_out = capfd.readouterr().out
    u1_gates, executed_gates = gate_verdicts(u1_out), gate_verdicts(executed_out)
    verdicts = certificate_verdicts
    with capfd.disabled():
        for text in (u1_out, executed_out):
            print("\n".join(line for line in text.splitlines() if " wall " in line and "peak RSS" in line))
        print(f"[{name}] gates: U1 twin {u1_gates}; executed {executed_gates}")
        print(f"[{name}] certificates: U1 twin {sorted(verdicts(u1).items())}; "
              f"executed {sorted(verdicts(executed).items())}")
    assert executed.identity.platform.get("residency_plan") == {"step": "floor"}
    assert "residency_plan" not in u1.identity.platform
    assert executed_gates == u1_gates
    assert RAN[name] <= set(executed_gates)
    assert all(verdict != "FAIL" for _, _, verdict in executed_gates)
    assert verdicts(executed) == verdicts(u1)
    assert {n for n, backend in verdicts(executed) if backend == STATIC_BACKEND} == STATIC_RECORDS
    assert uncertified(executed) == [] == uncertified(u1), (uncertified(executed), uncertified(u1))
    assert executed.schedule.pass_count >= 2


#: Gate lines as t0_44r's two runs print them (abridged): the executed run differs from its twin in passes and
#: numbers only, which ``gate_verdicts`` must ignore, and in a verdict, which it must not.
_TWIN_LINES = """[convert_torch_model] value parity PASS (converted model vs native_eval_identity): relative error 0.000e+00
[ValueParityGate] PASS model\u2261identity [AQ grid] (max|delta|=6.337e-08 = 0.00 LSB, bound=1 LSB, n=2)
[HardCoreMappingStep] Hybrid program (scheduled): 5 neural segment(s) (seg 0: 9 passes), 6 compute op(s)
[SpikeCountCertificate] SKIP (none): None
[SpikeCountCertificate] spike-count certificate [sanafe/exact]: PASS exact=1.000000 over 1940 neuron-windows
Loihi spike parity: PASS (2 neural segment(s), 7 core(s), sample_index=0)
[ValueTwinCertificate] PASS backend=value_twin samples=2 windows=16060 max|delta|=8.882e-16"""


def test_gate_verdicts_ignore_numbers_and_keep_verdicts():
    twin = gate_verdicts(_TWIN_LINES)
    assert twin == [("convert_torch_model", "(converted model vs native_eval_identity)", "PASS"),
                    ("ValueParityGate", "", "PASS"), ("SpikeCountCertificate", "", "SKIP"),
                    ("SpikeCountCertificate", "[sanafe/exact]", "PASS"), ("Loihi spike parity", "", "PASS"),
                    ("ValueTwinCertificate", "", "PASS")]
    renumbered = _TWIN_LINES.replace("9 passes", "1 pass").replace("1940", "2068").replace("16060", "8030")
    assert gate_verdicts(renumbered) == twin
    skipped = _TWIN_LINES.replace("[ValueTwinCertificate] PASS backend", "[ValueTwinCertificate] SKIP (unarmed)")
    assert gate_verdicts(skipped) != twin
    assert RAN["t0_44r"] <= set(twin) and not RAN["t0_44r"] <= set(gate_verdicts(skipped))


def _static_report(*, in_scope=True, scope_note="", statuses=(), unknown=False, cascaded=False):
    """A static check report of the given condition statuses (``(condition, theorem, status)``), as
    ``check_deployment`` assembles one, so its records carry the verdict texts the check writes."""
    from mimarsinan.certification.program_check.report import (
        ConditionResult, ConditionSpec, StaticCheckReport, Violation,
    )

    report = StaticCheckReport(
        semantics="value_integer", in_scope=in_scope, scope_note=scope_note,
        results={c: ConditionResult(c, status) for c, _theorem, status in statuses},
        specs=tuple(ConditionSpec(c, theorem, "program", "") for c, theorem, _status in statuses),
        unknown_forms=[Violation("forms", "a_form", "unit 0", "unregistered")] if unknown else [])
    if cascaded:
        report.proposition2_note = "not claimed: cascaded TTFS is in Theorem 1 only (its greedy schedule is lossy)"
    return report


def _record(*records):
    from types import SimpleNamespace

    from mimarsinan.deployment_record.schema import CertificateRecord

    return SimpleNamespace(accuracy=SimpleNamespace(certificates=tuple(CertificateRecord.from_dict(r) for r in records)))


def test_the_static_verdicts_a_certified_run_may_record_are_named_and_never_failed():
    """[fix round M1] W15 records a static record as passed only where its theorem is established; an outside-scope,
    conditional or not-claimed verdict is recorded by name, never as a pass. A certified run's records: every sampled
    certificate passes, and every static record is established or carries its stated reason or its conditions."""
    from mimarsinan.certification.program_check.report import ESTABLISHED, FAILED, NOT_STATIC, P2, T1
    from mimarsinan.deployment_record.schema import CertificateRecord
    from mimarsinan.pipelining.core.gates.static_program_gate import _error_records

    sampled = CertificateRecord("spike_count_streaming_twin", "hcm", True, 10, 1.0, 0.0, "").to_dict()
    established = _static_report(statuses=(("A", T1, ESTABLISHED), ("B", P2, ESTABLISHED))).certificate_records()
    outside = _static_report(in_scope=False, scope_note="the value domain without a readout law runs the fp64 float "
                             "path, not the integer law").certificate_records()
    conditional = _static_report(statuses=(("A", T1, ESTABLISHED), ("H4", P2, NOT_STATIC))).certificate_records()
    not_claimed = _static_report(statuses=(("A", T1, ESTABLISHED),), cascaded=True).certificate_records()
    assert [static_verdict(CertificateRecord.from_dict(r)) for r in established] == [("established", "")] * 2
    assert [static_verdict(CertificateRecord.from_dict(r)) for r in outside] == [
        ("outside_scope", "the value domain without a readout law runs the fp64 float path, not the integer law")] * 2
    assert [static_verdict(CertificateRecord.from_dict(r)) for r in conditional] == [
        ("established", ""), ("conditional", "H4")]
    assert static_verdict(CertificateRecord.from_dict(not_claimed[1]))[0] == "not_claimed"
    for records in (established, outside, conditional, not_claimed):
        assert uncertified(_record(sampled, *records)) == []
        assert {n for (n, b) in certificate_verdicts(_record(sampled, *records)) if b == STATIC_BACKEND} == \
            STATIC_RECORDS
    # every failed form, a reason or condition missing, a pass on an unestablished verdict, a sampled failure
    failed = _static_report(statuses=(("A", T1, FAILED), ("B", P2, ESTABLISHED))).certificate_records()
    unknown = _static_report(statuses=(("A", T1, ESTABLISHED),), unknown=True).certificate_records()
    unnamed = _static_report(in_scope=False, scope_note="").certificate_records()
    errored = _error_records(ValueError("an element the check reads"))
    passed_conditional = [dict(r, passed=True) for r in conditional]
    for records, bad in ((failed, 1), (unknown, 2), (unnamed, 2), (errored, 2), (passed_conditional, 1)):
        assert len(uncertified(_record(sampled, *records))) == bad, (records, uncertified(_record(sampled, *records)))
    assert static_verdict(CertificateRecord.from_dict(errored[0]))[0] == "failed"
    assert uncertified(_record(dict(sampled, passed=False), *established)) == [
        "spike_count_streaming_twin [hcm] did not pass"]


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_t0_53rs_u1_twin_and_its_residency_steps_are_admitted_on_nevresim(monkeypatch, tmp_path):
    """At plan time (``run_pass_transfer`` with the built program, what the hard-core mapping step holds), with nevresim
    enabled as t0_53 enables it. [W33] t0_53r's U1 twin, whose second pass reads a carried slice at latency 1, and its
    residency variant are admitted (refused by name before W33: nevresim read its input at the global cycle). [W03] So
    is every step of the variant: on tile-granular units each step cuts inside a layer (the floor r 1 splits layer 1's
    two tiles, against the layer-granular floor's r 2 that kept one layer per pass), so a later layer of the pass reads
    a carried slice at latency 1. On a nevresim tree without W33's core-local read (base 90af7d2), which reads a stage
    input at the global cycle, the twin and every step are refused by name. At the fit (C = F) the program is one pass
    with no late reader, admitted on either tree."""
    from unit.mapping.residency.executed import certified as CF
    from unit.mapping.residency.executed import templates as TV
    from unit.chip_simulation import nevresim_trees as NT
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.mapping.packing.pass_timing import carried_nodes, late_stage_input_readers
    from mimarsinan.models.spiking.hybrid import carry as C
    from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

    vehicle = TV.template_vehicle("t0_53")
    assert vehicle.config.get("enable_nevresim_simulation")
    resolved = build_platform_constraints_resolved(vehicle.config)

    def scaled(scale):
        return {**resolved, "cores": [{**ct, "count": max(1, int(ct["count"] * scale))} for ct in resolved["cores"]]}

    platform = scaled(VARIANTS["t0_53r"][1])
    twin = build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=vehicle.config)
    assert any(late_stage_input_readers(s, carried_nodes(twin.stages, s)) for s in twin.stages if s.kind == "neural")
    assert run_pass_transfer(vehicle.config, hybrid_mapping=twin) == "verbatim"
    variant = build_hybrid_mapping_for_pipeline(vehicle.ir, {**platform, "residency_plan": {"step": "floor"}},
                                                pipeline_config=vehicle.config)
    assert run_pass_transfer(vehicle.config, hybrid_mapping=variant) == "verbatim"
    steps = CF.builder_composition(vehicle.ir, platform, {"step": "floor"}).plan.steps
    assert steps[0][0] == 1 and len(steps) >= 2
    step_programs = [build_hybrid_mapping_for_pipeline(
        vehicle.ir, {**platform, "residency_plan": {"step": "index", "index": index}}, pipeline_config=vehicle.config)
        for index in range(len(steps))]
    for program in step_programs:
        assert any(late_stage_input_readers(s, carried_nodes(program.stages, s))
                   for s in program.stages if s.kind == "neural")
        assert run_pass_transfer(vehicle.config, hybrid_mapping=program) == "verbatim"
    fit = build_hybrid_mapping_for_pipeline(vehicle.ir, {**scaled(3.0), "residency_plan": {"step": "floor"}},
                                            pipeline_config=vehicle.config)
    assert len([s for s in fit.stages if s.kind == "neural"]) == 1
    assert run_pass_transfer(vehicle.config, hybrid_mapping=fit) == "verbatim"
    monkeypatch.setattr(NevresimDriver, "nevresim_path", str(NT.base_tree(tmp_path)))     # a tree without W33's read
    with pytest.raises(C.ChipClockReplayRefusal, match="nevresim reads a stage input at the global cycle"):
        run_pass_transfer(vehicle.config, hybrid_mapping=twin)
    for program in step_programs:
        with pytest.raises(C.ChipClockReplayRefusal, match="stage-input rows read by cores that start above latency 0"):
            run_pass_transfer(vehicle.config, hybrid_mapping=program)
    assert run_pass_transfer(vehicle.config, hybrid_mapping=fit) == "verbatim"        # no late reader: either tree


def _nevresim_final_counts(mapping, xs, *, transfer: str):
    """nevresim's final output counts per sample (``output_sources`` gathered from the stages' output counts), run
    through ``SimulationRunner._run_hybrid`` with its refusals in force."""
    import tempfile

    import numpy as np
    import torch

    from mimarsinan.chip_simulation.nevresim.connectivity import default_nevresim_connectivity_mode
    from mimarsinan.chip_simulation.simulation_runner.core import SimulationRunner
    from unit.mapping.residency.executed import vehicles as V

    runner = object.__new__(SimulationRunner)
    vars(runner).update(
        spike_generation_mode="Uniform", firing_mode="Default", thresholding_mode="<=", spiking_mode="lif",
        nevresim_connectivity_mode=default_nevresim_connectivity_mode(), pass_transfer=transfer,
        simulation_step_timeout_s=600.0, membrane_readout=False, membrane_half_step_charge=False, weight_type=float,
        threshold_type=float, host_compute_device="cpu", stage_timer=None, simulation_length=V.T, mapping=mapping,
        num_classes=64, test_data=[(x[0].numpy().astype(np.float64), np.zeros(1)) for x in xs],
        test_targets=[torch.tensor(0) for _ in xs], test_input=[], working_directory=tempfile.mkdtemp())
    node = {}

    def record(stage, raw):
        counts = np.rint(np.asarray(raw)).astype(np.int64)
        for s in stage.output_map:
            node[int(s.node_id)] = counts[:, int(s.offset):int(s.offset) + int(s.size)]
    runner.stage_count_recorder = record
    runner._run_hybrid(mapping)
    return np.stack([node[int(s.node_id)][:, int(s.index)] for s in np.asarray(mapping.output_sources).ravel()], 1)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_nevresim_runs_a_re_timed_chain_cut_equal_to_the_hcm_identity_run():
    """[Review round 2] Windowed lif (per-hop re-timed level units, COLLAPSE: the deployed lifsync), chain C 3: pass 4
    reads its carried slice at latency 1 as a pass but at 0 in the level units nevresim executes, so nevresim admits
    the program, and its final counts equal the HCM's identity run (and the HCM's U1 run) on 3 seeded samples."""
    import numpy as np
    import torch

    from integration.parity_harness import ensure_nevresim_ready
    from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
    from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE
    from unit.mapping.residency.executed import runs as R
    from unit.mapping.residency.executed import vehicles as V

    ensure_nevresim_ready()
    ir = V.ir_of(V.chain_repr(seed=0))
    u1 = V.build(ir, 3, None, retimed_level_stages=True)
    identity = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=True)
    assert len(V.neural_passes(u1)) == 4
    xs = [torch.rand(1, V.IN, generator=torch.Generator().manual_seed(b)) for b in range(3)]

    def hcm(mapping):
        node = {}
        flow = R.flow(mapping, **R.WINDOWED_LIF)

        def record(stage, counts):
            c = counts.detach().cpu().numpy().astype(np.int64)
            for s in stage.output_map:
                node[int(s.node_id)] = c[:, int(s.offset):int(s.offset) + int(s.size)]
        flow.stage_count_recorder = record
        with torch.no_grad():
            flow(torch.cat(xs))
        return np.stack([node[int(s.node_id)][:, int(s.index)] for s in np.asarray(mapping.output_sources).ravel()], 1)

    want = hcm(identity)
    assert np.array_equal(hcm(u1), want) and want.std() > 0
    got = _nevresim_final_counts(u1, xs, transfer=COLLAPSE)
    assert np.array_equal(got.reshape(want.shape), want)
