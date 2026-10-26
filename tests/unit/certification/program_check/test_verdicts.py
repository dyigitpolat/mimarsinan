"""[W15] What a verdict says: Theorem 1 is established on the program, relative to the IR at the declared geometry,
with the executor code clauses it assumes named in the verdict and the certificate; Proposition 2 reads the seam
currencies and the half-step fold; a compile wall that is a cache load is never priced as a compile."""

from __future__ import annotations

import copy
import json

import torch

from mimarsinan.certification.program_check.report import NOT_STATIC

from unit.certification.program_check import programs as P
from unit.mapping.residency.executed import vehicles as V


def test_an_established_verdict_names_the_clauses_it_assumes():
    program, ir = P.streamed_multipass()
    report = P.run(program, ir, P.decl("streamed_lif"))
    assert report.theorem1.startswith("established on the program"), report.theorem1
    for clause in ("H0 dispatch", "H8 operand", "R6 walk", "stage-input local time", "IR tiling"):
        assert clause in report.theorem1, (clause, report.theorem1)
    assumed = {a["name"]: a for a in report.to_dict()["assumed"]}
    assert assumed["R6 walk"]["status"] == "assumed_executor" and "sync_counts.py" in assumed["R6 walk"]["site"]
    assert assumed["IR tiling"]["status"] == "assumed_reference"
    record = report.certificate_records()[0]
    detail = json.loads(record["detail"])
    assert record["passed"] and "H0 dispatch" in detail["assumed"]
    assert detail["reference"] == "the IR at the declared geometry"


def test_a_failed_verdict_still_names_its_failures_first():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    core = P.neural(bad)[0].hard_core_mapping.cores[0]
    core.threshold = float(core.threshold) + 1.0
    report = P.run(bad, ir, P.decl("streamed_lif"))
    assert report.theorem1.startswith("failed: H1-theta") and not report.established("theorem1")
    assert not report.certificate_records()[0]["passed"]


def _seam_consumer(ir):
    """An IR core that reads a host op's output (a seam of the stage loop's decode and re-encode)."""
    from mimarsinan.mapping.ir import ComputeOp, NeuralCore

    hosts = {int(n.id) for n in ir.nodes if isinstance(n, ComputeOp)}
    for node in ir.nodes:
        if isinstance(node, NeuralCore) and any(int(s.node_id) in hosts for s in node.input_sources.flatten()):
            return node
    raise AssertionError("no core reads a host op")


def test_a_seam_off_unit_gauges_leaves_proposition_2_to_the_sampled_run():
    """At unit gauges every seam is coherent by construction; one changed entry currency makes the seam's coherence
    the install certificate's question (calculus 11.2), so Proposition 2 reads conditional on it, never established."""
    ir = P.integer_ir(V.split_level_repr())
    program = V.build(ir, 3, None)
    clean = P.run(program, ir, P.decl("synchronized_lif"))
    assert clean.results["seam-currency"].status == "established" and clean.results["seam-currency"].checked
    bad_ir = copy.deepcopy(ir)
    _seam_consumer(bad_ir).input_activation_scale = torch.tensor(2.0)
    report = P.run(V.build(bad_ir, 3, None), bad_ir, P.decl("synchronized_lif"))
    assert report.results["seam-currency"].status == NOT_STATIC
    assert report.proposition2.startswith("conditional") and "seam-currency" in report.proposition2


def _seam_status(ir, identity) -> str:
    from mimarsinan.certification.program_check import network as N
    from mimarsinan.certification.program_check.report import ESTABLISHED, ConditionResult

    result = ConditionResult("seam-currency", ESTABLISHED)
    N.check_seams(ir, identity, result)
    assert result.checked, result.note
    return result.status


def test_a_per_channel_entry_currency_whose_mean_is_one_is_off_unit_gauges():
    """[exact V2] A seam reader's per-channel entry currency 0.5, 1.5, ... (mean 1) is off unit gauges on every
    channel; the seam's coherence is then the install certificate's, never 'coherent by construction' by its mean."""
    import numpy as np

    ir = P.integer_ir(V.split_level_repr())
    identity = P.identity_of(ir)
    assert _seam_status(ir, identity) == "established"
    bad = copy.deepcopy(ir)
    node = _seam_consumer(bad)
    width = int(np.asarray(node.core_matrix).shape[0])
    node.input_activation_scale = torch.tensor([0.5, 1.5] * (width // 2) + [1.0] * (width % 2))
    assert _seam_status(bad, identity) == NOT_STATIC


def test_a_per_channel_host_gauge_whose_mean_is_one_is_off_unit_gauges():
    """[exact V2] A host source emitting at the per-channel gauge 0.25, 1.75 (mean 1) is off unit gauges: the seam
    reader's coherence stays with the install certificate and the sampled run."""
    import numpy as np
    from mimarsinan.mapping.ir import ComputeOp

    ir = P.integer_ir(V.split_level_repr())
    identity = copy.deepcopy(P.identity_of(ir))
    scales = dict(getattr(identity, "node_activation_scales", None) or {})
    for node in ir.nodes:
        if isinstance(node, ComputeOp):
            scales[int(node.id)] = np.array([0.25, 1.75])
    identity.node_activation_scales = scales
    assert _seam_status(ir, identity) == NOT_STATIC


def test_a_folded_half_step_leaves_proposition_2_to_the_sampled_run():
    program, ir = P.streamed_multipass()
    plain = P.run(program, ir, P.decl("synchronized_ttfs"))
    assert plain.results["half-step"].status in ("established", "vacuous")
    for folded in (dict(comparator_half_step=True), dict(half_step_folded=True)):
        report = P.run(program, ir, P.decl("synchronized_ttfs", **folded))
        assert report.results["half-step"].status == NOT_STATIC, folded
        assert report.proposition2.startswith("conditional") and "half-step" in report.proposition2


def test_a_declaration_reads_the_half_step_fold_and_the_trains():
    from mimarsinan.certification.program_check.declaration import declaration_from_config

    decl = declaration_from_config({"spiking_family": "ttfs", "spiking_variant": "synchronized",
                                    "spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "synchronized",
                                    "activation_quantization": True, "simulation_steps": 4})
    assert decl.half_step_folded and not decl.comparator_half_step
    decl = declaration_from_config({"spiking_family": "lif", "spiking_variant": "streamed", "spiking_mode": "lif",
                                    "cycle_accurate_lif_forward": True, "simulation_steps": 4,
                                    "lif_half_step_bias": True})
    assert decl.cycle_accurate_trains and not decl.half_step_folded


def test_a_loaded_mapping_is_not_priced_as_a_compile(tmp_path):
    from types import SimpleNamespace

    from mimarsinan.pipelining.core.gates import static_program_gate as gate

    program, ir = P.streamed_multipass()
    pipeline = SimpleNamespace(config={"spiking_family": "lif", "spiking_variant": "streamed",
                                       "simulation_steps": P.T}, working_directory=str(tmp_path))
    report, records = gate.run_static_program_check(pipeline, ir, program, cores=P.V.cores(3),
                                                    compile_wall_s=0.25, compile_source="loaded")
    assert report.compile_wall_s is None and report.counts["compile_source"] == "loaded"
    assert json.loads(records[0]["detail"])["compile_source"] == "loaded"
    report, _ = gate.run_static_program_check(pipeline, ir, program, cores=P.V.cores(3), compile_wall_s=0.25,
                                              compile_source="built")
    assert report.compile_wall_s == 0.25


def test_the_cached_mapping_lookup_the_step_prices_by():
    from types import SimpleNamespace

    from mimarsinan.pipelining.core.hybrid_mapping_consumer import cached_hybrid_mapping

    ir = SimpleNamespace(build_token="t1")
    mapping = SimpleNamespace(source_ir_build_token="t1")
    cache = {"hybrid_mapping": mapping}
    pipeline = SimpleNamespace(cache=SimpleNamespace(get=cache.get))
    step = SimpleNamespace(get_entry=lambda key: ir)
    assert cached_hybrid_mapping(pipeline, step) is mapping
    ir.build_token = "t2"
    assert cached_hybrid_mapping(pipeline, step) is None
