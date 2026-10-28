"""[E1 open item 1] The pipeline's static check receives the IR at unbounded geometry: Soft Core Mapping maps the IR a
second time without core limits (no ``max_axons``, ``max_neurons`` or coalescing) through the same passes, on a copy of
the model as the declared mapping finds it, and promises it as ``ir_graph_unbounded``; Hard Core Mapping hands it to
the static check, so a pipeline verdict establishes ``ir-tiling`` instead of assuming it. The deployed IR and model are
those of a run without the second mapping, and a second mapping that cannot run is recorded by name, never a refusal."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.certification import program_check
from mimarsinan.certification.program_check.report import ESTABLISHED, ConditionResult
from mimarsinan.certification.program_check.unbounded import check_ir_tiling
from mimarsinan.pipelining.core.gates import static_program_gate as gate
from mimarsinan.pipelining.pipeline_steps.mapping import soft_core_mapping_step as scm
from mimarsinan.pipelining.pipeline_steps.mapping.hard_core_mapping_step import HardCoreMappingStep
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import SoftCoreMappingStep

from unit.certification.program_check import programs as P

#: The tiny flow's second 16-neuron layer (the first is its encoding layer, on the host) splits into two tiles at 8
#: neurons per core.
TILED = {"cores": [{"max_axons": 128, "max_neurons": 8, "count": 40}], "weight_bits": 8}


def _run_scm(directory):
    pipeline = MockPipeline(config=default_config(), working_directory=str(directory))
    pipeline.config["onchip_majority_gate"] = False
    pipeline.config["scm_torch_sim_parity_check"] = False
    torch.manual_seed(0)
    model = make_tiny_supermodel(hidden_layers=2)
    for perceptron in model.get_perceptrons():
        perceptron.normalization = nn.Identity()
    pipeline.seed("fused_model", model, step_name="Normalization Fusion")
    pipeline.seed("platform_constraints_resolved", json.loads(json.dumps(TILED)), step_name="Model Configuration")
    step = SoftCoreMappingStep(pipeline)
    step.name = "Soft Core Mapping"
    pipeline.prepare_step(step)
    step.run()
    return pipeline, model


def _ir_fingerprint(ir):
    cores = []
    for node in ir.get_neural_cores():
        cores.append((node.name, int(node.latency or 0), np.asarray(node.get_core_matrix(ir)).tobytes(),
                      float(np.asarray(node.threshold)), tuple(getattr(node, "perceptron_output_slice", None) or ())))
    return cores


def test_soft_core_mapping_promises_it_and_hard_core_mapping_requires_it():
    assert "ir_graph_unbounded" in SoftCoreMappingStep.PROMISES
    assert "ir_graph_unbounded" in HardCoreMappingStep.REQUIRES


def test_the_ir_at_unbounded_geometry_holds_every_perceptron_untiled_and_ir_tiling_holds(tmp_path):
    pipeline, model = _run_scm(tmp_path)
    ir = pipeline.cache["Soft Core Mapping.ir_graph"]
    reference = pipeline.cache["Soft Core Mapping.ir_graph_unbounded"]
    assert reference["note"] == "built"
    unbounded = reference["ir"]
    unbounded_cores = unbounded.get_neural_cores()
    assert len(ir.get_neural_cores()) > len(unbounded_cores) >= 2
    assert len({int(c.perceptron_index) for c in unbounded_cores}) == len(unbounded_cores)   # one core per perceptron
    declaration = program_check.declaration_from_config(pipeline.config, cores=TILED["cores"], has_laws=False)
    result = ConditionResult("ir-tiling", ESTABLISHED)
    check_ir_tiling(ir, unbounded, declaration=declaration, result=result)
    assert result.violation_count == 0 and result.checked > 0, [v.text() for v in result.violations[:3]]


def test_the_deployed_ir_and_model_are_those_of_a_run_without_the_second_mapping(tmp_path, monkeypatch):
    with_reference, model_a = _run_scm(tmp_path / "with")
    monkeypatch.setattr(SoftCoreMappingStep, "_derive_unbounded_ir",
                        lambda self, *args, **kwargs: {"ir": None, "note": "not derived (test)"})
    without, model_b = _run_scm(tmp_path / "without")
    assert _ir_fingerprint(with_reference.cache["Soft Core Mapping.ir_graph"]) == \
        _ir_fingerprint(without.cache["Soft Core Mapping.ir_graph"])
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    assert state_a.keys() == state_b.keys() and all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    assert without.cache["Soft Core Mapping.ir_graph_unbounded"] == {"ir": None, "note": "not derived (test)"}


def test_a_second_mapping_that_cannot_run_is_recorded_by_name(tmp_path, monkeypatch):
    def broken(model):
        raise ValueError("no copy of this model")

    monkeypatch.setattr(scm, "model_copy_for_reference", broken)
    pipeline, _model = _run_scm(tmp_path)
    reference = pipeline.cache["Soft Core Mapping.ir_graph_unbounded"]
    assert reference["ir"] is None and reference["note"].startswith("ValueError: no copy of this model")
    assert pipeline.cache["Soft Core Mapping.ir_graph"] is not None


def test_the_static_gate_establishes_ir_tiling_given_the_ir_at_unbounded_geometry(tmp_path, monkeypatch):
    """X3a R4 seed 7 (a tiled, relabelled, dithered deployment): through the gate, against its IR at unbounded
    geometry, the verdict establishes ir-tiling and assumes no IR tiling."""
    from dataclasses import replace
    from types import SimpleNamespace

    from unit.certification.program_check import test_ir_tiling as T

    instance, constraints, arm = T._setting(7)
    unbounded = T.reference(instance, arm).ir
    ir = T._tiled_ir(instance, constraints, 7, arm)
    program = T._program_of(ir, instance, constraints)
    decl = replace(T._windowed(True), window=int(instance.timesteps),
                   core_types=tuple(dict(c) for c in constraints["cores"]))
    monkeypatch.setattr(gate.program_check, "declaration_from_config", lambda *a, **k: decl)
    pipeline = SimpleNamespace(config={}, working_directory=str(tmp_path))
    monkeypatch.setattr(gate, "backend_accumulation", lambda _pipeline: {})
    report, records = gate.run_static_program_check(pipeline, ir, program, cores=constraints["cores"],
                                                    unbounded_ir=unbounded)
    assert report.results["ir-tiling"].status == ESTABLISHED and report.results["ir-tiling"].checked
    detail = json.loads(records[0]["detail"])
    assert detail["conditions"]["ir-tiling"] == ESTABLISHED and "ir-tiling" not in detail["assumed_conditions"]
    assert "IR tiling" not in detail["assumed"] and detail["reference"] == "the IR at unbounded geometry"


def test_without_it_the_gate_keeps_ir_tiling_assumed_and_names_why(tmp_path):
    from types import SimpleNamespace

    program, ir = P.streamed_multipass()
    pipeline = SimpleNamespace(config={"spiking_family": "lif", "spiking_variant": "streamed",
                                       "simulation_steps": P.T}, working_directory=str(tmp_path))
    report, _records = gate.run_static_program_check(pipeline, ir, program, cores=P.V.cores(3), unbounded_ir=None,
                                                     unbounded_note="ValueError: no copy of this model")
    tiling = report.results["ir-tiling"]
    assert tiling.status == "assumed" and "ValueError: no copy of this model" in tiling.note


def test_hard_core_mapping_hands_the_entry_to_the_static_check(tmp_path, monkeypatch):
    import mimarsinan.pipelining.pipeline_steps.mapping.hard_core_mapping_step as hcm
    from unit.deployment_record.test_deployment_record_entries import _hcm_pipeline, _run_hard_core_mapping

    seen = {}
    real = hcm.run_static_program_check

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(hcm, "run_static_program_check", spy)
    pipeline = _hcm_pipeline(tmp_path)
    marker = object()
    pipeline.seed("ir_graph_unbounded", {"ir": marker, "note": "built"})
    monkeypatch.setattr(gate.program_check, "check_deployment", lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    _run_hard_core_mapping(pipeline, monkeypatch)
    assert seen["unbounded_ir"] is marker and seen["unbounded_note"] == "built"
