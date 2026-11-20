"""CL3 (d)-(e): executed residency plans certified on the integrated tree (``RA/11_wave2/explore/cl3/`` prints the
same rows as evidence, ``scripts/cl3_certify.py``).

(d) Per (member, step) of the floor and stateless steps: the record's residency marks and plan equal the executed
    program's attributes; the record-plane period census equals ``plan_period_census`` at the executed step (the
    builder's composition) in the pass partition, Q, the pins and the carry per boundary, and in cells and bytes
    where both packers pack every phase alike (A1); it equals the candidate plane's census on the unpruned
    programs, while on t0_44r (pruning) the difference is printed; the value programs run NF = SCM = HCM = the int64
    golden at atol 0 under the integer readout law; t0_53r's pipeline gates and sampled certificates all pass. [fix
    round M1] Every static program check record (W15) is established, outside scope with its stated reason, or
    conditional with its conditions, never failed; under the integer readout law both are established.
(e) The step that minimizes the toolchain-priced writes plus carry under ``generic_estimated_22nm`` is the floor on
    CIFAR ResNet-18 w1/4 at every K (its step function has one step: every cut of the value program is a forced
    segment cut, so the carry is 0 at every r); on the spiking deep CNN of t0_03 at twice its cores a non-floor step
    wins and runs equal to the identity run.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import pytest
import torch

from integration.test_residency_executed import (
    RAN, STATIC_BACKEND, STATIC_RECORDS, _run, certificate_verdicts, gate_verdicts, uncertified, variant_config,
)
from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.cost.absolute.step_choice import cheapest_step
from mimarsinan.deployment_record.cost.absolute.walls.static_power import NO_STATIC
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer
from mimarsinan.pipelining.cache.load_store_strategies import PickleLoadStoreStrategy, TorchModelLoadStoreStrategy
from mimarsinan.pipelining.core.simulation_factory import (
    build_hybrid_mapping_for_pipeline,
    build_identity_mapping_for_pipeline,
    build_spiking_hybrid_flow,
)
from mimarsinan.search.problems.joint.candidate_fragments import StageSemantics
from unit.chip_simulation.value_run.integer_programs import build_integer_program, integer_inputs, resnet18_quarter
from unit.mapping.residency.executed import certified as CF
from unit.mapping.residency.executed import templates as TV

R18_CORES = [{"max_axons": 1153, "max_neurons": 128, "count": 400, "has_bias": False}]
R18_PLATFORM = {"cores": R18_CORES, "weight_bits": 8, "core_semantics": "mvm", "activation_bits": 8,
                "allow_scheduling": True, "allow_coalescing": False, "allow_neuron_splitting": False,
                "max_schedule_passes": 32}
READOUT_LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 16}
STEPS = ({"step": "floor"}, {"step": "stateless"})


@pytest.fixture(scope="module")
def resnet18_quarter_program():
    pmap = resolve_platform_mapping_params(R18_CORES, allow_coalescing=False)
    with contextlib.redirect_stdout(io.StringIO()):
        program = build_integer_program(resnet18_quarter(), (3, 32, 32), num_classes=10,
                                        max_axons=pmap.effective_max_axons, max_neurons=pmap.effective_max_neurons)
    return program, CF.deferred_walk(program.model.get_mapper_repr(), R18_CORES, allow_coalescing=False)


def _attributes_sealed(hybrid, schedule):
    assert CF.record_marks(schedule) == CF.executed_marks(hybrid)
    assert CF.record_plan(schedule) == CF.executed_plan(hybrid)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("plan", STEPS, ids=lambda p: p["step"])
def test_resnet18_quarter_executes_and_certifies_the_step(resnet18_quarter_program, plan):
    program, walk = resnet18_quarter_program
    hybrid = build_hybrid_mapping_for_pipeline(program.ir, {**R18_PLATFORM, "residency_plan": plan})
    _attributes_sealed(hybrid, schedule_record_from_mapping(hybrid, weight_bits=8, params_reloaded=0))
    planes = CF.planes(hybrid, program.ir, walk, R18_PLATFORM, plan, timesteps=None, semantics=StageSemantics())
    assert len(planes.record.passes) >= 2
    assert CF.differences(planes.record, planes.plan) == [] == CF.differences(planes.record, planes.candidate)
    certificate = CF.integer_certificate(program.model, program.ir, hybrid, integer_inputs(program, 64))
    print(f"[CL3 d] ResNet-18 w1/4 {plan['step']}: Q {len(planes.record.passes)}, certificate {certificate}")
    assert CF.exact(certificate) and certificate["nonzero_output_share"][0] > 0


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_the_priced_step_of_resnet18_quarter_is_its_floor(resnet18_quarter_program):
    program, _walk = resnet18_quarter_program
    composition = CF.builder_composition(program.ir, R18_PLATFORM, {"step": "floor"})
    assert len(composition.plan.steps) == 1 and composition.plan.steps[0][1] == 0
    census = CF.facts_period_census(composition.facts, forced=composition.units.forced,
                                    bytes_of_cells=lambda cells: CF.params_bytes(cells, 8), tile_cells=0,
                                    bytes_per_element=1)
    physics = get_platform_physics("generic_estimated_22nm")
    assert all(cheapest_step([census], physics, WritePolicy(batch_k=k), static=NO_STATIC) == 0 for k in (1, 16, 64))


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_a_priced_non_floor_step_executes_equal_to_the_identity_run():
    """t0_03 below its footprint: under generic_estimated_22nm at K 256 the cheapest step is not the floor; executed
    through ``{step: index}`` it runs equal to the identity run, and its record holds the executed attributes. [W03]
    On tile-granular units at twice its cores the floor stayed the cheapest step to K 64 and the carry-free step 1 won
    from K 256. [fix_c] At twice its cores (360 + 360) the declared pool holds the unscheduled pack (213 + 213), so the
    fit rule makes every declared step that pack, programmed once; the witness runs at x1.15 (206 + 206), where the
    steps are (r, carry, Q) = (30, 2432, 12), (31, 1792, 12), (32, 640, 12), (33, 0, 11) and the carry-free step 3 is
    the cheapest from K 16 (fix_c probe ``t0_03_priced_step.py``)."""
    vehicle = TV.template_vehicle("t0_03")
    platform = {**TV.residency_platform(vehicle, 1.15, None), "weight_bits": int(vehicle.config.get("weight_bits", 8))}
    timesteps = int(vehicle.config.get("simulation_steps", 32))
    steps = CF.builder_composition(vehicle.ir, platform, {"step": "floor"}).plan.steps
    censuses = []
    for index in range(len(steps)):
        comp = CF.builder_composition(vehicle.ir, platform, {"step": "index", "index": index})
        censuses.append(CF.facts_period_census(
            comp.facts, forced=comp.units.forced, bytes_of_cells=lambda cells: CF.params_bytes(cells, platform["weight_bits"]),
            tile_cells=0, bytes_per_element=CF.bytes_per_element(platform, timesteps)))
    index = cheapest_step(censuses, get_platform_physics("generic_estimated_22nm"), WritePolicy(batch_k=256), static=NO_STATIC)
    assert index > 0
    TV.activate(vehicle.ir, 32.0)
    plan = {"step": "index", "index": index}
    mapping = build_hybrid_mapping_for_pipeline(vehicle.ir, {**platform, "residency_plan": plan},
                                                pipeline_config=vehicle.config)
    _attributes_sealed(mapping, schedule_record_from_mapping(
        mapping, weight_bits=platform["weight_bits"], params_reloaded=0, timesteps=timesteps,
        pass_transfer=run_pass_transfer(vehicle.config, hybrid_mapping=mapping)))
    identity = build_spiking_hybrid_flow(vehicle.pipeline(), build_identity_mapping_for_pipeline(
        vehicle.ir, pipeline_config=vehicle.config))
    executed = build_spiking_hybrid_flow(vehicle.pipeline(), mapping)
    outputs = []
    for batch in range(4):
        x = torch.rand(4, *vehicle.input_shape, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            want = identity(x)
            assert torch.equal(executed(x), want)
        outputs.append(want)
    assert float(torch.cat(outputs).std(0).max()) > 0


def _pipeline(tmp_path, variant, plan, readout_law):
    cfg = variant_config(variant, plan, tmp_path / f"{variant}_{plan['step']}")
    if readout_law:
        cfg["platform_constraints"]["readout_law"] = dict(READOUT_LAW)
    cfg["device"] = "cpu"
    cfg["deployment_parameters"]["num_workers"] = 0
    record = _run(cfg)
    run_dir = next((tmp_path / f"{variant}_{plan['step']}").glob(f"{cfg['experiment_name']}_*"))
    return record, str(run_dir)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("variant, readout_law", [("t0_44r", True), ("t0_53r", False)])
@pytest.mark.parametrize("plan", STEPS, ids=lambda p: p["step"])
def test_the_pipeline_run_seals_and_certifies_the_executed_step(variant, readout_law, plan, tmp_path, monkeypatch,
                                                                 capfd):
    monkeypatch.setenv("MIMARSINAN_LOIHI_WAVE_WORKERS", "1")
    record, run_dir = _pipeline(tmp_path, variant, plan, readout_law)
    gates = gate_verdicts(capfd.readouterr().out)
    hybrid = PickleLoadStoreStrategy("Hard Core Mapping.hard_core_mapping").load(run_dir)
    ir = PickleLoadStoreStrategy("Soft Core Mapping.ir_graph").load(run_dir)
    platform = dict(record.identity.platform)
    assert platform["residency_plan"] == plan
    _attributes_sealed(hybrid, record.schedule)
    model_name = "Readout Law.model" if readout_law else sorted(
        Path(run_dir).glob("*.model.pt"), key=lambda p: p.stat().st_mtime)[-1].name[:-len(".pt")]
    model = TorchModelLoadStoreStrategy(model_name).load(run_dir)
    mapper_repr = model.get_mapper_repr()
    mapper_repr.assign_perceptron_indices()
    spiking = platform.get("core_semantics") != "mvm"
    planes = CF.planes(hybrid, ir, CF.deferred_walk(mapper_repr, platform["cores"],
                                                    allow_coalescing=bool(platform.get("allow_coalescing"))),
                       platform, plan, timesteps=int(record.timing.s_global) if spiking else None,
                       semantics=StageSemantics(), sealed_record=record)
    sealed = [d for d in CF.differences(planes.sealed, planes.record) if not d.startswith("carry per boundary")]
    with capfd.disabled():
        print(f"[CL3 d] {variant} {plan['step']}: Q {len(planes.record.passes)}; record vs plan "
              f"{CF.differences(planes.record, planes.plan)}; record vs candidate "
              f"{CF.differences(planes.record, planes.candidate)}; gates {gates}")
    assert sealed == []
    assert not [d for d in CF.differences(planes.record, planes.plan)
                if d.startswith(("Q ", "carry per boundary", "pass")) and "resident_cores" in d or d.startswith("Q ")]
    if variant != "t0_44r":
        assert CF.differences(planes.record, planes.candidate) == []
    assert RAN[variant] <= set(gates) and all(v != "FAIL" for _, _, v in gates)
    static = {n: v for (n, backend), v in certificate_verdicts(record).items() if backend == STATIC_BACKEND}
    assert set(static) == STATIC_RECORDS
    assert uncertified(record) == [], uncertified(record)
    if readout_law:
        # [fix round M1] under the integer readout law the static check establishes both theorems on the executed step
        assert all(v == ("established", "") for v in static.values()), static
        shape = tuple(TV.resolve(variant[:-1])["input_shape"])
        nf = CF.deployed_value_model(model, ir, weight_bits=int(platform["weight_bits"]))
        certificate = CF.integer_certificate(nf, ir, hybrid, integer_inputs(type("P", (), {"input_shape": shape})(), 64))
        with capfd.disabled():
            print(f"[CL3 d] {variant} {plan['step']}: integer certificate {certificate}")
        assert CF.exact(certificate) and certificate["nonzero_output_share"][0] > 0
