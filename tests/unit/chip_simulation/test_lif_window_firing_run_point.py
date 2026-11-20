"""[U-LIF2 r2] The point that the run resolved: no capability judgment and no policy build takes a point the run did not
resolve.

The two capability screens and the ledger's contract metadata take the run's point; each policy build receives the record
its caller holds; a holder that carries no run's point (a contract or behavior built from its own fields or stored before
the point existed, an executor built from its own fields) states its window firing explicitly, never through the declared
default; the ratchet t02c scans the capability judgments and refuses a mode-only point builder in src; a per-event
document names its own key at both doors under every declared set (a soma value its own key refuses never reaches the
joint table). [U-LIF2 r3] Every illegal value of every soma key names its own key at both doors and at the runtime
derivation; the ratchet refuses a plan resolved from a document read out of a run directory outside the readers' named
path, and holds its literal-document rule for a named literal too.

Red on the common part (its mode-only ``point_of_mode`` builds the declared default's point at these sites and drops the
run's window firing and execution extensions). Each test runs at both values of the window firing through the tests' own
widening fixture and with each execution extension. The new symbols are imported inside each test."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mimarsinan.config_schema.runtime import derive_run_config

FWI, ITE = "fire_while_integrating", "integrate_then_emit"
KEY = "lif_window_firing"
SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
_CORES = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 8}]}
_ENABLE = {"nevresim": "enable_nevresim_simulation", "sanafe": "enable_sanafe_simulation",
           "loihi": "enable_loihi_simulation"}
_EXTENSIONS = (("spike_phase_dither", True), ("lif_membrane_init", -0.25))
#: The four declared sets of the entry's data (its values, its default): today's, the two of option O1, option O2.
_SETTINGS = (((FWI,), FWI), ((FWI, ITE), FWI), ((FWI, ITE), ITE), ((ITE,), ITE))
#: Every (declared set, default, declaration, extension): silent and each declared value, with no extension and each.
_CASES = [(values, default, value, ext) for values, default in _SETTINGS for value in (None, *values)
          for ext in (None, *_EXTENSIONS)]
_IDS = [f"{'+'.join(v[0] for v in values)}~{default[0]}-{value or 'silent'}-{ext[0] if ext else 'plain'}"
        for values, default, value, ext in _CASES]
#: Both values with each extension, for the sites whose test runs a whole walk.
_BOTH_VALUES = [(value, ext) for value in (FWI, ITE) for ext in (None, *_EXTENSIONS)]


def _new(module: str, name: str):
    return getattr(importlib.import_module(module), name)


def _doc(**dp):
    return {"model_type": "simple_mlp", "spiking_family": "lif", "spiking_variant": "synchronized", **dp}


def _declarations(value, ext) -> dict:
    return {**({KEY: value} if value else {}), **(dict([ext]) if ext else {})}


@pytest.fixture
def declared(window_firing_declared):
    return window_firing_declared


@pytest.fixture
def run_point(declared):
    """(run config, the run's resolved point) of a synchronized LIF document under a declared set."""
    def at(values, default, value=None, ext=None, **dp):
        declared(values, default)
        cfg = derive_run_config(_doc(**_declarations(value, ext), **dp), _CORES)
        return cfg, _new("mimarsinan.chip_simulation.spiking_point", "resolve_spiking_point")(cfg)
    return at


# -- the two capability screens and the ledger's contract metadata take the run's point ------------------------------

@pytest.mark.parametrize("values,default,value,ext", _CASES, ids=_IDS)
def test_r2_the_two_screens_agree_with_the_run_record(run_point, values, default, value, ext):
    derive_applicability = _new("mimarsinan.chip_simulation.cross_sim_parity", "derive_applicability")
    screen_cell_pair = _new("mimarsinan.chip_simulation.cross_sim_parity", "screen_cell_pair")
    parity_error = _new("mimarsinan.chip_simulation.cross_sim_parity", "CrossSimParityError")
    classify = _new("mimarsinan.chip_simulation.parity_contract", "classify_backend_contract")
    refusal = _new("mimarsinan.chip_simulation.point_capability", "backend_refusal_reason")

    cfg, point = run_point(values, default, value, ext)
    for backend, enable in _ENABLE.items():
        runs = cfg[enable]
        applicable, reason = derive_applicability(backend, point)
        assert applicable is runs, backend
        assert (reason is None) is runs, backend
        if backend in ("nevresim", "sanafe"):
            assert (classify(backend=backend, point=point).value == "INAPPLICABLE") is (not runs), backend
        pair_runs = runs and refusal("hcm", point) is None
        if pair_runs:
            with pytest.raises(parity_error, match="must be RUN and measured"):
                screen_cell_pair(cell="c", backend_a="hcm", backend_b=backend, point=point, nf_record=None,
                                 scm_record=None, tolerance=0.0)
        else:
            outcome = screen_cell_pair(cell="c", backend_a="hcm", backend_b=backend, point=point, nf_record=None,
                                       scm_record=None, tolerance=0.0)
            assert outcome.state.value == "inapplicable" and outcome.reason, backend


#: The rows whose run's verdict differs from the declared default's: the other value declared, or an execution
#: extension under the default every executor runs. Where a row read from its mode alone would claim a parity contract
#: the run never measures, or deny one it does.
_ROW_CASES = [c for c in _CASES if (c[2] not in (None, c[1]) and c[3] is None) or (c[1] == FWI and c[3] is not None)]


@pytest.mark.parametrize("values,default,value,ext", _ROW_CASES,
                         ids=[i for c, i in zip(_CASES, _IDS) if c in _ROW_CASES])
def test_r2_the_ledger_contract_metadata_reads_the_row_as_its_run_resolves_it(run_point, values, default, value,
                                                                              ext):
    from mimarsinan.chip_simulation.ledger_schema import normalize_planned_ledger_row
    from mimarsinan.chip_simulation.parity_contract import parity_contract_metadata

    cfg, _point = run_point(values, default, value, ext)
    axes = {"spiking_family": "lif", "spiking_variant": "synchronized", **_declarations(value, ext)}
    for backend in ("nevresim", "sanafe"):
        expected = "BIT_PARITY" if cfg[_ENABLE[backend]] else "INAPPLICABLE"
        assert parity_contract_metadata({**axes, "backend": backend})["backend_contract"] == expected, backend
        row = normalize_planned_ledger_row({"axes": {**axes, "backend": backend}})
        assert row["parity_contracts"]["backend_contract"] == expected, backend


# -- each policy build receives the record its caller holds ----------------------------------------------------------

@pytest.mark.parametrize("values,default,value,ext", _CASES, ids=_IDS)
def test_r2_the_sanafe_soma_name_and_network_build_take_the_run_point(run_point, policy_builds, monkeypatch, values,
                                                                       default, value, ext):
    soma_name = _new("mimarsinan.chip_simulation.sanafe.neuron_model", "soma_hw_name_for_point")
    import mimarsinan.chip_simulation.sanafe.net_synth as net_synth
    from mimarsinan.chip_simulation.sanafe.net_synth.build import build_network_for_segment

    _cfg, point = run_point(values, default, value, ext)
    assert policy_builds == [point], policy_builds  # [composed head, rule 10] the derivation builds once, at the point
    policy_builds.clear()  # [composed head, rule 10] the builds of the site, from its resolved point on
    soma_name(point)
    monkeypatch.setattr(net_synth, "_sanafe", lambda: SimpleNamespace(Network=SimpleNamespace))
    build_network_for_segment(SimpleNamespace(tiles=[]), SimpleNamespace(cores=[]), tile_offset=0, core_offset=0,
                              point=point)
    assert policy_builds == [point, point]


@pytest.mark.parametrize("values,default,value,ext", _CASES, ids=_IDS)
def test_r2_the_calibration_walk_and_soft_core_mapping_take_the_run_point(run_point, policy_builds, monkeypatch,
                                                                         values, default, value, ext):
    calibration_forward_for_point = _new("mimarsinan.mapping.support.bias_compensation",
                                         "calibration_forward_for_point")
    from conftest import MockPipeline
    import mimarsinan.mapping.support.negative_boundary as negative_boundary
    from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import SoftCoreMappingStep

    cfg, point = run_point(values, default, value, ext)
    assert policy_builds == [point], policy_builds  # [composed head, rule 10] the derivation builds once, at the point
    policy_builds.clear()  # [composed head, rule 10] the builds of the site, from its resolved point on
    calibration_forward_for_point(point)
    assert policy_builds == [point]

    monkeypatch.setattr(negative_boundary, "apply_negative_boundary_policy", lambda *a, **k: {})
    step = SoftCoreMappingStep(MockPipeline(config={**cfg, "simulation_steps": 8, "device": "cpu"}))
    step.trainer = SimpleNamespace(iter_validation_batches=lambda n: iter([(torch.zeros(2, 3), torch.zeros(2))]))
    step._apply_negative_boundary_policy(object())
    assert policy_builds == [point, point]


@pytest.mark.parametrize("values,default,value,ext", _CASES, ids=_IDS)
def test_r2_nevresim_code_generation_takes_the_run_point(run_point, policy_builds, tmp_path, values, default, value,
                                                         ext):
    from mimarsinan.code_generation.generate_main import generate_main_function_runtime, get_config, resolve_exec_policy

    _cfg, point = run_point(values, default, value, ext)
    assert policy_builds == [point], policy_builds  # [composed head, rule 10] the derivation builds once, at the point
    policy_builds.clear()  # [composed head, rule 10] the builds of the site, from its resolved point on
    resolve_exec_policy(point=point, firing_mode="Default", thresholding_mode="<=", spike_gen_mode="Uniform",
                        weight_type="int", simulation_length=4, latency=1, output_count=2)
    generate_main_function_runtime(str(tmp_path), 4, 2, 4, 1, verbose=False, simulation_config=get_config(
        "Uniform", "Default", "int", point=point, threshold_type="int"))
    assert policy_builds == [point, point]


@pytest.mark.parametrize("values,default,value,ext", _CASES, ids=_IDS)
def test_r2_the_nevresim_driver_emits_its_run_point(run_point, policy_builds, monkeypatch, tmp_path, values, default,
                                                    value, ext):
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.chip_simulation.nevresim.profiling.synthetic_mapping import build_synthetic_mapping
    from mimarsinan.mapping.latency.chip import ChipLatency

    from mimarsinan.chip_simulation.backend_extensions import BackendExtensionRefusal
    from mimarsinan.chip_simulation.point_capability import BackendWindowFiringRefusal, backend_refusal_reason

    _cfg, point = run_point(values, default, value, ext)
    assert policy_builds == [point], policy_builds  # [composed head, rule 10] the derivation builds once, at the point
    policy_builds.clear()  # [composed head, rule 10] the builds of the site, from its resolved point on
    monkeypatch.setattr(NevresimDriver, "nevresim_path", str(tmp_path))
    hcm, input_size = build_synthetic_mapping("contiguous_input", 2, 4, 2)
    build = dict(spike_generation_mode="Deterministic", thresholding_mode="<=", point=point,
                 connectivity_mode="runtime", verbose=False)
    if backend_refusal_reason("nevresim", point) is not None:  # [U-LIF2 r3, n-5] judged before any work, by name
        with pytest.raises((BackendWindowFiringRefusal, BackendExtensionRefusal),
                           match="NevresimDriver: backend 'nevresim' does not execute"):
            NevresimDriver(input_size, hcm, str(tmp_path), float, **build)
        assert policy_builds == [] and not (tmp_path / "chip.json").exists()
        return
    driver = NevresimDriver(input_size, hcm, str(tmp_path), float, **build)
    assert (driver.spiking_mode, driver.soma_law) == (point.spiking_mode, point.soma_law)
    driver.emit_main(1, 4, ChipLatency(hcm).calculate(), verbose=False)
    assert policy_builds == [point]


@pytest.mark.parametrize("values,default", _SETTINGS)
def test_r2_the_nevresim_profiler_states_the_window_firing_it_profiles(declared, policy_builds, tmp_path, values,
                                                                      default):
    from mimarsinan.chip_simulation.nevresim.profiling.profile_runner import profile_mapping_compile
    from mimarsinan.chip_simulation.nevresim.profiling.synthetic_mapping import build_synthetic_mapping

    declared(values, default)
    hcm, input_size = build_synthetic_mapping("contiguous_input", 2, 4, 2)
    if FWI not in values:
        with pytest.raises(ValueError, match="lif_window_firing='fire_while_integrating'"):
            profile_mapping_compile(hcm, input_size, nevresim_path=str(tmp_path), out_dir=tmp_path, compile=False)
        return
    profile_mapping_compile(hcm, input_size, nevresim_path=str(tmp_path), out_dir=tmp_path, compile=False)
    assert [(p.spiking_mode, p.window_firing, p.execution_extensions) for p in policy_builds] == [("lif", FWI, ())]


@pytest.mark.parametrize("values,default", _SETTINGS)
def test_r2_the_randomized_executors_build_each_semantics_at_its_config_point(declared, policy_builds, values,
                                                                             default):
    from mimarsinan.certification.randomized.executors import RUNS, SemanticsRun
    from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point

    declared(values, default)
    for name, run in RUNS.items():
        if run.name == "value" or not run.flow:
            continue
        if run.config.get(KEY) not in (None, *values):  # [U-LIF2 W14] a declared value the set leaves out
            with pytest.raises(ValueError, match=KEY):
                run.decode  # noqa: B018
            continue
        policy_builds.clear()
        assert run.decode in ("count", "timing")
        assert policy_builds == [resolve_spiking_point(run.config)], name
    for value in values:
        run = SemanticsRun("declared", {**_doc(), KEY: value}, {"spiking_mode": "lif"})
        policy_builds.clear()
        assert run.decode == "count"
        assert [p.window_firing for p in policy_builds] == [value]


@pytest.mark.parametrize("values,default", _SETTINGS)
def test_r2_the_nevresim_chip_probe_states_its_semantics_point(declared, tmp_path, values, default):
    from mimarsinan.certification.randomized.executors import NEVRESIM_SEMANTICS, RUNS, _ChipProbe
    from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point

    declared(values, default)
    for semantics in NEVRESIM_SEMANTICS:
        instance = SimpleNamespace(semantics=semantics, inputs=[[1, 2]], timesteps=4)
        probe = _ChipProbe(SimpleNamespace(output_sources=torch.zeros(2).numpy()), instance, str(tmp_path), "collapse")
        assert probe.spiking_point == resolve_spiking_point(RUNS[semantics].config), semantics


def _flow(**kwargs):
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

    flow = SpikingHybridCoreFlow((4,), SimpleNamespace(stages=[]), 4, thresholding_mode="<=", **kwargs)
    flow._forward_rate = lambda x: x
    return flow


@pytest.mark.parametrize("values,default,value,ext", _CASES, ids=_IDS)
def test_r2_the_hybrid_flow_decodes_at_its_builder_point(run_point, policy_builds, values, default, value, ext):
    from conftest import MockPipeline
    from mimarsinan.pipelining.core.simulation_factory import build_spiking_hybrid_flow

    cfg, point = run_point(values, default, value, ext)
    assert policy_builds == [point], policy_builds  # [composed head, rule 10] the derivation builds once, at the point
    policy_builds.clear()  # [composed head, rule 10] the builds of the site, from its resolved point on
    flow = _flow(spiking_point=point, lif_execution_synchronized=point.integrates_then_emits,
                 phase_dither=bool(cfg.get("spike_phase_dither")), lif_membrane_init=cfg.get("lif_membrane_init", 0.0))
    flow.forward(torch.zeros(1, 4))
    assert policy_builds == [point] and flow.spiking_point() == point
    built = build_spiking_hybrid_flow(MockPipeline(config={**cfg, "input_shape": (4,), "device": "cpu"}),
                                      SimpleNamespace(stages=[]), device="cpu")
    assert built.spiking_point() == point


#: Documents away from the synchronized point: the builder's point names a variant the executor's fields cannot state.
_AWAY = {"streamed LIF": {"spiking_family": "lif", "spiking_variant": "streamed"},
         "cascaded TTFS": {"spiking_family": "ttfs", "spiking_variant": "cascaded"}}


@pytest.mark.parametrize("away", sorted(_AWAY))
@pytest.mark.parametrize("values,default", _SETTINGS)
def test_r2_every_flow_builder_passes_its_contract_point(declared, values, default, away):
    """The deployed-read factory and the two identity executors of the NF/SCM checks carry the contract's own point,
    variant included (a point stated from an executor's fields would read the legacy synchronized variant)."""
    from conftest import MockPipeline, make_tiny_ir_graph
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.pipelining.core import nf_scm_parity
    from mimarsinan.pipelining.core.simulation_factory import build_spiking_hybrid_flow

    declared(values, default)
    cfg = derive_run_config({"model_type": "simple_mlp", **_AWAY[away]}, _CORES)
    point = SpikingDeploymentContract.from_pipeline_config(cfg).spiking_point()
    pipeline = MockPipeline(config={**cfg, "input_shape": (8,), "device": "cpu"})
    flows = [build_spiking_hybrid_flow(pipeline, SimpleNamespace(stages=[]), device="cpu")]
    builder = (nf_scm_parity._build_streamed_identity_executor if away == "streamed LIF"
               else nf_scm_parity._build_cascaded_identity_executor)
    flows.append(builder(pipeline, SimpleNamespace(preprocessor=None), make_tiny_ir_graph()))
    assert [flow.spiking_point() for flow in flows] == [point, point]
    assert point.spiking_variant == _AWAY[away]["spiking_variant"]


@pytest.mark.parametrize("values,default,value,ext", _CASES, ids=_IDS)
def test_r2_the_nevresim_runner_holds_the_plan_point(run_point, monkeypatch, tmp_path, values, default, value, ext):
    import numpy as np
    from mimarsinan.chip_simulation.simulation_runner import core as runner_core

    cfg, point = run_point(values, default, value, ext)
    factory = SimpleNamespace(create_data_provider=lambda: SimpleNamespace(get_test_batch_size=lambda: 1),
                              create_test_loader=lambda batch_size, provider: [])
    monkeypatch.setattr(runner_core, "DataLoaderFactory", SimpleNamespace(for_pipeline=lambda pipeline: factory))
    monkeypatch.setattr(runner_core, "shutdown_data_loader", lambda loader: None)
    monkeypatch.setattr(runner_core, "load_verified_test_data",
                        lambda runner, loader, source: [(np.zeros(4), np.zeros(1))])
    pipeline = SimpleNamespace(config={**cfg, "input_size": 4, "num_classes": 2, "device": "cpu",
                                       "spike_generation_mode": "Uniform"}, working_directory=str(tmp_path))
    runner = runner_core.SimulationRunner(pipeline, None, 4)
    assert runner.spiking_point == point


def test_r2_a_flow_refuses_a_point_its_own_fields_do_not_execute(declared):
    refusal = _new("mimarsinan.models.spiking.hybrid.flow", "FlowPointRefusal")
    resolve = _new("mimarsinan.chip_simulation.spiking_point", "resolve_spiking_point")
    declared((FWI, ITE), FWI)
    with pytest.raises(refusal, match="lif_window_firing='integrate_then_emit'"):
        _flow(spiking_point=resolve(_doc(lif_window_firing=ITE)))
    with pytest.raises(refusal, match="spiking_mode"):
        _flow(spiking_point=resolve({"spiking_family": "ttfs", "spiking_variant": "analytical"}))
    with pytest.raises(refusal, match="spike_phase_dither"):
        _flow(spiking_point=resolve(_doc(spike_phase_dither=True)))


@pytest.mark.parametrize("value,ext", _BOTH_VALUES)
def test_r2_the_identity_flow_carries_its_builder_point(run_point, value, ext):
    from conftest import make_tiny_ir_graph
    from mimarsinan.models.spiking.hybrid.identity_flow import build_identity_spiking_flow

    cfg, point = run_point((FWI, ITE), FWI, value, ext)
    flow = build_identity_spiking_flow((8,), make_tiny_ir_graph(), 4, None, "Default", "Uniform", thresholding_mode="<=",
                                       phase_dither=bool(cfg.get("spike_phase_dither")),
                                       lif_membrane_init=float(cfg.get("lif_membrane_init", 0.0)), spiking_point=point)
    assert flow.spiking_point() == point
    assert flow.lif_execution_synchronized is point.integrates_then_emits


@pytest.mark.parametrize("values,default", _SETTINGS)
def test_r2_the_calibration_pipeline_builds_at_the_config_point(declared, policy_builds, values, default):
    from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point
    from mimarsinan.tuning.orchestration.calibration_pipeline import CalibrationPipeline

    declared(values, default)
    configs = [{"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": schedule}
               for schedule in ("cascaded", "synchronized")]
    configs += [_doc(**_declarations(value, ext)) for value in values for ext in (None, *_EXTENSIONS)]
    for config in configs:
        policy_builds.clear()
        CalibrationPipeline.resolve(config, synchronized=False, distmatch_driven=False)
        assert policy_builds == [resolve_spiking_point(config)], config


# -- a holder without the run's point states its window firing explicitly, never through the declared default --------

@pytest.mark.parametrize("values,default,value,ext", [c for c in _CASES if c[3] is not None],
                         ids=[i for c, i in zip(_CASES, _IDS) if c[3] is not None])
def test_r2_m3_a_contract_without_a_resolved_point_states_its_fields_and_the_historical_firing(
        run_point, values, default, value, ext):
    from mimarsinan.chip_simulation.backend_extensions import BackendExtensionRefusal
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.chip_simulation.odin_fpga.runner import OdinFpgaRunner
    from mimarsinan.chip_simulation.sanafe.runner.core import SanafeRunner

    cfg, point = run_point(values, default, value, ext)
    contract = SpikingDeploymentContract.from_pipeline_config(cfg)
    stored = dataclasses.replace(contract, behavior=dataclasses.replace(contract.behavior, point=None))
    stated = stored.spiking_point()
    assert stated.execution_extensions == (ext[0],)
    assert stated.window_firing == FWI
    assert dataclasses.replace(stated, window_firing=point.window_firing) == point
    with pytest.raises(BackendExtensionRefusal, match=rf"SanafeRunner: backend 'sanafe' does not execute the "
                                                      rf"declared execution extension\(s\) \['{ext[0]}'\]"):
        SanafeRunner(None, 4, contract=stored)
    with pytest.raises(BackendExtensionRefusal, match=rf"OdinFpgaRunner: backend 'odin_fpga' does not execute the "
                                                      rf"declared execution extension\(s\) \['{ext[0]}'\]"):
        OdinFpgaRunner(None, 4, contract=stored, transport=None, weight_bits=8, effective_max_axons=256,
                       max_neurons=256, pass_transfer="collapse")
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver  # [U-LIF2 r3, n-5]

    with pytest.raises(BackendExtensionRefusal, match=rf"NevresimDriver: backend 'nevresim' does not execute the "
                                                      rf"declared execution extension\(s\) \['{ext[0]}'\]"):
        NevresimDriver(1, None, None, None, spike_generation_mode="Uniform", thresholding_mode="<=", point=stated)


@pytest.mark.parametrize("values,default", _SETTINGS)
def test_r2_a_behavior_and_a_flow_built_from_fields_state_their_own_window_firing(declared, policy_builds, values,
                                                                                 default):
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig

    declared(values, default)
    behavior = NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<=",
                                    spike_generation_mode="Uniform")
    assert behavior.spiking_point().window_firing == FWI
    assert behavior.spiking_point().execution_extensions == ()
    for flag, firing in ((False, FWI), (True, ITE)):
        for key, extension in ((None, None), *_EXTENSIONS):
            policy_builds.clear()
            kwargs = {"lif_execution_synchronized": flag}
            if key == "spike_phase_dither":
                kwargs["phase_dither"] = True
            elif key == "lif_membrane_init":
                kwargs["lif_membrane_init"] = extension
            _flow(**kwargs).forward(torch.zeros(1, 4))
            (built,) = policy_builds
            assert built.window_firing == firing and built.execution_extensions == ((key,) if key else ())


def test_r2_a_backend_question_with_a_mode_alone_is_refused_by_name():
    refusal = _new("mimarsinan.chip_simulation.spiking_point", "UnresolvedPointRefusal")
    from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
    from mimarsinan.chip_simulation.point_capability import backend_refusal_reason

    with pytest.raises(refusal, match="never a mode alone"):
        BACKEND_REGISTRY.get("sanafe").supports("lif")
    with pytest.raises(refusal, match="never a mode alone"):
        BACKEND_REGISTRY.get("loihi").require_supported("lif", context="ctx")
    with pytest.raises(refusal, match="never a mode alone"):
        backend_refusal_reason("sanafe", "lif")  # type: ignore[arg-type]


def test_r2_a_stated_point_never_reads_a_document_window_firing():
    stated_point = _new("mimarsinan.chip_simulation.spiking_point", "stated_point")
    historical = _new("mimarsinan.chip_simulation.spiking_point", "HISTORICAL_WINDOW_FIRING")
    refusal = _new("mimarsinan.chip_simulation.window_firing", "WindowFiringRefusal")
    assert historical == FWI
    with pytest.raises(refusal, match="is resolved by the run, never stated"):
        stated_point(_doc(lif_window_firing=FWI), window_firing=FWI)
    with pytest.raises(refusal, match="is resolved by the run, never stated"):
        stated_point(_doc(lif_execution_discipline="synchronized"), window_firing=FWI)
    # The holder states what it executes, never judged here: an executor that cannot run it refuses by its own name.
    assert stated_point(_doc(firing_mode="Novena"), window_firing=ITE).window_firing == ITE
    assert stated_point({"spiking_family": "lif", "spiking_variant": "streamed"}, window_firing=FWI).window_firing is None


# -- the ratchet t02c: the capability judgments take a point, no mode-only builder lives in src ----------------------

#: The holders that carry no run's point and state their own (each with its reason): the only callers of stated_point.
_STATED_POINT_HOLDERS = {
    "chip_simulation/deployment_contract.py": "a contract whose behavior carries no resolved point",
    "chip_simulation/behavior_config.py": "a behavior built from its own fields or stored before the point existed",
    "models/spiking/hybrid/flow.py": "an executor built from its own fields with no builder's point",
    # [U-LIF2 r3] the stored run's own point
    "chip_simulation/stored_point.py": "a stored run that persisted no point (every run made before runs persisted it)",
}
#: Each capability judgment and policy build with where its point argument stands (a position or a keyword).
_POINT_TAKERS = {
    "backend_refusal_reason": (1, "point"), "require_backend_executes": (0, "point"), "sim_enable_refusals": (0, "point"),
    "derive_applicability": (1, "point"), "classify_backend_contract": (None, "point"),
    "screen_cell_pair": (None, "point"), "soma_hw_name_for_point": (0, "point"),
    "calibration_forward_for_point": (0, "point"), "build_network_for_segment": (None, "point"),
    "resolve_exec_policy": (None, "point"), "policy_for_spiking_mode": (0, "point"),
}


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        yield path.relative_to(SRC).as_posix(), ast.parse(path.read_text())


def _call_name(node: ast.Call):
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _calls():
    """Each call in src with its module, its name, and the scopes that hold it (the module, then each enclosing
    function), so a name it passes can be read where it is bound."""
    for rel, tree in _modules():
        stack = [(tree, (tree,))]
        while stack:
            node, scopes = stack.pop()
            for child in ast.iter_child_nodes(node):
                inner = (*scopes, child) if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scopes
                if isinstance(child, ast.Call):
                    yield rel, _call_name(child), child, inner
                stack.append((child, inner))


def _point_argument(node: ast.Call, position, keyword):
    for kw in node.keywords:
        if kw.arg == keyword or kw.arg is None:
            return kw.value
    if position is not None and len(node.args) > position:
        return node.args[position]
    return None


def _a_literal_document(node) -> bool:
    if isinstance(node, ast.Dict):
        return all(key is not None for key in node.keys)  # no ** spread of a held record
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args


def _bound_value(name: str, scopes):
    """[U-LIF2 r3, n-6] The value a name is bound to by a plain assignment in the innermost scope that binds it (the
    enclosing functions, then the module), or None."""
    for scope in reversed(scopes):
        body = scope.body if isinstance(scope, ast.Module) else list(ast.walk(scope))
        for statement in body:
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                         for t in statement.targets):
                return statement.value
            if (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
                    and statement.target.id == name):
                return statement.value
    return None


def _declares_the_window_firing(document: ast.AST) -> bool:
    keys = document.keys if isinstance(document, ast.Dict) else [ast.Constant(kw.arg) for kw in document.keywords]
    return any((isinstance(k, ast.Constant) and k.value == KEY) or (isinstance(k, ast.Name) and k.id == "WINDOW_FIRING_KEY")
               for k in keys)


def _a_mode_only_literal(node, scopes) -> bool:
    """[U-LIF2 r3, n-6] A literal document handed to the resolution, written at the call or bound to a name first,
    that does not declare the window firing (one that declares it, as the profiler's document does, states its point)."""
    if isinstance(node, ast.Name):
        node = _bound_value(node.id, scopes)
    return node is not None and _a_literal_document(node) and not _declares_the_window_firing(node)


def test_r2_t02c_no_mode_only_point_builder_and_every_judgment_takes_a_point():
    calls = list(_calls())
    mode_only = sorted(f"{rel}:{node.lineno}" for rel, name, node, _scopes in calls if name == "point_of_mode")
    defined = sorted(path.relative_to(SRC).as_posix() for path in SRC.rglob("*.py")
                     if "def point_of_mode" in path.read_text())
    assert mode_only == [] and defined == []
    holders = sorted({rel for rel, name, _node, _scopes in calls if name == "stated_point"})
    assert holders == sorted(_STATED_POINT_HOLDERS)
    bad = []
    for rel, name, node, scopes in calls:
        if name in _POINT_TAKERS:
            argument = _point_argument(node, *_POINT_TAKERS[name])
            if argument is None or isinstance(argument, ast.Constant) or _a_literal_document(argument):
                bad.append(f"{rel}:{node.lineno} {name}")
        if name == "resolve_spiking_point" and node.args and _a_mode_only_literal(node.args[0], scopes):
            bad.append(f"{rel}:{node.lineno} resolve_spiking_point of a literal document")
    assert bad == []
    # [U-LIF2 r3, n-6] the rule reaches a literal bound to a name, and admits one that declares the window firing.
    probe = ast.parse("SILENT = {'spiking_mode': 'lif'}\nSTATED = {'spiking_mode': 'lif', WINDOW_FIRING_KEY: 'x'}\n"
                      "def f():\n    local = dict(spiking_mode='lif')\n    return local\n")
    function = probe.body[2]
    assert _a_mode_only_literal(ast.Name("SILENT"), (probe,)) and not _a_mode_only_literal(ast.Name("STATED"), (probe,))
    assert _a_mode_only_literal(ast.Name("local"), (probe, function))


#: [U-LIF2 r3] What reads a run directory's document: the run's configuration folder named, or one of its readers called.
_RUN_DOCUMENT_READERS = ("read_run_document", "load_saved_config_from_run_dir", "get_run_config")
_RUN_DIRECTORY_NAMES = ("RUN_CONFIG_DIRECTORY", "RUN_DOCUMENT_FILENAME")
#: What resolves a plan, a point or a derivation from a document (a policy built from a point is t02c's above).
_RESOLUTIONS = ("resolve_spiking_point", "fold_spiking_point", "resolve_window_firing", "run_config", "derive_run_config",
                "derive_deployment_parameters", "derive_pipeline_runtime_parameters", "build_flat_pipeline_config",
                "build_config_display_view", "mode_policy", "require_spiking_point", "from_pipeline_config",
                "from_deployment_config")
_RESOLVING_CLASSES = {"DeploymentPlan": ("resolve", "of"), "SomaLaw": ("resolve",), "ConversionPolicy": ("derive",)}
#: The readers' named path: the functions that read a run directory's document and derive from it, each taking the
#: point its run resolved (``stored_run_point``), with the reason.
_STORED_RUN_READERS = {
    "config_schema/display_view.py::build_pipeline_config_view": "a run directory's view, derived at its stored point",
}
#: The scope that derives a saved document at its stored point is entered only on the readers' path.
_STORED_POINT_SCOPES = {"stored_window_firing": ("chip_simulation/stored_point.py",),
                        "derived_at_stored_point": ("chip_simulation/stored_point.py", "config_schema/display_view.py"),
                        # [U-LIF2 r4] the writer: a run takes its directory's point, entered only by the session
                        "run_at_directory_point": ("pipelining/session.py",)}


def _resolves(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.attr in _RESOLVING_CLASSES.get(func.value.id, ()):
            return True
    return _call_name(node) in _RESOLUTIONS


def _own_nodes(function):
    """The nodes of one function, without the bodies of the functions and classes defined inside it (each its own)."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def _reads_a_run_document(function) -> bool:
    docstring = ast.get_docstring(function, clean=False)
    for node in _own_nodes(function):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "_RUN_CONFIG" in node.value:
            if node.value != docstring:
                return True
        if isinstance(node, ast.Name) and node.id in _RUN_DIRECTORY_NAMES:
            return True
        if isinstance(node, ast.Attribute) and node.attr in _RUN_DIRECTORY_NAMES:
            return True
        if isinstance(node, ast.Call) and _call_name(node) in _RUN_DOCUMENT_READERS:
            return True
    return False


def _functions():
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield f"{rel}::{node.name}", node


def test_r3_t02c_no_plan_is_resolved_from_a_run_directory_document_outside_the_readers_path():
    """A reader of a stored run takes the point the run resolved: a function that reads a run directory's document and
    resolves a plan, a point or a derivation from it (directly, or through a function of src that does) is refused,
    unless it is on the readers' named path, and each function there takes the stored point."""
    functions = list(_functions())
    named_path = {name.split("::")[1] for name in _STORED_RUN_READERS}
    resolvers = {node.name for _name, node in functions if node.name not in named_path
                 and any(isinstance(call, ast.Call) and _resolves(call) for call in _own_nodes(node))}
    refused, unstated = [], []
    for name, node in functions:
        if not _reads_a_run_document(node):
            continue
        calls = [call for call in _own_nodes(node) if isinstance(call, ast.Call)]
        if name in _STORED_RUN_READERS:
            if not any(_call_name(call) == "stored_run_point" for call in calls):
                unstated.append(name)
        elif any(_resolves(call) or _call_name(call) in resolvers for call in calls):
            refused.append(name)
    assert refused == [] and unstated == [], (refused, unstated)
    scopes = {}
    for rel, name, _node, _scopes in _calls():
        if name in _STORED_POINT_SCOPES:
            scopes.setdefault(name, set()).add(rel)
    assert {name: tuple(sorted(rels)) for name, rels in scopes.items()} == _STORED_POINT_SCOPES


#: [U-LIF2 r4] A run directory holds one point: the functions that write a run's point, with the reason.
_RUN_POINT_WRITERS = {
    "pipelining/session.py::__init__": "the session, once its pipeline is built at its directory's point",
    "gui/runtime/run_cache_seed.py::_carry_run_point": "the seed: entries copied from a run bring that run's point",
}
#: Where a run directory's own point is read: the session's door and the seed.
_DIRECTORY_POINT_READERS = ("gui/runtime/run_cache_seed.py", "pipelining/session.py")
#: The files a run directory's entries live in, as a function that copies them names them.
_ENTRY_FILE_NAMES = ("metadata.json", "steps.json")
#: The functions that copy a file of one of those names that is no run's entry, with the reason (a finite list).
_NOT_A_RUN_ENTRY = {
    "chip_simulation/nevresim/compile_cache.py::store_binary": "a compiled simulator's own metadata in its cache entry",
}
_COPIES = ("copy", "copy2", "copyfile", "copytree")


def _scope_of(node, name: str) -> bool:
    return isinstance(node, (ast.With, ast.AsyncWith)) and any(
        isinstance(item.context_expr, ast.Call) and _call_name(item.context_expr) == name for item in node.items)


def _own_calls(function):
    return [node for node in _own_nodes(function) if isinstance(node, ast.Call)]


def test_r4_t02c_a_run_point_is_written_only_at_the_directory_point_by_the_session_and_by_the_seed():
    """[U-LIF2 r4] ``persist_run_point`` is called only by the session, after its pipeline was built inside the scope of
    its directory's point (``run_at_directory_point``), and by the seed's carry; the session module resolves nothing
    outside that scope; every src function that copies a run's entry file reads the source's point before its first
    copy and carries it after its last (U-LIF2 r5, n-1 (b): a refusal leaves the new run without entries); the
    directory's point is read only by the session and the seed; no other src module builds a pipeline."""
    functions = dict(_functions())
    writers = sorted(name for name, node in functions.items()
                     if any(_call_name(call) == "persist_run_point" for call in _own_calls(node)))
    assert writers == sorted(_RUN_POINT_WRITERS)
    session = functions["pipelining/session.py::__init__"]
    scopes = [node for node in _own_nodes(session) if _scope_of(node, "run_at_directory_point")]
    assert len(scopes) == 1 and any(isinstance(node, ast.Call) and _call_name(node) == "DeploymentPipeline"
                                    for node in ast.walk(scopes[0]))
    assert all(call.lineno > scopes[0].end_lineno for call in _own_calls(session)
               if _call_name(call) == "persist_run_point")
    module = ast.parse((SRC / "pipelining" / "session.py").read_text())
    inside = {id(node) for scope in ast.walk(module) if _scope_of(scope, "run_at_directory_point")
              for node in ast.walk(scope)}
    outside = [f"session.py:{call.lineno} {_call_name(call)}" for call in ast.walk(module)
               if isinstance(call, ast.Call) and id(call) not in inside
               and (_resolves(call) or _call_name(call) in ("document_declaration_errors", "DeploymentPipeline"))]
    assert outside == []
    copying = sorted(name for name, node in functions.items() if name not in _NOT_A_RUN_ENTRY
                     and any(_call_name(call) in _COPIES for call in _own_calls(node))
                     and any(isinstance(leaf, ast.Constant) and leaf.value in _ENTRY_FILE_NAMES
                             for leaf in _own_nodes(node)))
    assert copying
    for name in copying:  # [U-LIF2 r5] the source's point read before the first copy, carried after the last
        lines = {}
        for call in _own_calls(functions[name]):
            lines.setdefault(_call_name(call), []).append(call.lineno)
        copies = [line for copy in _COPIES for line in lines.get(copy, ())]
        reads, carries = lines.get("run_directory_point", []), lines.get("_carry_run_point", [])
        assert reads and carries and max(reads) < min(copies) and min(carries) > max(copies), (name, lines)
    readers = sorted({rel for rel, name, _node, _scopes in _calls() if name == "run_directory_point"}
                     - {"chip_simulation/stored_point.py"})
    assert readers == sorted(_DIRECTORY_POINT_READERS)
    builders = sorted({rel for rel, name, _node, _scopes in _calls() if name == "DeploymentPipeline"})
    assert builders == ["pipelining/session.py"], builders  # no pipeline is built outside the directory's point


def test_r2_t02c_every_point_parameter_is_required():
    modules = ("mimarsinan.chip_simulation.point_capability", "mimarsinan.chip_simulation.cross_sim_parity",
               "mimarsinan.chip_simulation.parity_contract", "mimarsinan.chip_simulation.sanafe.neuron_model",
               "mimarsinan.mapping.support.bias_compensation", "mimarsinan.chip_simulation.sanafe.net_synth.build",
               "mimarsinan.code_generation.generate_main", "mimarsinan.chip_simulation.spiking_mode_policy")
    found = {}
    for module in modules:
        for name, function in inspect.getmembers(importlib.import_module(module), inspect.isfunction):
            if name in _POINT_TAKERS and function.__module__ == module:
                found[name] = inspect.signature(function).parameters["point"].default is inspect.Parameter.empty
    assert found == {name: True for name in _POINT_TAKERS}


# -- m-1: a soma value its own key refuses never reaches the joint table ----------------------------------------------

@pytest.mark.parametrize("values,default", _SETTINGS)
def test_r2_m1_a_per_event_document_names_its_own_key_at_both_doors(declared, values, default):
    from mimarsinan.config_schema.runtime import run_derivation_errors

    declared(values, default)
    for value in (None, *values):
        dp = _doc(firing_granularity="per_event", **_declarations(value, None))
        with pytest.raises(ValueError) as refused:
            derive_run_config(dp, _CORES)
        assert "firing_granularity" in str(refused.value) and KEY not in str(refused.value), value
        errors = run_derivation_errors({"deployment_parameters": dp, "platform_constraints": _CORES})
        assert errors and all("firing_granularity" in e and KEY not in e for e in errors), (value, errors)


# -- n-1: every soma key's illegal value is that key's own refusal, at both doors and the runtime derivation ---------

def _names_only(message: str, key: str) -> bool:
    return key in message and KEY not in message


@pytest.mark.parametrize("values,default", _SETTINGS)
def test_r3_n1_every_illegal_soma_value_names_its_own_key_at_both_doors_and_the_runtime_derivation(declared, values,
                                                                                                   default):
    """[U-LIF2 r3, n-1] The joint table abstains wherever any soma key's value lies outside that key's own legal set,
    so each such document is refused by its own key under every declared set, silent and declaring each value."""
    table = _new("mimarsinan.chip_simulation.soma_axes", "SOMA_LEGAL_VALUES")
    from mimarsinan.config_schema.deployment_derivation import derive_pipeline_runtime_parameters
    from mimarsinan.config_schema.registry import REGISTRY
    from mimarsinan.config_schema.runtime import run_derivation_errors

    point_keys = ("spiking_family", "spiking_variant", KEY)
    assert set(table) == {key for key, entry in REGISTRY.items() if entry.group == "spiking" and entry.domain == "event"
                          and entry.legal_values is not None and key not in point_keys}
    assert all(REGISTRY[key].legal_values is table[key] for key in table)  # one truth: the registry reads the table
    declared(values, default)
    cases = [(key, value) for key in sorted(table) for value in REGISTRY[key].resolved_options() or ()
             if value not in table[key](_doc())]
    assert cases
    for key, value in cases:
        for declared_value in (None, *values):
            dp = _doc(**{key: value}, **_declarations(declared_value, None))
            with pytest.raises(ValueError) as refused:
                derive_run_config(dp, _CORES)
            assert _names_only(str(refused.value), key), (key, value, declared_value, str(refused.value))
            errors = run_derivation_errors({"deployment_parameters": dp, "platform_constraints": _CORES})
            assert errors and all(_names_only(e, key) for e in errors), (key, value, declared_value, errors)
            with pytest.raises(ValueError) as refused:
                derive_pipeline_runtime_parameters({"spiking_mode": "lif", key: value,
                                                    **_declarations(declared_value, None)})
            assert _names_only(str(refused.value), key), (key, value, declared_value, str(refused.value))
