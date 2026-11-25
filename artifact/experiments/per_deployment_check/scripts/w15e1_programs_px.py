""
from __future__ import annotations

import contextlib
import glob
import io
import json
import math
import os
import pickle
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __import__("mimarsinan").__file__))))
sys.path.insert(0, os.path.join(TREE, "tests"))

import numpy as np
import torch

torch.set_num_threads(1)

from mimarsinan.certification.program_check import check_deployment, declaration_from_config
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir.core_law import graph_carries_laws

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_rule as W

COLD = "/archive/cold/runs/mimarsinan"


def _row_of(report) -> dict:
    return {"semantics": report.semantics, "theorem1": report.theorem1, "proposition2": report.proposition2,
            "failures": report.failures_by_condition(),
            "statuses": {c: r.status for c, r in report.results.items()},
            "violations": {c: [v.text() for v in r.violations[:3]] for c, r in report.results.items() if r.violations},
            "unknown_forms": [v.text() for v in report.unknown_forms],
            "check_s": round(report.wall_s, 4), "counts": report.counts, "reference": report.reference,
            "ir_tiling": report.results["ir-tiling"].status,
            "ir_tiling_rules": sorted({v.rule for v in report.results["ir-tiling"].violations}),
            "ir_tiling_violations": report.results["ir-tiling"].violation_count}


def _resolve(cfg: dict) -> dict:
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline, merge_pipeline_config
    from mimarsinan.pipelining.session import _resolve_platform_constraints
    shapes = {"MNIST_DataProvider": ((1, 28, 28), 10), "CIFAR10_DataProvider": ((3, 32, 32), 10),
              "ImageNet_DataProvider": ((3, 224, 224), 1000)}
    dp = dict(cfg["deployment_parameters"])
    dp.setdefault("seed", int(cfg.get("seed", 0)))
    DeploymentPipeline.apply_preset(cfg.get("pipeline_mode", "phased"), dp)
    pc = _resolve_platform_constraints(cfg, dp)
    config = merge_pipeline_config(dp, pc if isinstance(pc, dict) else {})
    shape, classes = shapes[cfg["data_provider_name"]]
    config.update(input_shape=shape, num_classes=classes, input_size=int(np.prod(shape)), device="cpu")
    return config


def _random_init_ir(config: dict):
    from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.latency.ir import IRLatency
    from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
    from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
    from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.transformations.quantization_bounds import quantization_bounds

    plan = DeploymentPlan.resolve(config)
    pcfg = build_platform_constraints_resolved(config)
    pk = packaging_contract_for(plan)
    placement = str(config.get("encoding_layer_placement", "subsume"))
    shape, classes = tuple(config["input_shape"]), int(config["num_classes"])
    builder = BUILDERS_REGISTRY[config["model_type"]]("cpu", shape, classes, config)
    torch.manual_seed(0)
    model = build_model(builder, config.get("model_config", {}), encoding_placement=placement, packaging=pk)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *shape))
    if hasattr(model, "get_mapper_repr"):
        flow = model
    else:
        torch.manual_seed(1)
        flow = convert_torch_model(model, shape, classes, packaging=pk, encoding_layer_placement=placement)
    if os.environ.get("W15_HOST_MODE", "eval") == "eval":
        flow.eval()
    rep = flow.get_mapper_repr()
    if hasattr(rep, "assign_perceptron_indices"):
        rep.assign_perceptron_indices()
    params = resolve_platform_mapping_params(pcfg["cores"], allow_coalescing=bool(pcfg.get("allow_coalescing", False)))
    bits = int(config.get("weight_bits", 8))

    def ir_at(unbounded: bool):
        ""
        from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract

        ir = IRMapping(q_max=float(quantization_bounds(bits)[1]), firing_mode=str(config["firing_mode"]),
                       max_axons=None if unbounded else params.effective_max_axons,
                       max_neurons=None if unbounded else params.effective_max_neurons,
                       allow_coalescing=False if unbounded else params.allow_coalescing,
                       hardware_bias=params.hardware_bias).map(rep)
        if W.cascade_of(config):
            W.through_window(ir, config, thresholding_mode=SpikingDeploymentContract.from_pipeline_config(
                config).thresholding_mode, q_max=int(quantization_bounds(bits)[1]))
        if plan.weight_quantization:
            for node in ir.get_neural_cores():
                node.parameter_scale = torch.tensor(0.0)
            for bank in ir.weight_banks.values():
                bank.parameter_scale = torch.tensor(0.0)
        quantize_ir_graph(ir, bits, weight_quantization=plan.weight_quantization)
        IRLatency(ir).calculate()
        return ir

    return ir_at(False), pcfg, ir_at


class _InertGenerator:
    ""

    def __init__(self, *args, **kwargs):
        self.archived = args

    def __setstate__(self, state):
        self.archived_state = state


class _CpuUnpickler(pickle.Unpickler):
    ""

    def find_class(self, module, name):
        import torch._utils as tu
        if (module, name) == ("torch._C", "Generator"):
            return _InertGenerator
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        if module == "torch._utils" and name in ("_rebuild_device_tensor_from_cpu_tensor",
                                                 "_rebuild_device_tensor_from_numpy"):
            original = getattr(tu, name)
            return lambda data, dtype, device, requires_grad: original(data, dtype, "cpu", requires_grad)
        try:
            return super().find_class(module, name)
        except AttributeError:
            if module.startswith("mimarsinan.mapping.ir"):
                return super().find_class("mimarsinan.mapping.ir", name)
            raise


def _load_cpu(path: str):
    with open(path, "rb") as handle:
        return _CpuUnpickler(handle).load()


def _backends(config: dict) -> dict:
    ""
    from types import SimpleNamespace

    from mimarsinan.pipelining.core.gates.static_program_gate import backend_accumulation
    return backend_accumulation(SimpleNamespace(config=config))


def _enter_ir_tiling(report, ir, ir_at, decl, row) -> None:
    ""
    import gc

    from mimarsinan.certification.program_check.report import ConditionResult, ESTABLISHED, FAILED
    from mimarsinan.certification.program_check.unbounded import check_ir_tiling

    gc.collect()
    t0 = time.perf_counter()
    unbounded = ir_at(True)
    row["unbounded_ir_s"] = round(time.perf_counter() - t0, 3)
    result = ConditionResult("ir-tiling", ESTABLISHED)
    t1 = time.perf_counter()
    check_ir_tiling(ir, unbounded, declaration=decl, result=result, identity_facts=report.identity_facts)
    row["ir_tiling_s"] = round(time.perf_counter() - t1, 4)
    if result.violation_count:
        result.status = FAILED
    report.results["ir-tiling"] = result
    row.update(_tiling_facts(ir, unbounded))
    if os.environ.get("W15_SKIP_UNBOUNDED_IDENTITY") != "1":
        from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
        t2 = time.perf_counter()
        theorem = build_identity_hybrid_mapping(ir_graph=unbounded, retimed_level_stages=decl.windowed)
        row["unbounded_timing_derivation"] = _derivation(unbounded, theorem)
        row["unbounded_identity_s"] = round(time.perf_counter() - t2, 3)


def _derivation(graph, identity) -> str:
    ""
    from mimarsinan.certification.program_check.unbounded.latency import facts_of, identity_timing

    latency, stage = identity_timing(graph)
    facts = facts_of(identity, timed=True)
    differ = [n for n in set(latency) | set(facts.latency or {})
              if (latency.get(n), stage.get(n)) != ((facts.latency or {}).get(n), facts.stage.get(n))]
    return "equal" if not differ else f"DIFFERS on {len(differ)} node(s)"


def _tiling_facts(ir, unbounded) -> dict:
    cores = lambda g: sum(1 for n in g.nodes if type(n).__name__ == "NeuralCore")
    tiled = any(getattr(n, "perceptron_output_slice", None) and n.perceptron_output_slice[0] > 0
                for n in ir.get_neural_cores())
    return {"ir_cores": cores(ir), "unbounded_ir_cores": cores(unbounded), "ir_tiled": bool(tiled)}


def _check_config(path: str, ir_source: str) -> dict:
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

    name = os.path.splitext(os.path.basename(path))[0]
    row = {"config": name, "ir_source": ir_source}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            config = _resolve(json.load(open(path)))
            t0 = time.perf_counter()
            ir_at = None
            if ir_source == "random_init":
                ir, pcfg, ir_at = _random_init_ir(config)
            else:
                runs = sorted(glob.glob(f"{COLD}/{name}*_deployment_run/Soft Core Mapping.ir_graph.pickle"),
                              key=os.path.getmtime, reverse=True)
                if not runs:
                    row["skipped"] = "no archived run of this config"
                    return row
                row["ir_pickle"] = runs[0]
                ir = _load_cpu(runs[0])
                pcfg = build_platform_constraints_resolved(config)
            row["ir_s"] = round(time.perf_counter() - t0, 3)
            t1 = time.perf_counter()
            program = build_hybrid_mapping_for_pipeline(ir, pcfg, pipeline_config=config)
            row["compile_s"] = round(time.perf_counter() - t1, 3)
            decl = declaration_from_config(config, cores=pcfg["cores"], has_laws=graph_carries_laws(ir),
                                           backends=_backends(config))
            report = check_deployment(program, ir_graph=ir, declaration=decl, compile_wall_s=row["compile_s"])
            row["passes"] = sum(1 for s in program.stages if s.kind == "neural")
            row["w15_check_s"] = round(report.wall_s, 4)
            del program
            if ir_at is not None:
                _enter_ir_tiling(report, ir, ir_at, decl, row)
        row.update(_row_of(report))
    except ToolchainRefusal as exc:
        row["refused"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        row["trace"] = traceback.format_exc()[-1500:]
    return row


SYNC_TTFS = dict(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized", firing_mode="TTFS",
                 spike_mode="TTFS", pass_transfer="collapse")


def _fuzz(out, s0: int, s1: int, counts) -> None:
    from mimarsinan.certification.program_check import CheckDeclaration
    from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.latency.ir import IRLatency
    from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
    from unit.mapping.residency.executed import runs as R
    from unit.mapping.residency.executed import vehicles as V

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import window_rule as W

    runs = dict(zip(R.RUN_IDS, R.ALL_RUNS))
    runs["synchronized_ttfs"] = SYNC_TTFS
    semantics = {"streamed_lif": "streamed_lif", "windowed_lif": "windowed_lif", "synchronized_lif":
                 "synchronized_lif", "cascaded_ttfs": "cascaded_ttfs", "synchronized_ttfs": "synchronized_ttfs"}
    family = os.environ.get("W15_FAMILY") or "base"
    caps = dict(allow_coalescing=True, allow_neuron_splitting=True) if family == "coalesce" else {}
    coalescing = family == "coalesce"


    window = {name: W.cascade_of(runs[name]) for name in semantics}


    first_of = {}
    for name in semantics:
        first_of.setdefault(window[name], name)


    arming = {W.armed(runs[name]) for name in semantics if window[name]}
    if len(arming) > 1:
        raise SystemExit(f"REFUSED: WINDOW_FORMS: the semantics that read the window are armed unlike ({arming})")

    def through_window(ir, form):
        name = first_of[form]
        return W.through_window(ir, runs[name], thresholding_mode="<=", q_max=127)

    def rep_of(seed):
        return V.wide_rep(seed) if family == "coalesce" else V.random_rep(seed)

    def integer_ir(rep, form, unbounded=False):
        ""
        ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=None if unbounded else V.MAX_AXONS,
                       max_neurons=None if unbounded else V.MAX_NEURONS,
                       allow_coalescing=False if unbounded else coalescing).map(rep)
        through_window(ir, form)
        for node in ir.get_neural_cores():
            node.parameter_scale = torch.tensor(0.0)
        for bank in ir.weight_banks.values():
            bank.parameter_scale = torch.tensor(0.0)
        quantize_ir_graph(ir, 8, weight_quantization=True)
        IRLatency(ir).calculate()
        return ir

    def float_ir(rep, form):
        ""
        ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS,
                       allow_coalescing=coalescing).map(rep)
        through_window(ir, form)
        IRLatency(ir).calculate()
        return ir

    for seed in range(s0, s1):
        forms = {}
        for form in sorted(set(window.values())):
            try:
                ir = integer_ir(rep_of(seed), form)
                unbounded = integer_ir(rep_of(seed), form, unbounded=True)
                retimes = (False, True)
                idents = {r: build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=r) for r in retimes}

                theorem_idents = {r: build_identity_hybrid_mapping(ir_graph=unbounded, retimed_level_stages=r)
                                  for r in retimes}
                forms[form] = (ir, unbounded, idents, theorem_idents)
            except Exception as exc:
                forms[form] = f"{type(exc).__name__}: {str(exc)[:200]}"
        if isinstance(forms.get(False), str):
            print(json.dumps({"seed": seed, "vehicle_error": forms[False]}), file=out, flush=True)
            continue
        ir, unbounded, idents, theorem_idents = forms[False]
        try:
            ir_float = V.ir_of(rep_of(seed), coalescing=coalescing)
        except Exception as exc:
            print(json.dumps({"seed": seed, "vehicle_error": f"{type(exc).__name__}: {str(exc)[:200]}"}), file=out,
                  flush=True)
            continue
        sizes = [len(s.hard_core_mapping.cores) for s in idents[False].stages if s.kind == "neural"]
        facts = _tiling_facts(ir, unbounded)
        facts["unbounded_timing_derivation"] = _derivation(unbounded, theorem_idents[False])
        facts["tiled_timing_derivation"] = _derivation(ir, idents[False])
        for count in counts:
            row = {"seed": seed, "count": count, "family": family, "identity_stage_cores": sizes, **facts}
            programs = {}
            for form, built in sorted(forms.items()):
                if isinstance(built, str):
                    continue
                for retimed in (False, True):
                    t0 = time.perf_counter()
                    key = f"{int(retimed)}" if not form else f"window_{int(retimed)}"
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            programs[(form, retimed)] = V.build(built[0], count, None, retimed_level_stages=retimed,
                                                                **caps)
                        row[f"compile_s_{key}"] = round(time.perf_counter() - t0, 4)
                    except ToolchainRefusal as exc:
                        row[f"refused_{key}"] = type(exc).__name__
                    except Exception as exc:
                        row[f"error_{key}"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            if (False, False) in programs:
                passes = [s for s in programs[(False, False)].stages if s.kind == "neural"]
                row["passes"] = len(passes)
                row["multipass"] = len(passes) > len(sizes)
            for name, sem in semantics.items():
                retimed = name == "windowed_lif"
                form = window[name]
                built = forms.get(form)
                if isinstance(built, str):
                    row[name] = {"window_build": built}
                    continue
                if (form, retimed) not in programs:
                    if form:
                        row[name] = {"refused": row.get(f"refused_window_{int(retimed)}")
                                     or row.get(f"error_window_{int(retimed)}")}
                    continue
                ir_s, unbounded_s, idents_s, theorem_s = built
                program = programs[(form, retimed)]
                decl = CheckDeclaration(semantics=sem, window=V.T, pass_transfer=runs[name]["pass_transfer"],
                                        core_types=tuple(V.cores(count)))
                try:
                    report = check_deployment(program, ir_graph=ir_s, declaration=decl, identity=idents_s[retimed],
                                              unbounded_ir=unbounded_s)
                    cell = {"t1": report.theorem1, "p2": report.proposition2, "fail": report.failures_by_condition(),
                            "check_s": round(report.wall_s, 4), "ir_tiling": report.results["ir-tiling"].status,
                            "ir_tiling_s": report.counts.get("ir_tiling_s")}
                except Exception as exc:
                    cell = {"check_error": f"{type(exc).__name__}: {str(exc)[:160]}"}
                if form:
                    cell["relay_cores"] = sum(1 for n in ir_s.get_neural_cores() if getattr(n, "_depth_relay", False))
                    cell["passes"] = sum(1 for s in program.stages if s.kind == "neural")
                xs = [torch.rand(6, V.IN, generator=torch.Generator().manual_seed(b)) for b in range(3)]
                try:
                    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                        wants = [R.flow(idents_s[retimed], **runs[name])(x) for x in xs]
                        gots = [R.flow(program, **runs[name])(x) for x in xs]
                        theorem = [R.flow(theorem_s[retimed], **runs[name])(x) for x in xs]
                    cell["sampled_unbounded"] = ("equal" if all(bool(torch.equal(g, w)) for g, w in zip(gots, theorem))
                                                 else "DIFFERS")
                    cell["fused_cores"] = sum(1 for st in program.stages if st.kind == "neural"
                                              for c in st.hard_core_mapping.cores
                                              if c.fused_component_axons and len(c.fused_component_axons) > 1)
                    cell["sampled"] = "equal" if all(bool(torch.equal(g, w)) for g, w in zip(gots, wants)) else "DIFFERS"
                except Exception as exc:
                    cell["sampled"] = f"run_error {type(exc).__name__}: {str(exc)[:100]}"
                row[name] = cell
            if (False, False) in programs:


                streamed = window["streamed_lif"]
                try:
                    flt = float_ir(rep_of(seed), streamed) if streamed else ir_float
                    float_program = V.build(flt, count, None, **caps)
                    decl = CheckDeclaration(semantics="streamed_lif", window=V.T, pass_transfer="verbatim",
                                            core_types=tuple(V.cores(count)))
                    report = check_deployment(float_program, ir_graph=flt, declaration=decl)
                    row["float_ir_streamed"] = {"t1": report.theorem1, "fail": report.failures_by_condition()}
                except Exception as exc:
                    row["float_ir_streamed"] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
            print(json.dumps(row), file=out, flush=True)


def _value_rows(out, label, program_of, builds) -> None:
    from mimarsinan.certification.program_check import CheckDeclaration
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        prog = program_of()
    ir_s = round(time.perf_counter() - t0, 2)
    for name, kwargs in builds:
        row = {"vehicle": label, "build": name, "ir_s": ir_s, "ir_cores": len(prog.ir.get_neural_cores())}
        try:
            t1 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                program = prog.packed_mapping(**kwargs)
            row["compile_s"] = round(time.perf_counter() - t1, 2)
            row["passes"] = sum(1 for s in program.stages if s.kind == "neural")
            row["hard_cores"] = sum(len(s.hard_core_mapping.cores) for s in program.stages if s.kind == "neural")
            decl = CheckDeclaration(semantics="value_integer", window=1, pass_transfer="collapse",
                                    core_types=({"max_axons": kwargs["axons"], "max_neurons": kwargs["neurons"],
                                                 "count": kwargs.get("count", 999)},),
                                    accumulation={"value_census": "float64", "int64_golden": "int64"},
                                    upload={"value_census": "float32"})
            report = check_deployment(program, ir_graph=prog.ir, declaration=decl, compile_wall_s=row["compile_s"],
                                      unbounded_ir=prog.ir)
            row.update(_row_of(report))
            row["unbounded_ir"] = "the program's own IR (build_integer_program maps without core limits)"
        except ToolchainRefusal as exc:
            row["refused"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            row["trace"] = traceback.format_exc()[-1500:]
        print(json.dumps(row), file=out, flush=True)


def _x3a(out, screen: str) -> None:
    ""
    import mimarsinan.certification.randomized.executors as ex
    from mimarsinan.certification.randomized.arms import ARMS
    from mimarsinan.certification.randomized.runner import semantics_for
    from mimarsinan.mapping.ir_mapping_class import IRMapping

    keyed_of = ARMS["R4"]

    def binding(seed):
        return W.building(IRMapping, "map", ex.RUNS[semantics_for(seed, keyed_of)].config)
    return _x3a_rows(out, screen, binding)


def _x3a_rows(out, screen: str, binding) -> None:
    ""
    from dataclasses import replace

    from mimarsinan.certification.program_check import CheckDeclaration
    from mimarsinan.certification.randomized.arms import ARMS
    from mimarsinan.certification.randomized.cost_draws import platform_constraints
    from mimarsinan.certification.randomized.runner import (
        _run, compare, deployment, instance_for, reference, targeted_declaration)

    off = ARMS["R4"]
    keyed = replace(off, id="R4/keyed", switch=None)
    rows = [json.loads(line) for line in open(os.path.join(screen, "R4.jsonl"))]
    for screen_row in rows:
        if screen_row["outcome"] not in ("equal", "counterexample", "vacuous"):
            continue
        seed = int(screen_row["seed"])
        row = {"seed": seed, "screen_outcome": screen_row["outcome"]}
        try:
            with contextlib.redirect_stdout(io.StringIO()), binding(seed):
                instance = instance_for(seed, keyed)
                cons = platform_constraints(targeted_declaration(seed, keyed), hardware_bias=instance.hardware_bias)
                ref = reference(instance, keyed)
                t0 = time.perf_counter()
                ir, mapping = deployment(instance, cons, seed, keyed)
                row["compile_s"] = round(time.perf_counter() - t0, 4)
                decl = CheckDeclaration(semantics="windowed_lif", window=int(instance.timesteps),
                                        pass_transfer="collapse", dithered_encode=True,
                                        core_types=tuple(dict(c) for c in cons["cores"]))
                report = check_deployment(mapping, ir_graph=ir, declaration=decl, unbounded_ir=ref.ir,
                                          compile_wall_s=row["compile_s"])
                with torch.no_grad():
                    row["sampled_mismatches"] = compare(ref.ir, ref.nodes, ir, _run(mapping, instance, keyed))[1]
                    with off.engaged():
                        moved = _run(mapping, instance, off)
                    row["sampled_mismatches_position_keyed"] = compare(ref.ir, ref.nodes, ir, moved)[1]
            row.update(_row_of(report))
            row.update(_tiling_facts(ir, ref.ir))
            row["unbounded_timing_derivation"] = _derivation(ref.ir, ref.mapping)
        except ToolchainRefusal as exc:
            row["refused"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            row["trace"] = traceback.format_exc()[-1500:]
        print(json.dumps(row), file=out, flush=True)


def main() -> int:
    mode, dst = sys.argv[1], sys.argv[2]
    with open(dst, "a") as out:
        if mode in ("configs", "archived"):
            for path in sys.argv[3:]:
                row = _check_config(path, "random_init" if mode == "configs" else "archived")
                print(json.dumps(row), file=out, flush=True)
                print(row["config"], row.get("theorem1") or row.get("refused") or row.get("error")
                      or row.get("skipped"), flush=True)
        elif mode == "x3a":
            _x3a(out, sys.argv[3])
        elif mode == "fuzz":
            _fuzz(out, int(sys.argv[3]), int(sys.argv[4]), [int(c) for c in sys.argv[5].split(",")])
        elif mode == "r18":
            ckpt, screen = sys.argv[3], sys.argv[4]
            sys.path.insert(0, screen)
            from screen_models import CifarResNet18
            from unit.chip_simulation.value_run.integer_programs import build_integer_program

            def program_of():
                model = CifarResNet18(1.0, 100)
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
                return build_integer_program(model.eval(), (3, 32, 32), num_classes=100,
                                             readout_law={"adc_bits": 40, "range": "calibrated", "mult_bits": 16},
                                             activation_bits=4, weight_bits=4, calibration_size=32)
            half = int(os.environ.get("W15_R18_HALF", "2868"))
            _value_rows(out, "cifar_resnet18_w1_C1", program_of, [
                ("resident_256x512", dict(axons=256, neurons=512, count=200000, coalescing=True)),
                ("scheduled_256x512_half", dict(axons=256, neurons=512, count=half, coalescing=True, passes=4096)),
            ])
        elif mode == "lenet":
            from unit.chip_simulation.value_run.integer_programs import build_integer_program, lenet5
            _value_rows(out, "lenet5_mnist", lambda: build_integer_program(lenet5(), (1, 28, 28), num_classes=10), [
                ("resident_1024x128", dict(axons=1024, neurons=128, count=5000)),
                ("coalesced_256x128", dict(axons=256, neurons=128, count=5000, coalescing=True)),
                ("scheduled_1024x128_c4", dict(axons=1024, neurons=128, count=4, passes=4096)),
                ("scheduled_256x128_c8_coalesced", dict(axons=256, neurons=128, count=8, coalescing=True,
                                                        passes=4096)),
            ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
