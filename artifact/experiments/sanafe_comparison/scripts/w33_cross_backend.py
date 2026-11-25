""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import time
import traceback
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__import__("mimarsinan").__file__))))
sys.path.insert(0, os.path.join(TREE, "tests"))

import numpy as np
import torch

torch.set_num_threads(1)
os.environ.setdefault("MIMARSINAN_LOIHI_WAVE_WORKERS", "1")
os.environ.setdefault("MIMARSINAN_LOIHI_QUIET", "1")

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import enumerate_execution_stages
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.latency.chip import PINNED_LATENCIES
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.pass_timing import (
    REPLAY_DELAYS, carried_nodes, executed_core_latencies, replay_delays)
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.nn.lif_kernels import measurement_plane

from unit.mapping.residency.executed import runs as R
from unit.mapping.residency.executed import vehicles as V

N_SAMPLES = int(os.environ.get("W33_SAMPLES", "3"))

ONLY = set((os.environ.get("W33_BACKENDS") or "nevresim,sanafe,lava").split(","))

SEMANTICS = {
    "streamed_lif": dict(hcm=R.STREAMED_LIF, retimed=False, transfer=VERBATIM, spiking_mode="lif",
                         firing_mode="Default", spike="Uniform", schedule="cascaded",
                         backends=("nevresim", "sanafe", "lava")),
    "windowed_lif": dict(hcm=R.WINDOWED_LIF, retimed=True, transfer=COLLAPSE, spiking_mode="lif",
                         firing_mode="Default", spike="Uniform", schedule="cascaded",
                         backends=("nevresim", "sanafe", "lava")),
    "cascaded_ttfs": dict(hcm=R.CASCADED_TTFS, retimed=False, transfer=COLLAPSE, spiking_mode="ttfs_cycle_based",
                          firing_mode="TTFS", spike="TTFS", schedule="cascaded", backends=("nevresim", "sanafe")),
}

NOT_EXECUTED = {("cascaded_ttfs", "lava"): "Lava implements LIF only (chip_simulation/backend.py _loihi_ttfs_error)"}


def _units(mapping):
    return [(k, stage, unit) for k, stage, unit in enumerate_execution_stages(mapping.stages) if unit.kind == "neural"]


def timing_facts(mapping) -> dict:
    ""
    host = carried = delayed = 0
    for _k, stage, unit in _units(mapping):
        nodes = carried_nodes(mapping.stages, stage)
        rows = {}
        for s in unit.input_map:
            rows.update({i: int(s.node_id) for i in range(int(s.offset), int(s.offset) + int(s.size))})
        hcm = unit.hard_core_mapping
        for core, lat in zip(hcm.cores, executed_core_latencies(hcm)):
            if lat <= 0:
                continue
            read = {rows.get(int(src.neuron_)) for src in core.axon_sources if getattr(src, "is_input_", False)
                    and not getattr(src, "is_off_", False) and not getattr(src, "is_always_on_", False)}
            host += any(n not in nodes for n in read)
            carried += any(n in nodes for n in read)
        if not getattr(unit, "is_retimed_level", False):
            delayed += len(replay_delays(unit))
    passes = [s for s in mapping.stages if s.kind == "neural"]
    return {"passes": len(passes), "units": len(_units(mapping)), "host_late": host, "carried_late": carried,
            "delayed": delayed,
            "pinned": sum(getattr(s.hard_core_mapping, PINNED_LATENCIES, None) is not None for s in passes)}


def hcm_units(mapping, xs, sem, hcm_kwargs=None, flow=None):
    ""
    seen = []
    flow = flow if flow is not None else R.flow(mapping, **(hcm_kwargs or SEMANTICS[sem]["hcm"]))
    flow.stage_count_recorder = lambda stage, counts: seen.append(counts.detach().cpu().numpy().astype(np.float64))
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        out = flow(torch.cat(xs)).detach().cpu().numpy()
    flow.stage_count_recorder = None
    return [np.rint(c).astype(np.int64) for c in seen], out


def _declaration(sem, decl=None):
    d = dict(SEMANTICS[sem]) if decl is None else decl
    return d


def _compile_in_process(fn, task_args, **_kwargs):
    ""
    return {key: fn(*args) for key, args in task_args.items()}


def nevresim_units(mapping, xs, sem, decl=None):
    from mimarsinan.chip_simulation.nevresim.connectivity import default_nevresim_connectivity_mode
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.chip_simulation.simulation_runner.core import SimulationRunner
    from mimarsinan.chip_simulation.simulation_runner import emit as nevresim_emit
    NevresimDriver.nevresim_path = os.path.join(TREE, "nevresim")
    nevresim_emit.run_tasks_in_pool_bounded = _compile_in_process
    d = _declaration(sem, decl)
    seen = []
    with tempfile.TemporaryDirectory(prefix="w33nev_") as work:
        runner = object.__new__(SimulationRunner)
        vars(runner).update(
            spike_generation_mode=d["spike"], firing_mode=d["firing_mode"], thresholding_mode=d.get("thresholding", "<="),
            spiking_mode=d["spiking_mode"], nevresim_connectivity_mode=default_nevresim_connectivity_mode(),
            pass_transfer=d["transfer"], simulation_step_timeout_s=900.0, membrane_readout=False,
            membrane_half_step_charge=False, weight_type=float, threshold_type=float, host_compute_device="cpu",
            stage_timer=None, simulation_length=d.get("T", V.T), mapping=mapping, num_classes=1024,
            test_data=[(x[0].numpy().reshape(-1).astype(np.float64), np.zeros(1)) for x in xs],
            test_targets=[torch.tensor(0) for _ in xs], test_input=[], working_directory=work)
        runner.stage_count_recorder = lambda stage, raw: seen.append(np.rint(np.asarray(raw)).astype(np.int64))
        with contextlib.redirect_stdout(io.StringIO()):
            runner._run_hybrid(mapping)
    return seen


def sanafe_units(mapping, xs, sem, decl=None):
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    d = _declaration(sem, decl)
    behavior = NeuralBehaviorConfig(spiking_mode=d["spiking_mode"], firing_mode=d["firing_mode"],
                                    thresholding_mode=d.get("thresholding", "<="), spike_generation_mode=d["spike"])


    runner = SanafeRunner(mapping=mapping, simulation_length=d.get("T", V.T), behavior=behavior,
                          ttfs_cycle_schedule=d["schedule"], pass_transfer=d["transfer"],
                          log_message_trace=os.environ.get("W33_SANAFE_MESSAGE_TRACE", "1") != "0")
    per_unit = {}
    for x in xs:
        with contextlib.redirect_stdout(io.StringIO()):
            rec = runner.run(x.numpy().reshape(1, -1).astype(np.float64), sample_index=0)
        for k, seg in rec.segments.items():
            per_unit.setdefault(k, []).append(np.asarray(seg.seg_output_spike_count, dtype=np.int64)[None])
    return [np.concatenate(per_unit[k]) for k in sorted(per_unit)]


def lava_units(mapping, xs, sem, decl=None):
    ""
    from mimarsinan.chip_simulation.lava_loihi import LavaLoihiRunner
    d = _declaration(sem, decl)
    T = d.get("T", V.T)
    behavior = NeuralBehaviorConfig(spiking_mode=d["spiking_mode"], firing_mode=d["firing_mode"],
                                    thresholding_mode=d.get("thresholding", "<="), spike_generation_mode=d["spike"])
    runner = LavaLoihiRunner(mapping=mapping, simulation_length=T, behavior=behavior, pass_transfer=d["transfer"])
    x_np = torch.cat(xs).numpy().reshape(len(xs), -1).astype(np.float64)
    runner.pipeline = SimpleNamespace(reporter=SimpleNamespace(report=lambda *a, **k: None))
    runner._load_test_samples = lambda: (x_np, np.zeros(len(xs), dtype=np.int64))
    seen = []
    inner = runner._run_neural_segment

    def capture(*args, **kwargs):
        out = inner(*args, **kwargs)
        seen.append(np.rint(np.asarray(out, dtype=np.float64) * T).astype(np.int64))
        return out
    runner._run_neural_segment = capture
    with contextlib.redirect_stdout(io.StringIO()):
        runner.run()
    return seen


LAVA_WALL_S = float(os.environ.get("W33_LAVA_WALL_S", "300"))


class LavaWallCap(Exception):
    ""


def lava_units_capped(mapping, xs, sem, decl=None):
    ""
    import multiprocessing as mp
    import pickle
    import signal
    ctx = mp.get_context("fork")
    reader, writer = ctx.Pipe(duplex=False)

    def child():
        os.setpgrp()
        try:
            writer.send(("ok", pickle.dumps(lava_units(mapping, xs, sem, decl))))
        except BaseException as exc:
            writer.send(("error", f"{type(exc).__name__}: {str(exc)[:300]}"))
    proc = ctx.Process(target=child)
    proc.start()
    try:
        if reader.poll(LAVA_WALL_S):
            kind, payload = reader.recv()
            proc.join(30)
            if kind == "ok":
                return pickle.loads(payload)
            raise RuntimeError(payload)
        raise LavaWallCap(f"Lava run past the {LAVA_WALL_S:.0f} s harness wall cap")
    finally:
        if proc.is_alive():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.join(5)


BACKENDS = {"nevresim": nevresim_units, "sanafe": sanafe_units, "lava": lava_units_capped}


def compare_units(got, want) -> dict:
    if len(got) != len(want):
        return {"equal": False, "units": len(want), "units_got": len(got), "units_equal": 0}
    eq = [g.shape == w.shape and bool(np.array_equal(g, w)) for g, w in zip(got, want)]
    differ = sum(int((g != w).sum()) if g.shape == w.shape else int(w.size) for g, w in zip(got, want))
    return {"equal": all(eq), "units": len(want), "units_equal": sum(eq), "entries_differ": differ,
            "entries": int(sum(w.size for w in want))}


def run_cell(program, identity, xs, sem, backends, decl=None, hcm_kwargs=None, flows=None) -> dict:
    ""
    with measurement_plane():
        return _run_cell(program, identity, xs, sem, backends, decl, hcm_kwargs, flows)


def _run_cell(program, identity, xs, sem, backends, decl, hcm_kwargs, flows) -> dict:
    cell = {}
    t0 = time.perf_counter()
    try:
        want, out = hcm_units(program, xs, sem, hcm_kwargs, flow=None if flows is None else flows[0])
        _, out_identity = hcm_units(identity, xs, sem, hcm_kwargs, flow=None if flows is None else flows[1])
        cell["hcm_equals_identity"] = bool(np.array_equal(out, out_identity))
        cell["discriminates"] = bool(np.asarray(out_identity).std(0).max() > 0)
        cell["hcm_s"] = round(time.perf_counter() - t0, 2)
    except ToolchainRefusal as exc:
        cell["hcm_refused"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return cell
    except Exception as exc:
        cell["hcm_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return cell
    for backend in backends:
        if (sem, backend) in NOT_EXECUTED:
            cell[backend] = {"not_executed": NOT_EXECUTED[(sem, backend)]}
            continue
        t1 = time.perf_counter()
        try:
            got = BACKENDS[backend](program, xs, sem, decl)
            res = compare_units(got, want)
        except ToolchainRefusal as exc:
            res = {"refused": f"{type(exc).__name__}: {str(exc)[:240]}"}
        except Exception as exc:
            res = {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "trace": traceback.format_exc()[-800:]}
        res["s"] = round(time.perf_counter() - t1, 2)
        cell[backend] = res
    return cell


def integer_ir(rep, bits: int = 8):
    ""
    from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.latency.ir import IRLatency
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS).map(rep)
    for node in ir.get_neural_cores():
        node.parameter_scale = torch.tensor(0.0)
    for bank in ir.weight_banks.values():
        bank.parameter_scale = torch.tensor(0.0)
    quantize_ir_graph(ir, bits, weight_quantization=True)
    IRLatency(ir).calculate()
    return ir


def _fuzz_programs(seed, counts, retimed):
    ir = integer_ir(V.random_rep(seed))
    identity = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=retimed)
    yield 0, identity, identity
    for count in counts:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                program = V.build(ir, count, None, retimed_level_stages=retimed)
        except ToolchainRefusal as exc:
            yield count, f"refused {type(exc).__name__}", identity
            continue
        yield count, program, identity


def _xs(width, n=N_SAMPLES):
    return [torch.rand(1, width, generator=torch.Generator().manual_seed(b)) for b in range(n)]


def run_fuzz(out, pairs, semantics=None) -> None:
    ""
    by_seed = {}
    for seed, count in pairs:
        by_seed.setdefault(int(seed), set()).add(int(count))
    for seed in sorted(by_seed):
        counts = sorted(c for c in by_seed[seed] if c)
        want_identity = 0 in by_seed[seed]
        for sem in semantics or SEMANTICS:
            spec = SEMANTICS[sem]
            try:
                programs = list(_fuzz_programs(seed, counts, spec["retimed"]))
            except Exception as exc:
                print(json.dumps({"seed": seed, "semantics": sem, "vehicle_error": f"{type(exc).__name__}: {exc}"[:300]}),
                      file=out, flush=True)
                continue
            for count, program, identity in programs:
                if count == 0 and not want_identity:
                    continue
                row = {"set": "fuzz", "seed": seed, "count": count, "semantics": sem}
                if isinstance(program, str):
                    row["build"] = program
                    print(json.dumps(row), file=out, flush=True)
                    continue
                row.update(timing_facts(program))
                row["multipass"] = row["passes"] > sum(1 for s in identity.stages if s.kind == "neural")
                row.update(run_cell(program, identity, _xs(V.IN), sem, [b for b in spec["backends"] if b in ONLY]))
                print(json.dumps(row), file=out, flush=True)


def plan(out, s0, s1, counts) -> None:
    for seed in range(s0, s1):
        for retimed in (False, True):
            try:
                for count, program, identity in _fuzz_programs(seed, counts, retimed):
                    row = {"seed": seed, "count": count, "retimed": retimed}
                    if isinstance(program, str):
                        row["build"] = program
                    else:
                        row.update(timing_facts(program))
                        row["multipass"] = row["passes"] > sum(1 for s in identity.stages if s.kind == "neural")
                    print(json.dumps(row), file=out, flush=True)
            except Exception as exc:
                print(json.dumps({"seed": seed, "retimed": retimed, "error": f"{type(exc).__name__}: {exc}"[:300]}),
                      file=out, flush=True)


CENSUS = [("chain", 3), ("chain", 5), ("skip", 2), ("skew3", 2), ("skew3", 3), ("shift_probe", 2),
          ("shift_probe", 3), ("shift_streamed", 3), ("late_skip", 2), ("deep_skip", 2), ("seed71", 2),
          ("seed191", 2), ("chain", 6), ("late_skip", 3), ("deep_skip", 4), ("shift_streamed", 4), ("seed58", 2),
          ("seed2", 2), ("seed0", 3), ("seed5", 2), ("seed11", 3)]


def _census_ir(name):
    if name == "chain":
        return V.ir_of(V.chain_repr(seed=0))
    return V.ir_of(V.random_rep(int(name[4:])) if name.startswith("seed") else V.branchy_repr(name))


def run_census(out, semantics=None) -> None:
    for name, count in CENSUS:
        for sem in semantics or SEMANTICS:
            spec = SEMANTICS[sem]
            ir = _census_ir(name)
            identity = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=spec["retimed"])
            with contextlib.redirect_stdout(io.StringIO()):
                u1 = V.build(ir, count, None, retimed_level_stages=spec["retimed"])
            for which, program in (("identity", identity), ("u1", u1)):
                row = {"set": "census", "vehicle": name, "C": count, "program": which, "semantics": sem}
                row.update(timing_facts(program))
                row.update(run_cell(program, identity, _xs(V.IN), sem, [b for b in spec["backends"] if b in ONLY]))
                print(json.dumps(row), file=out, flush=True)


def _template_semantics(cfg) -> str | None:
    from mimarsinan.chip_simulation.activation_semantics import resolve_activation_semantics
    s = resolve_activation_semantics(cfg)
    return {("lif", "streamed"): "streamed_lif", ("lif", "synchronized"): "windowed_lif",
            ("ttfs", "cascaded"): "cascaded_ttfs"}.get((s.family, s.variant))


def run_templates(out, scale, names) -> None:
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.chip_simulation.activation_semantics import resolve_activation_semantics
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.pipelining.core.simulation_factory import (
        build_hybrid_mapping_for_pipeline, build_identity_mapping_for_pipeline, build_spiking_hybrid_flow)
    from unit.mapping.residency.executed import templates as TV
    for name in names:
        row = {"set": "templates", "template": name, "scale": scale}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                vehicle = TV.template_vehicle(name)


            for op in vehicle.ir.get_compute_ops():
                module = (getattr(op, "params", None) or {}).get("module")
                if module is not None and hasattr(module, "eval"):
                    module.eval()
            cfg = dict(vehicle.config)
            s = resolve_activation_semantics(cfg)
            row["declared"] = f"{s.family}/{s.variant}" if str(cfg.get("core_semantics", "spiking")) != "mvm" else "mvm"
            sem = None if row["declared"] == "mvm" else _template_semantics(cfg)
            if sem is None:
                row["outside"] = ("value domain: no time axis, certified against the int64 golden"
                                  if row["declared"] == "mvm" else "time-free or HCM-only semantics")
                print(json.dumps(row), file=out, flush=True)
                continue
            row["semantics"] = sem
            contract = SpikingDeploymentContract.from_pipeline_config(cfg)
            T = int(cfg["simulation_steps"])
            decl = dict(SEMANTICS[sem], spiking_mode=contract.spiking_mode, firing_mode=contract.firing_mode,
                        thresholding=contract.thresholding_mode, spike=contract.spike_generation_mode, T=T,
                        schedule=str(cfg.get("ttfs_cycle_schedule", "cascaded")))
            from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer
            decl["transfer"] = run_pass_transfer(cfg)
            xs = [torch.rand(1, *vehicle.input_shape, generator=torch.Generator().manual_seed(b))
                  for b in range(N_SAMPLES)]
            bits = int(cfg.get("weight_bits", 8)) if DeploymentPlan.resolve(cfg).weight_quantization else 0
            gain = _activate(vehicle, cfg, xs, bits)
            row.update(gain=gain, quantized_bits=bits, T=T)
            platform = TV.residency_platform(vehicle, scale, None)
            with contextlib.redirect_stdout(io.StringIO()):
                identity = build_identity_mapping_for_pipeline(vehicle.ir, pipeline_config=cfg)
                program = build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=cfg)
            row.update(timing_facts(program))
            row["multipass"] = row["passes"] > sum(1 for st in identity.stages if st.kind == "neural")
            pipe = SimpleNamespace(config=cfg)
            flows = (build_spiking_hybrid_flow(pipe, program), build_spiking_hybrid_flow(pipe, identity))
            backends = [b for b in ("nevresim", "sanafe", "lava") if b in ONLY]
            row.update(run_cell(program, identity, xs, sem, backends, decl=decl, flows=flows))
        except ToolchainRefusal as exc:
            row["refused"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            row["trace"] = traceback.format_exc()[-1200:]
        print(json.dumps(row), file=out, flush=True)


def _activate(vehicle, cfg, xs, bits) -> float:
    ""
    from mimarsinan.pipelining.core.simulation_factory import build_identity_mapping_for_pipeline, build_spiking_hybrid_flow
    from unit.mapping.residency.executed import templates as TV
    original = {int(c.id): None if c.core_matrix is None else np.array(c.core_matrix) for c in vehicle.ir.get_neural_cores()}
    banks = {k: np.array(b.core_matrix) for k, b in vehicle.ir.weight_banks.items()}
    chosen = 0.0
    for gain in (0.0, 8.0, 32.0):
        for c in vehicle.ir.get_neural_cores():
            if original[int(c.id)] is not None:
                c.core_matrix = np.array(original[int(c.id)])
        for k, b in vehicle.ir.weight_banks.items():
            b.core_matrix = np.array(banks[k])
        if gain:
            TV.activate(vehicle.ir, gain)
        chosen = gain
        with contextlib.redirect_stdout(io.StringIO()):
            flow = build_spiking_hybrid_flow(SimpleNamespace(config=cfg),
                                             build_identity_mapping_for_pipeline(vehicle.ir, pipeline_config=cfg))
            with torch.no_grad():
                outs = torch.cat([flow(x) for x in xs])
        if float(outs.std(0).max()) > 0:
            break
    if bits:
        from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
        for node in vehicle.ir.get_neural_cores():
            node.parameter_scale = torch.tensor(0.0)
        for bank in vehicle.ir.weight_banks.values():
            bank.parameter_scale = torch.tensor(0.0)
        quantize_ir_graph(vehicle.ir, bits, weight_quantization=True)
    return chosen


def main() -> int:
    mode, out_path = sys.argv[1], sys.argv[2]
    sems = os.environ.get("W33_SEMANTICS")
    sems = sems.split(",") if sems else None
    with open(out_path, "a") as out:
        if mode == "plan":
            plan(out, int(sys.argv[3]), int(sys.argv[4]), [int(c) for c in sys.argv[5].split(",")])
        elif mode == "fuzz":
            counts = [int(c) for c in sys.argv[5].split(",")]
            run_fuzz(out, [(s, c) for s in range(int(sys.argv[3]), int(sys.argv[4])) for c in [0] + counts], sems)
        elif mode == "pick":
            pick = json.load(open(sys.argv[3]))
            run_fuzz(out, pick["programs"], sems)
        elif mode == "census":
            run_census(out, sems)
        elif mode == "templates":
            run_templates(out, float(sys.argv[3]), sys.argv[4:])
        else:
            raise SystemExit(f"unknown mode {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
