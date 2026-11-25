""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
import mimarsinan

SRC = os.path.dirname(os.path.abspath(mimarsinan.__file__))
TESTS = os.environ.get("W16_TESTS", "/work/worktrees/w2-exact/tests")
sys.path.insert(0, TESTS)

import numpy as np
import torch

torch.set_num_threads(1)

from mimarsinan.certification.program_check import CheckDeclaration, check_deployment
from mimarsinan.common.refusal import ToolchainRefusal

SYNC_TTFS = dict(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized", firing_mode="TTFS",
                 spike_mode="TTFS", pass_transfer="collapse")


def _brief(report) -> dict:
    return {"t1": report.theorem1, "p2": report.proposition2, "fail": report.failures_by_condition(),
            "statuses": {c: r.status for c, r in report.results.items()},
            "violations": {c: [v.text() for v in r.violations[:2]] for c, r in report.results.items()
                           if r.violations},
            "notes": {c: r.note for c, r in report.results.items() if r.note and c in ("H10", "R4", "H8")},
            "check_s": round(report.wall_s, 4)}


def _gate(want: torch.Tensor, got: torch.Tensor) -> dict:
    ""
    want, got = want.double().reshape(want.shape[0], -1), got.double().reshape(got.shape[0], -1)
    if want.shape != got.shape:
        return {"samples": int(want.shape[0]), "outputs": int(want.shape[0]), "decisions": int(want.shape[0]),
                "shape_mismatch": True}
    same = (want == got) | (torch.isnan(want) & torch.isnan(got))
    return {"samples": int(want.shape[0]), "outputs": int((~same.all(dim=1)).sum()),
            "decisions": int((want.argmax(dim=1) != got.argmax(dim=1)).sum())}


def _add(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("samples", "outputs", "decisions")}


def _x3a_declaration(instance, arm, constraints) -> CheckDeclaration:
    from mimarsinan.certification.randomized import executors as ex
    semantics = instance.semantics
    run = ex.RUNS[semantics]
    accumulation = ({"value_census": arm.value_dtype, "int64_golden": "int64"} if semantics == "value"
                    else {"hcm": "float64"})
    return CheckDeclaration(semantics={"value": "value_integer"}.get(semantics, semantics),
                            window=int(instance.timesteps), pass_transfer=arm.transfer or run.transfer,
                            encoder_position_free=not arm.phase_dither,
                            core_types=tuple({k: c[k] for k in ("max_axons", "max_neurons", "count")}
                                             for c in constraints["cores"]),
                            accumulation=accumulation)


def _x3a_one(seed: int, arm) -> dict:
    from mimarsinan.certification.randomized import executors as ex
    from mimarsinan.certification.randomized import program as xp
    from mimarsinan.certification.randomized import runner as RN
    from mimarsinan.certification.randomized.cost_draws import platform_constraints

    instance = RN.instance_for(seed, arm)
    declaration = RN.targeted_declaration(seed, arm)
    constraints = platform_constraints(declaration, hardware_bias=instance.hardware_bias)
    row: dict = {"semantics": instance.semantics, "timesteps": instance.timesteps}
    try:
        ref = RN.reference(instance, arm)
    except RN.UNNAMED_FAILURES + (ToolchainRefusal,) as exc:
        row.update(outcome="reference_error", detail=f"{type(exc).__name__}: {str(exc)[:160]}")
        return row
    decl = _x3a_declaration(instance, arm, constraints)
    with arm.engaged():
        try:
            dep_ir, dep_map = RN.deployment(instance, constraints, seed, arm)
        except ToolchainRefusal as exc:
            row.update(outcome="refused", detail=f"{type(exc).__name__}: {str(exc)[:160]}")
            return row
        except RN.UNNAMED_FAILURES as exc:
            row.update(outcome="refused_unnamed", detail=f"{type(exc).__name__}: {str(exc)[:160]}")
            return row
        try:
            dep_nodes = RN._run(dep_map, instance, arm)
            _w, mismatches, _f = RN.compare(ref.ir, ref.nodes, dep_ir, dep_nodes)
            golden = 0
            if ex.RUNS[instance.semantics].law is not None:
                x = ex.input_tensor(instance, torch.float64).numpy()
                for mapping, nodes in ((ref.mapping, ref.nodes), (dep_map, dep_nodes)):
                    golden += xp.golden_relation(mapping, x, nodes)[1]
        except RN.UNNAMED_FAILURES + (ToolchainRefusal,) as exc:
            row.update(outcome="run_error", detail=f"{type(exc).__name__}: {str(exc)[:160]}")
            return row
        try:
            from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
            ident_map = build_identity_hybrid_mapping(ir_graph=dep_ir,
                                                      retimed_level_stages=ex.RUNS[instance.semantics].retimed)
            ident_nodes = RN._run(ident_map, instance, arm, identity=True)
            row["same_ir"] = {"deployment_vs_identity": int(RN.compare(dep_ir, ident_nodes, dep_ir, dep_nodes)[1]),
                              "identity_vs_reference": int(RN.compare(ref.ir, ref.nodes, dep_ir, ident_nodes)[1])}
        except RN.UNNAMED_FAILURES + (ToolchainRefusal,) as exc:
            row["same_ir"] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        try:
            row["check_in"] = _brief(check_deployment(dep_map, ir_graph=dep_ir, declaration=decl))
        except Exception as exc:
            row["check_in"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    try:
        row["check_out"] = _brief(check_deployment(dep_map, ir_graph=dep_ir, declaration=decl))
    except Exception as exc:
        row["check_out"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    placed = ex.per_neuron(ref.ir, ref.nodes)
    flat = torch.cat([v.flatten() for v in placed.values()])
    active = float((flat != 0).double().mean()) if flat.numel() else 0.0
    departs = mismatches or golden
    row.update(outcome="counterexample" if departs else ("equal" if active > 0 else "vacuous"),
               mismatches=int(mismatches), golden_mismatches=int(golden),
               gate=_gate(ref.nodes[-1], dep_nodes[-1]),
               multipass=bool(RN.structure(dep_map)["multipass"]),
               passes=sum(1 for s in dep_map.stages if s.kind == "neural"))
    return row


def arms_mode(out, screen_dir: str, arm_ids) -> None:
    from mimarsinan.certification.randomized.arms import ARMS
    limit = int(os.environ.get("W16_LIMIT", "0") or 0)
    for arm_id in arm_ids:
        lines = open(os.path.join(screen_dir, f"{arm_id}.jsonl")).read().splitlines()
        for line in lines[:limit] if limit else lines:
            old = json.loads(line)
            row = {"arm": arm_id, "seed": int(old["seed"]), "closure_outcome": old["outcome"],
                   "closure_mismatches": old.get("mismatches", 0), "closure_classes": old.get("classes", [])}
            t0 = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    row["arm_run"] = _x3a_one(row["seed"], ARMS[arm_id])
                    if row["arm_run"].get("outcome") in ("counterexample", "equal", "vacuous"):
                        row["control_run"] = _x3a_one(row["seed"], ARMS[arm_id].control())
            except Exception as exc:
                row["harness_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                row["trace"] = traceback.format_exc()[-1200:]
            row["wall_s"] = round(time.perf_counter() - t0, 3)
            print(json.dumps(row), file=out, flush=True)


def reseed_mode(out, screen_file: str, arm_id: str, switch: str, outcomes) -> None:
    ""
    from dataclasses import replace
    from mimarsinan.certification.randomized.arms import ARMS
    for line in open(screen_file):
        old = json.loads(line)
        if old.get("outcome") not in outcomes:
            continue
        row = {"arm": arm_id, "switch": switch, "seed": int(old["seed"]), "closure_outcome": old["outcome"],
               "closure_classes": old.get("classes", []), "screen_file": screen_file}
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                arm = replace(ARMS[arm_id], switch=lambda: _switch(switch))
                with arm.engaged():
                    row["arm_run"] = _x3a_one(row["seed"], arm)
                row["control_run"] = _x3a_one(row["seed"], ARMS[arm_id])
        except Exception as exc:
            row["harness_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            row["trace"] = traceback.format_exc()[-1200:]
        row["wall_s"] = round(time.perf_counter() - t0, 3)
        print(json.dumps(row), file=out, flush=True)


@contextlib.contextmanager
def _patched(owner, name, value):
    saved = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, saved)


def _switch(name: str):
    ""
    from mimarsinan.certification.randomized import arms as A
    from mimarsinan.mapping.packing import pass_timing
    from mimarsinan.models.spiking.hybrid import rate_forward

    def refusals_only(stage, pl, fused, *_produced):
        pass_timing._core_latencies(pl, fused, stage.name)

    def unscoped(store, stage, *, publish=False):
        ""
        return store.setdefault("run", {}) if publish else store.get("run", {})

    if name == "none":
        return contextlib.nullcontext()
    if name == "pins_delays_off":
        return A._own_pass_timing()
    if name == "pins_off":
        return _patched(pass_timing, "_pin", refusals_only)
    if name == "delays_off":
        return _patched(pass_timing, "_delay", refusals_only)
    if name == "store_unscoped":
        return _patched(rate_forward, "carry_scope", unscoped)
    if name == "walks_latency":
        return A._latency_order_walks()
    if name == "coalescing_unleveled":
        return A._coalescing_unleveled()
    if name == "count_rebuild_base":
        from mimarsinan.models.spiking.hybrid.executors import sync_counts
        from mimarsinan.models.spiking.wire_semantics import lif_count_staircase

        def base_rebuild(total, theta, T, *, compare_mode="<="):
            ""
            stair = lif_count_staircase(total / float(T), theta, T, compare_mode=compare_mode)
            return stair * float(T) / torch.clamp(torch.as_tensor(theta, dtype=stair.dtype), min=1e-12)
        return _patched(sync_counts, "lif_window_count", base_rebuild)
    raise SystemExit(f"unknown switch {name!r}")


def fuzz_mode(out, s0: int, s1: int, counts, switch: str, semantics) -> None:
    from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
    from unit.certification.program_check import programs as P
    from unit.mapping.residency.executed import runs as R
    from unit.mapping.residency.executed import vehicles as V

    runs = dict(zip(R.RUN_IDS, R.ALL_RUNS))
    runs["synchronized_ttfs"] = SYNC_TTFS
    retimed_needed = sorted({s == "windowed_lif" for s in semantics})
    xs = [torch.rand(6, V.IN, generator=torch.Generator().manual_seed(b)) for b in range(3)]
    for seed in range(s0, s1):
        try:
            irs = {"integer": P.integer_ir(V.random_rep(seed)), "float": V.ir_of(V.random_rep(seed))}
            idents = {(k, r): build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=r)
                      for k, ir in irs.items() for r in retimed_needed}
        except Exception as exc:
            print(json.dumps({"seed": seed, "vehicle_error": f"{type(exc).__name__}: {str(exc)[:200]}"}),
                  file=out, flush=True)
            continue
        sizes = [len(s.hard_core_mapping.cores) for s in idents[("integer", retimed_needed[0])].stages
                 if s.kind == "neural"]
        for count in counts:
            row: dict = {"seed": seed, "count": count, "switch": switch}
            programs = {}
            with _switch(switch):
                for kind, ir in irs.items():
                    for r in retimed_needed:
                        try:
                            with contextlib.redirect_stdout(io.StringIO()):
                                programs[(kind, r)] = V.build(ir, count, None, retimed_level_stages=r)
                        except ToolchainRefusal as exc:
                            row[f"refused_{kind}_{int(r)}"] = type(exc).__name__
                        except Exception as exc:
                            row[f"error_{kind}_{int(r)}"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                base = programs.get(("integer", False)) or programs.get(("integer", True))
                if base is None:
                    print(json.dumps(row), file=out, flush=True)
                    continue
                passes = [s for s in base.stages if s.kind == "neural"]
                row["passes"] = len(passes)
                row["multipass"] = len(passes) > len(sizes)
                row["pinned"] = sum(getattr(s.hard_core_mapping, "pinned_latencies", None) is not None for s in passes)
                row["delayed"] = sum(len(getattr(s, "replay_delays", None) or {}) for s in passes)
                if not row["multipass"]:
                    print(json.dumps(row), file=out, flush=True)
                    continue
                for sem in semantics:
                    r = sem == "windowed_lif"
                    cell: dict = {}
                    for kind in ("float", "integer"):
                        prog = programs.get((kind, r))
                        if prog is None:
                            cell[kind] = {"sampled": "not built"}
                            continue
                        try:
                            gate = {"samples": 0, "outputs": 0, "decisions": 0}
                            equal = True
                            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                                for x in xs:
                                    want = R.flow(idents[(kind, r)], **runs[sem])(x)
                                    got = R.flow(prog, **runs[sem])(x)
                                    equal &= bool(torch.equal(got, want))
                                    gate = _add(gate, _gate(want, got))
                            cell[kind] = {"sampled": "equal" if equal else "DIFFERS", "gate": gate}
                        except Exception as exc:
                            cell[kind] = {"sampled": f"run_error {type(exc).__name__}: {str(exc)[:100]}"}
                    prog = programs.get(("integer", r))
                    if prog is not None:
                        decl = CheckDeclaration(semantics=sem, window=V.T, pass_transfer=runs[sem]["pass_transfer"],
                                                core_types=tuple(V.cores(count)))
                        try:
                            report = check_deployment(prog, ir_graph=irs["integer"], declaration=decl,
                                                      identity=idents[("integer", r)])
                            cell["static"] = {k: v for k, v in _brief(report).items() if k != "statuses"}
                        except Exception as exc:
                            cell["static"] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
                    row[sem] = cell
            print(json.dumps(row), file=out, flush=True)


def _program_set():
    ""
    from unit.certification.program_check import programs as P
    from unit.mapping.residency.executed import vehicles as V
    from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping

    spiking = ("streamed_lif", "windowed_lif", "synchronized_lif", "cascaded_ttfs", "synchronized_ttfs")
    items = []
    for seed in range(0, 12):
        for count in (2, 3, 4):
            def fuzz(seed=seed, count=count):
                ir = P.integer_ir(V.random_rep(seed))
                return {r: V.build(ir, count, None, retimed_level_stages=r) for r in (False, True)}, ir, \
                    [P.decl(s, count) for s in spiking], None
            items.append((f"fuzz_s{seed}_c{count}", fuzz))
    for seed in range(0, 6):
        for count in (2, 4):
            def wide(seed=seed, count=count):
                ir = P.integer_ir(V.wide_rep(seed), coalescing=True)
                build = lambda r: V.build(ir, count, None, retimed_level_stages=r, allow_coalescing=True,
                                          allow_neuron_splitting=True)
                return {False: build(False), True: build(True)}, ir, [P.decl(s, count) for s in spiking], None
            items.append((f"coalesce_s{seed}_c{count}", wide))
    for step in ("floor", "stateless"):
        def planned(step=step):
            program, ir = P.residency_planned(step=step)
            return {False: program}, ir, [P.decl(s, 4) for s in ("streamed_lif", "synchronized_lif")], None
        items.append((f"residency_planned_{step}", planned))

    def chain():
        program, ir = P.resident_chain()
        cores = ({"max_axons": 8, "max_neurons": 8, "count": 2},)
        return {False: program}, ir, [CheckDeclaration(semantics=s, window=P.T, pass_transfer="collapse",
                                                       core_types=cores)
                                      for s in ("synchronized_lif", "synchronized_ttfs")], None
    items.append(("resident_chain", chain))

    def value_mlp():
        program, ir = P.value_program(passes=4)
        decl = CheckDeclaration(semantics="value_integer", window=1, pass_transfer="collapse",
                                core_types=({"max_axons": 32, "max_neurons": 32, "count": 1},),
                                accumulation={"value_census": "float64", "int64_golden": "int64"})
        return {False: program}, ir, [decl], ("mlp", (8,))
    items.append(("value_mlp_4pass", value_mlp))

    def value_lenet():
        from unit.chip_simulation.value_run.integer_programs import build_integer_program, lenet5
        prog = build_integer_program(lenet5(), (1, 28, 28), num_classes=10)
        program = prog.packed_mapping(axons=1024, neurons=128, count=4, passes=4096)
        decl = CheckDeclaration(semantics="value_integer", window=1, pass_transfer="collapse",
                                core_types=({"max_axons": 1024, "max_neurons": 128, "count": 4},),
                                accumulation={"value_census": "float64", "int64_golden": "int64"})
        return {False: program}, prog.ir, [decl], ("lenet", (1, 28, 28), build_identity_hybrid_mapping(
            ir_graph=prog.ir))
    items.append(("value_lenet5_c4", value_lenet))
    return items


def _value_equal(program, identity, shape) -> dict:
    from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(-3, 4, (8, *shape), generator=gen).double()
    with torch.no_grad():
        want = ValueHybridCoreFlow(identity, dtype=torch.float64)(x)
        got = ValueHybridCoreFlow(program, dtype=torch.float64)(x)
    return {"equal": bool(torch.equal(want, got)), "max_abs": float((want - got).abs().max()),
            "gate": _gate(want, got)}


def programset_mode(out, tracer=None) -> None:
    from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
    executed: dict = {}
    limit = int(os.environ.get("W16_LIMIT", "0") or 0)
    items = _program_set()
    for label, builder in (items[:limit] + items[-3:] if limit else items):
        row: dict = {"program": label}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                programs, ir, decls, value = builder()
            reports = {}
            for decl in decls:
                program = programs[decl.windowed] if decl.windowed in programs else programs[False]
                if tracer is not None:
                    sys.settrace(tracer(executed))
                try:
                    report = check_deployment(program, ir_graph=ir, declaration=decl)
                finally:
                    sys.settrace(None)
                brief = _brief(report)
                brief.pop("check_s")
                reports[decl.semantics] = brief
            row["reports"] = reports
            if value is not None:
                program = programs[False]
                identity = value[2] if len(value) > 2 else build_identity_hybrid_mapping(ir_graph=ir)
                row["value_run"] = _value_equal(program, identity, value[1])
                row["resident_passes"] = sum(1 for s in program.stages if s.kind == "neural"
                                             and getattr(s, "schedule_weights_resident", False))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            row["trace"] = traceback.format_exc()[-1200:]
        print(json.dumps(row, sort_keys=True), file=out, flush=True)
    if tracer is not None:
        print(json.dumps({"executed": {f: sorted(v) for f, v in sorted(executed.items())}}), file=out, flush=True)


def _line_tracer(executed: dict):
    prefix = SRC + os.sep

    def local(frame, event, _arg):
        if event == "line":
            executed.setdefault(os.path.relpath(frame.f_code.co_filename, os.path.dirname(SRC)), set()).add(
                frame.f_lineno)
        return local

    def global_(frame, event, _arg):
        if frame.f_code.co_filename.startswith(prefix):
            name = os.path.relpath(frame.f_code.co_filename, os.path.dirname(SRC))
            executed.setdefault(name, set()).add(frame.f_code.co_firstlineno)
            return local
        return None
    return global_


@contextlib.contextmanager
def _d9_reseeded():
    ""
    from mimarsinan.mapping.mappers import conv_helpers
    original = conv_helpers.gather_conv_patches

    def wrong(padded_sources, kernel_size, stride, dilation, output_size):
        patches = original(padded_sources, kernel_size, stride, dilation, output_size)
        if len(kernel_size) != 1:
            return patches
        positions, channels, k = int(output_size[0]), int(padded_sources.shape[0]), int(kernel_size[0])
        return patches.reshape(positions, channels, k).transpose(1, 0, 2).reshape(positions, channels * k)

    with _patched(conv_helpers, "gather_conv_patches", wrong):
        yield


def conv1d_mode(out) -> None:
    from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from unit.chip_simulation.value_run.integer_programs import build_integer_program, conv1d_net

    row: dict = {"vehicle": "conv1d_net: Conv1d(2, 6, 3, padding=1), ReLU, Flatten, Linear(96, 5)",
                 "defect": "D9"}
    with _d9_reseeded():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                convert_torch_model(conv1d_net(), (2, 16), 5, device="cpu", packaging=MVM_PACKAGING,
                                    value_program_stage=True)
            row["conversion_probe"] = "passed"
        except Exception as exc:
            row["conversion_probe"] = f"refused {type(exc).__name__}: {str(exc)[:300]}"
    builds = {}
    for name, ctx in (("fixed", contextlib.nullcontext()), ("reseeded", _d9_reseeded())):
        with ctx, contextlib.redirect_stdout(io.StringIO()):
            builds[name] = build_integer_program(conv1d_net(), (2, 16), num_classes=5)
    x = torch.randint(-3, 4, (64, 2, 16), generator=torch.Generator().manual_seed(1)).double()
    from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
    outputs = {}
    for name, prog in builds.items():
        for label, kwargs in (("resident", dict(axons=128, neurons=16)),
                              ("scheduled_c2", dict(axons=128, neurons=16, count=2, passes=64))):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    program = prog.packed_mapping(**kwargs)
            except Exception as exc:
                row[f"{name}_{label}"] = {"build": f"{type(exc).__name__}: {str(exc)[:200]}"}
                continue
            decl = CheckDeclaration(semantics="value_integer", window=1, pass_transfer="collapse",
                                    core_types=({"max_axons": 128, "max_neurons": 16,
                                                 "count": kwargs.get("count", 999)},),
                                    accumulation={"value_census": "float64", "int64_golden": "int64"})
            report = check_deployment(program, ir_graph=prog.ir, declaration=decl)
            with torch.no_grad():
                got = ValueHybridCoreFlow(program, dtype=torch.float64)(x)
                ident = ValueHybridCoreFlow(build_identity_hybrid_mapping(ir_graph=prog.ir), dtype=torch.float64)(x)
            outputs[(name, label)] = got
            row[f"{name}_{label}"] = {"static": {k: v for k, v in _brief(report).items() if k != "statuses"},
                                      "passes": sum(1 for s in program.stages if s.kind == "neural"),
                                      "against_own_identity": _gate(ident, got)}
    for label in ("resident", "scheduled_c2"):
        if ("fixed", label) not in outputs or ("reseeded", label) not in outputs:
            continue
        want, got = outputs[("fixed", label)], outputs[("reseeded", label)]
        row[f"reseeded_against_fixed_{label}"] = {"max_abs": float((want - got).abs().max()),
                                                  "max_ref": float(want.abs().max()), "gate": _gate(want, got)}
    print(json.dumps(row), file=out, flush=True)


def residency_mode(out) -> None:
    ""
    from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
    from unit.chip_simulation.value_run.integer_programs import build_integer_program, conv1d_net, lenet5
    vehicles = (("conv1d_net", conv1d_net, (2, 16), 5, 3, ((128, 3, 2), (128, 3, 4), (128, 6, 2))),
                ("lenet5", lenet5, (1, 28, 28), 10, 4, ((1024, 4, 2), (1024, 4, 4), (1024, 8, 4))),
                ("lenet5", lenet5, (1, 28, 28), 10, None, ((1024, 128, 4),)))
    for label, model, shape, classes, max_neurons, packs in vehicles:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                prog = build_integer_program(model(), shape, num_classes=classes, max_neurons=max_neurons)
            identity = build_identity_hybrid_mapping(ir_graph=prog.ir)
        except Exception as exc:
            print(json.dumps({"vehicle": label, "ir_max_neurons": max_neurons,
                              "ir_build": f"{type(exc).__name__}: {str(exc)[:200]}"}), file=out, flush=True)
            continue
        for axons, neurons, count in packs:
            row: dict = {"vehicle": f"{label} integer value program", "ir_max_neurons": max_neurons,
                         "axons": axons, "neurons": neurons, "count": count}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    program = prog.packed_mapping(axons=axons, neurons=neurons, count=count, passes=1 << 16)
            except Exception as exc:
                row["build"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                print(json.dumps(row), file=out, flush=True)
                continue
            neural = [s for s in program.stages if s.kind == "neural"]
            resident = [s for s in neural if getattr(s, "schedule_weights_resident", False)]
            distinct = sum(1 for s in resident if len({c.core_matrix_key() for c in s.hard_core_mapping.cores}) > 1)
            decl = CheckDeclaration(semantics="value_integer", window=1, pass_transfer="collapse",
                                    core_types=({"max_axons": axons, "max_neurons": neurons, "count": count},),
                                    accumulation={"value_census": "float64", "int64_golden": "int64"})
            report = check_deployment(program, ir_graph=prog.ir, declaration=decl)
            row.update(passes=len(neural), resident_passes=len(resident),
                       resident_passes_with_distinct_ordinals=distinct,
                       static={k: v for k, v in _brief(report).items() if k != "statuses"},
                       value_run=_value_equal(program, identity, shape))
            print(json.dumps(row), file=out, flush=True)


def main() -> int:
    mode, dst = sys.argv[1], sys.argv[2]
    with open(dst, "a") as out:
        print(json.dumps({"mode": mode, "argv": sys.argv[1:], "src": SRC, "tests": TESTS,
                          "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), file=out, flush=True)
        if mode == "arms":
            arms_mode(out, sys.argv[3], sys.argv[4].split(","))
        elif mode == "reseed":
            reseed_mode(out, sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6].split(","))
        elif mode == "fuzz":
            fuzz_mode(out, int(sys.argv[3]), int(sys.argv[4]), [int(c) for c in sys.argv[5].split(",")],
                      sys.argv[6], sys.argv[7].split(","))
        elif mode == "programset":
            programset_mode(out)
        elif mode == "trace":
            programset_mode(out, tracer=_line_tracer)
        elif mode == "conv1d":
            conv1d_mode(out)
        elif mode == "residency":
            residency_mode(out)
        else:
            raise SystemExit(f"unknown mode {mode!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
