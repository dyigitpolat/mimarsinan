""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import pickle
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

import mimarsinan

PROFILES = ("generic_estimated_22nm", "isaac_like", "loihi", "odin", "truenorth")
_RS = "/work/research"

FROZEN_RECORDS = (
    f"{_RS}/mimarsinan_research/runs/thesis/EX-E/declared_0p1118__nsga2__s0/gen/"
    "ex_e__declared_0p1118__nsga2__s0_phased_deployment_run/deployment_record.json",
    f"{_RS}/mimarsinan_research/runs/thesis/EX-D/control_i_clean__nsga2__s0/gen/"
    "ex_d__control_i_clean__nsga2__s0_phased_deployment_run/deployment_record.json",
    f"{_RS}/earlier_study/round_2/tier1_evidence_capsule/inputs/ablation_runs/M_D/"
    "lenet_bank_clustered_n6/deployment_record.json",
    f"{_RS}/mimarsinan_research/runs/thesis/EX-0/mvm_s05__nsga2__s0/gen/"
    "ex_0__mvm_s05__nsga2__s0_phased_deployment_run/deployment_record.json",
    f"{_RS}/earlier_study/round_2/evidence/admission_runs/P5_capacity_schedule_reprogramming/"
    "P5_capacity_schedule_reprogramming_vanilla_deployment_run/deployment_record.json",
    "/archive/cold/runs/mimarsinan/t0_44_mvm_lenet5_wq_sched_pruned_phased_deployment_run/"
    "deployment_record.json",
)
_COLD = "/archive/cold/runs/mimarsinan"

FROZEN_IR = {
    "t0_41": (f"{_COLD}/t0_41_mvm_lenet5_wq_pruned_rb1_phased_deployment_run/Soft Core Mapping.ir_graph.pickle",
              "t0_41_mvm_lenet5_wq_pruned.json"),
    "t0_44": (f"{_COLD}/t0_44_mvm_lenet5_wq_sched_pruned_phased_deployment_run/Soft Core Mapping.ir_graph.pickle",
              "t0_44_mvm_lenet5_wq_sched_pruned.json"),
    "t0_53": (f"{_COLD}/t0_53_lifs_simplemlp_wq_s4_sched_phased_deployment_run/Soft Core Mapping.ir_graph.pickle",
              "t0_53_lifs_simplemlp_wq_s4_sched.json"),
}
_PINS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_identity_w1_pins.json")


def _tree_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(mimarsinan.__file__))))


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _pins() -> dict:
    with open(_PINS) as f:
        return json.load(f)


def _canon(x):
    ""
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return repr(x)
    return x


def jdigest(x) -> str:
    return hashlib.sha256(json.dumps(_canon(x), sort_keys=True).encode()).hexdigest()[:16]


def _feed(h, x, seen):
    ""
    if x is None or isinstance(x, (bool, int, str)):
        h.update(f"{type(x).__name__}:{x!r};".encode())
    elif isinstance(x, float):
        h.update(f"f:{float(x).hex()};".encode())
    elif isinstance(x, (np.integer, np.floating, np.bool_)):
        _feed(h, x.item(), seen)
    elif isinstance(x, np.ndarray):
        h.update(f"nd{x.dtype.str}{x.shape};".encode())
        if x.dtype == object:
            for e in x.flat:
                _feed(h, e, seen)
        else:
            h.update(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        h.update(f"t{t.dtype}{tuple(t.shape)};".encode())
        h.update(t.contiguous().numpy().tobytes() if t.numel() else b"")
    elif isinstance(x, (list, tuple)):
        h.update(f"{type(x).__name__}[{len(x)}];".encode())
        for e in x:
            _feed(h, e, seen)
    elif isinstance(x, (set, frozenset)):
        h.update(f"set[{len(x)}];".encode())
        for e in sorted(x, key=repr):
            _feed(h, e, seen)
    elif isinstance(x, dict):
        h.update(f"dict[{len(x)}];".encode())
        for k in sorted(x, key=repr):
            _feed(h, k, seen)
            _feed(h, x[k], seen)
    elif isinstance(x, torch.nn.Module):
        h.update(f"mod:{type(x).__name__};".encode())
        if id(x) not in seen:
            seen.add(id(x))
            for k, v in sorted(x.state_dict().items()):
                _feed(h, k, seen)
                _feed(h, v, seen)
    elif callable(x) and not hasattr(x, "__dict__"):
        h.update(f"fn:{getattr(x, '__qualname__', repr(x))};".encode())
    else:
        h.update(f"obj:{type(x).__name__};".encode())
        if id(x) in seen:
            return
        seen.add(id(x))
        state = getattr(x, "__dict__", None)
        if state is None:
            slots = [n for c in type(x).__mro__ for n in getattr(c, "__slots__", ())]
            state = {n: getattr(x, n) for n in slots if hasattr(x, n)}
        for k, v in sorted(state.items()):
            if k == "build_token":
                continue
            _feed(h, k, seen)
            _feed(h, v, seen)


def content_digest(x) -> str:
    h = hashlib.sha256()
    _feed(h, x, set())
    return h.hexdigest()[:16]


def _provider_facts():
    path = os.path.join(_tree_root(), "tests", "unit", "config_schema", "test_golden_resolution_snapshot.py")
    spec = importlib.util.spec_from_file_location("golden_resolution_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROVIDER_FACTS


def resolve(cfg: dict, facts) -> dict:
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline, merge_pipeline_config
    from mimarsinan.pipelining.session import _resolve_platform_constraints
    dp = dict(cfg["deployment_parameters"])
    dp.setdefault("seed", int(cfg.get("seed", 0)))
    DeploymentPipeline.apply_preset(cfg.get("pipeline_mode", "phased"), dp)
    pc = _resolve_platform_constraints(cfg, dp)
    config = merge_pipeline_config(dp, pc if isinstance(pc, dict) else {})
    f = facts[cfg["data_provider_name"]]
    config["input_shape"], config["num_classes"] = tuple(f.base_input_shape), int(f.num_classes)
    config["input_size"] = int(np.prod(f.base_input_shape))
    config["device"] = "cpu"
    return config


def _problem(config: dict, shared_cache: dict):
    from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
    from mimarsinan.pipelining.pipeline_steps.config.architecture_search_problem import build_search_problem
    pipeline = SimpleNamespace(config=config, data_provider_factory=None)
    problem, _names = build_search_problem(
        pipeline, search_mode="hardware",
        builder_cls=ModelRegistry.get_builder_cls(str(config["model_type"])),
        arch_options=[], model_config_assembler=lambda raw: dict(raw),
        seed=int(config.get("seed", 0) or 0),
    )
    problem._requires_fragment = lambda fragment: True
    problem._hw_only_cache = shared_cache
    return problem


def _report_fields(view) -> dict:
    quantities = view.quantities
    report = view.cost_report()
    out = {"quantities": {k: [quantities.get(k).value, quantities.get(k).provenance] for k in quantities.keys()}}
    if report is None:
        out["report"] = None
        out["refused"] = ["no report: no physics"]
    else:
        data = report.to_dict()
        out["report"] = {g: {t["name"]: t for t in data[g]} for g in ("energy", "latency", "area", "throughput")}
        out["refused"] = [n for n in data["notes"] if n.startswith("refused ")]
    return out


def candidate_rows(path: str, facts) -> list:
    cfg = json.load(open(path))
    rel = os.path.relpath(path, _tree_root())
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):
        config = resolve(cfg, facts)
    shared: dict = {}
    overlay = {"model_config": {}, "platform_constraints": {}}
    candidate = None
    for profile in PROFILES:
        t0 = time.time()
        row = {"kind": "candidate", "config": rel, "profile": profile}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                problem = _problem({**config, "platform_physics_profile": profile}, shared)
                if str(config.get("hw_config_mode", "fixed")) == "search" and candidate is None:
                    x = np.random.default_rng(0).uniform(problem.xl, problem.xu)
                    decoded = problem.decode(x)
                    candidate = {"cores": decoded["platform_constraints"]["cores"]}
                if candidate is not None:
                    overlay = {"model_config": {}, "platform_constraints": dict(candidate)}
                view = problem.candidate_layout(copy.deepcopy(overlay)).view
                row.update(_report_fields(view))
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "refused"
            row["refused"] = [f"{type(exc).__name__}: {str(exc).splitlines()[0][:240] if str(exc) else ''}"]
        if candidate is not None:
            row["candidate"] = candidate
        row["wall_s"] = round(time.time() - t0, 2)
        row["digest"] = jdigest({k: row.get(k) for k in ("status", "quantities", "report")})
        rows.append(row)
        rows.append({"kind": "refusals", "config": rel, "profile": profile,
                     "refused": row.get("refused", []), "digest": jdigest(row.get("refused", []))})
    return rows


def _with_physics(record, physics_dict):
    platform = dict(record.identity.platform)
    platform["platform_physics_resolved"] = physics_dict
    return dataclasses.replace(record, identity=dataclasses.replace(record.identity, platform=platform))


def record_rows() -> list:
    from mimarsinan.deployment_record.objectives.views import DeploymentRecordView
    from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
    from mimarsinan.deployment_record.schema import load_deployment_record
    sys.path.insert(0, os.path.join(_tree_root(), "tests"))
    from unit.deployment_record.record_fixtures import make_full_record
    pins = _pins()["records"]
    sources = [("make_full_record", make_full_record())]
    for path in FROZEN_RECORDS:
        assert _sha(path) == pins[path], f"frozen record changed: {path}"
        sources.append((path, load_deployment_record(path)))
    rows = []
    for name, record in sources:
        rows.append({"kind": "record_dict", "record": name, "digest": jdigest(record.to_dict())})
        variants = [("sealed", record)] + [
            (p, _with_physics(record, resolve_platform_physics(p, {}).to_dict())) for p in PROFILES
        ]
        for label, variant in variants:
            row = {"kind": "record", "record": name, "physics": label}
            try:
                row.update(_report_fields(DeploymentRecordView(record=variant)))
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = "refused"
                row["refused"] = [f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"]
            row["digest"] = jdigest({k: row.get(k) for k in ("status", "quantities", "report", "refused")})
            rows.append(row)
    return rows


def _hybrid_inputs(name: str, facts):
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    ir_path, template = FROZEN_IR[name]
    assert _sha(ir_path) == _pins()["ir"][ir_path], f"frozen IR changed: {ir_path}"
    cfg = json.load(open(os.path.join(_tree_root(), "templates", "tier_0", template)))
    with contextlib.redirect_stdout(io.StringIO()):
        config = resolve(cfg, facts)
    with open(ir_path, "rb") as f:
        ir = pickle.load(f)
    return ir, config, build_platform_constraints_resolved(config)


def _default_hybrid(ir, config, pcfg):
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
    with contextlib.redirect_stdout(io.StringIO()):
        return build_hybrid_mapping_for_pipeline(ir, pcfg, pipeline_config=config)


def build_hybrid(out_dir: str) -> None:
    facts = _provider_facts()
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for name in FROZEN_IR:
        ir, config, pcfg = _hybrid_inputs(name, facts)
        mapping = _default_hybrid(ir, config, pcfg)
        path = os.path.join(out_dir, f"{name}.hybrid_mapping.pickle")
        with open(path, "wb") as f:
            pickle.dump(mapping, f)
        manifest[name] = {"pickle": os.path.basename(path), "sha256": _sha(path),
                          "content_digest": content_digest(mapping), "built_by": mimarsinan.__file__}
        print(name, manifest[name]["content_digest"], flush=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)


def hybrid_rows(hybrid_dir: str) -> list:
    from mimarsinan.deployment_record.build.from_mapping import (
        placement_record_from_mapping, schedule_record_from_mapping,
    )
    from mimarsinan.mapping.weight_reuse import weight_reuse_plan_from_graph
    from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer
    facts = _provider_facts()
    manifest = json.load(open(os.path.join(hybrid_dir, "manifest.json")))
    rows = []
    for name in FROZEN_IR:
        ir, config, pcfg = _hybrid_inputs(name, facts)
        path = os.path.join(hybrid_dir, manifest[name]["pickle"])
        assert _sha(path) == manifest[name]["sha256"], f"cached hybrid mapping changed: {path}"
        with open(path, "rb") as f:
            cached = pickle.load(f)
        cached_content = content_digest(cached)
        schedule = schedule_record_from_mapping(
            cached, weight_bits=pcfg.get("weight_bits"),
            params_reloaded=int(weight_reuse_plan_from_graph(ir).params_reloaded),
            timesteps=int(config.get("simulation_steps", 32)), pass_transfer=run_pass_transfer(config),
        ).to_dict()
        placement = placement_record_from_mapping(cached).to_dict()
        row = {"kind": "hybrid", "mapping": name,
               "schedule_fields": {k: jdigest(v) for k, v in schedule.items() if k != "stages"},
               "schedule_stages": [jdigest(s) for s in schedule["stages"]],
               "placement_fields": {k: jdigest(v) for k, v in placement.items()},
               "cached_content": cached_content,
               "rebuilt_content": content_digest(_default_hybrid(ir, config, pcfg))}
        row["digest"] = jdigest({k: row[k] for k in row if k not in ("rebuilt_content",)})
        rows.append(row)
    return rows


def _key(row: dict) -> tuple:
    return tuple(row.get(k) for k in ("kind", "config", "profile", "record", "physics", "mapping"))


def compare(before_path: str, after_path: str, intended_path: str | None) -> int:
    before = {_key(r): r for r in map(json.loads, open(before_path))}
    after = {_key(r): r for r in map(json.loads, open(after_path))}
    intended = json.load(open(intended_path)) if intended_path else {}
    differing, unnamed = [], []
    for key in sorted(set(before) | set(after), key=repr):
        b, a = before.get(key), after.get(key)
        same = b is not None and a is not None and b.get("digest") == a.get("digest")
        if key[0] == "hybrid" and a is not None and b is not None:
            same = same and a["rebuilt_content"] == b["cached_content"]
        if not same:
            label = "|".join(str(k) for k in key if k is not None)
            differing.append(label)
            if label not in intended:
                unnamed.append(label)
    print(json.dumps({"rows_before": len(before), "rows_after": len(after), "differing": differing,
                      "unnamed": unnamed, "intended": sorted(intended)}, indent=1))
    return 1 if unnamed else 0


def main() -> int:
    mode = sys.argv[1]
    with open(os.path.abspath(__file__), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    stamp = {"script_sha256_16": sha, "mimarsinan_file": mimarsinan.__file__}
    if mode == "compare":
        return compare(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    if mode == "build-hybrid":
        build_hybrid(sys.argv[2])
        return 0
    out = sys.argv[2]
    if mode == "candidate":
        facts = _provider_facts()
        produce = (lambda p=p: candidate_rows(p, facts) for p in sys.argv[3:])
    elif mode == "records":
        produce = iter([record_rows])
    elif mode == "hybrid":
        produce = iter([lambda: hybrid_rows(sys.argv[3])])
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    with open(out, "a") as f:
        for job in produce:
            for row in job():
                f.write(json.dumps(_canon({**row, **stamp}), sort_keys=True) + "\n")
                f.flush()
                print(row.get("kind"), row.get("config") or row.get("record") or row.get("mapping"),
                      row.get("profile") or row.get("physics") or "", row.get("status", ""), row["digest"],
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
