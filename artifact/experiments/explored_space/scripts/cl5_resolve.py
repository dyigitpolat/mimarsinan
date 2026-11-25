""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FORMAT_JOBS = "cl5-resolution/1"
FORMAT_DEPLOYMENTS = "cl5-deployments/1"

POINT_PREFIXES = ("p:", "c:", "i:", "m:", "mc:")

IDENTITY_FIELDS = ("architecture", "classes", "input", "weights")

NOT_OF_RECORD = ("disc",)

RANDOM_WEIGHTS = "random (torch seed 0)"
RANDOM_READ = {"from": "the roster builder at torch.manual_seed(0) (scaling_ladder.build_network): no checkpoint"}
LOADER = ("the tree's description loader: scaling_ladder.resolved_platform (with cl5_ladder's declared weight grid) "
          "or merge_pipeline_config + build_platform_constraints_resolved; lanes_executed; "
          "resolve_activation_semantics over DEFAULT_DEPLOYMENT_PARAMETERS")


def _int(v: Any) -> int:
    return int(v)


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def computation_of(pc: Mapping[str, Any], dp: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    ""
    from mimarsinan.chip_simulation.activation_semantics import resolve_activation_semantics
    from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics, resolve_core_semantics
    from mimarsinan.config_schema.defaults import DEFAULT_DEPLOYMENT_PARAMETERS, DEFAULT_PLATFORM_CONSTRAINTS
    from mimarsinan.mapping.lanes.declaration import lanes_executed

    cfg = {**DEFAULT_DEPLOYMENT_PARAMETERS, **(dp or {}), **pc}
    value = is_mvm_core_semantics(resolve_core_semantics(cfg))
    if value:
        neuron = {"core_semantics": "mvm"}
    else:
        sem = resolve_activation_semantics(cfg)
        neuron = {"core_semantics": "spiking", "spiking_family": sem.family, "spiking_variant": sem.variant,
                  "simulation_steps": _int(pc.get("simulation_steps") or DEFAULT_PLATFORM_CONSTRAINTS["simulation_steps"]),
                  "target_tq": _int(cfg.get("target_tq") or DEFAULT_PLATFORM_CONSTRAINTS["target_tq"]),
                  "firing_granularity": str(cfg.get("firing_granularity") or "per_cycle")}
    lanes = lanes_executed(pc)
    return {
        "neuron model": neuron,
        "weight grid": {"weight_bits": _int(pc.get("weight_bits") or DEFAULT_PLATFORM_CONSTRAINTS["weight_bits"])},
        "activation grid": ({"activation_bits": pc.get("activation_bits")} if value else {"spikes": True}),
        "converter law": ({"readout_law": pc.get("readout_law"), "readout_ranges": pc.get("readout_ranges")}
                          if value else {"spiking": True}),
        "operand cores": [dict(c["operand"]) for c in pc.get("cores") or [] if c.get("operand")],
        "supported classes": {"declared": pc.get("onchip_arithmetic"),
                              "executed": sorted(lanes.classes) if lanes is not None else []},
        "coalescing": {"allow_coalescing": bool(pc.get("allow_coalescing"))},
        "on-chip bias": {"has_bias": [bool(c.get("has_bias")) for c in pc.get("cores") or []]},
    }


def _widths(model) -> List[int]:
    ""
    import torch.nn as nn
    convs = [m.out_channels for m in model.modules() if isinstance(m, nn.Conv2d)]
    if convs:
        return list(dict.fromkeys(convs))[:8]
    lins = [m.out_features for m in model.modules() if isinstance(m, nn.Linear)][:-1]
    return list(dict.fromkeys(lins))[:8]


def _head(model) -> Optional[int]:
    import torch.nn as nn
    lins = [m.out_features for m in model.modules() if isinstance(m, nn.Linear)]
    return int(lins[-1]) if lins else None


def dataset_label(name: str) -> str:
    ""
    import re
    token = re.sub(r"_?DataProvider$", "", str(name)).split("_")[-1]
    m = re.fullmatch(r"([A-Za-z]+?)(\d+)", token)
    return f"{m.group(1)}-{m.group(2)}" if m else token or str(name)


def identity_of(model, *, input_shape: Sequence[int], label: str, dataset: str, weights: str, builder: str,
                config: Mapping[str, Any], dataset_words: Optional[str] = None,
                weights_read: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    ""
    shapes = [[k, list(v.shape)] for k, v in model.state_dict().items()]
    out = {"architecture": digest(shapes), "classes": _head(model), "input": [int(x) for x in input_shape],
           "weights": weights, "label": label, "widths": _widths(model), "dataset": dataset,
           "dataset_label": dataset_words or dataset, "builder": builder,
           "config": dict(config), "params": int(sum(p.numel() for p in model.parameters()))}
    if weights_read is not None:
        out["weights_read"] = dict(weights_read)
    return out


def member_weights(member: Optional[Mapping[str, Any]], net: str) -> Tuple[str, Dict[str, Any]]:
    ""
    if member is None:
        return RANDOM_WEIGHTS, dict(RANDOM_READ)
    if not member.get("k0") or not member.get("weights"):
        raise SystemExit(f"{net}: the member's network was built without its checkpoint's K0 record (cl5_ladder._builder)")
    return str(member["weights"]), {"from": "the load of the explored network (cl5_ladder._builder: the route (b) "
                                            "driver's loader.load_checkpoint, K0)", **member["k0"]}


def identity_id(identity: Mapping[str, Any]) -> str:
    return digest({f: identity.get(f) for f in IDENTITY_FIELDS})


def _opts(args: Sequence[str], name: str) -> List[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == name and i + 1 < len(args)]


def job_reading(job: Mapping[str, Any]) -> Dict[str, Any]:
    ""
    import scaling_ladder as ladder
    args = list(job["args"])
    nets, targets = _opts(args, "--networks"), _opts(args, "--targets")
    if len(nets) != 1 or len(targets) != 1 or "," in nets[0]:
        raise SystemExit(f"{job['id']}: a point job names one network and its targets ({nets}, {targets})")
    core = {}
    for item in _opts(args, "--core-field"):
        k, v = item.split("=", 1)
        try:
            core[k] = json.loads(v)
        except json.JSONDecodeError:
            core[k] = v
    res = _opts(args, "--resolution")
    return {"network": nets[0], "targets": targets[0].split(","),
            "declared": ladder.declared_values(_opts(args, "--declare")), "core_fields": core,
            "resolution": int(res[0]) if res else None, "members": _opts(args, "--member")}


def resolve_jobs(wt: Path, run_dir: Path) -> Dict[str, Any]:
    import cl5_ladder
    cl5_ladder.load_ladder(wt)
    import scaling_ladder as ladder
    import torch

    jobs, seen = [], set()
    for name in ("jobs_final.json", "jobs_census.json"):
        path = run_dir / name
        if path.exists():
            for job in json.loads(path.read_text()):
                if job["id"].startswith(POINT_PREFIXES) and job["id"] not in seen:
                    seen.add(job["id"])
                    jobs.append(job)
    out: Dict[str, Any] = {"jobs": {}, "identities": {}, "computations": {}}
    built: Dict[Tuple[str, Optional[int]], str] = {}
    for job in jobs:
        r = job_reading(job)
        for item in r["members"]:
            cl5_ladder.register_member(item)
        net = r["network"]
        key = (net, r["resolution"])
        if key not in built:
            entry = ladder.ROSTER[net]
            if entry is None:
                raise SystemExit(f"{job['id']}: {net} has no registered builder (the ladder's roster)")
            shape = ladder.member_shape(net, r["resolution"])
            torch.manual_seed(0)
            with contextlib.redirect_stdout(io.StringIO()):
                model = ladder._builder(entry[0])(shape, entry[2], entry[1])
            member = cl5_ladder.MEMBERS_REGISTERED.get(net)
            weights, read = member_weights(member, net)
            ident = identity_of(model, input_shape=shape,
                                label=member["label"] if member else net,
                                dataset=member["dataset"] if member else "random weights (torch seed 0)",
                                dataset_words=dataset_label(member["dataset"]) if member else "random weights",
                                weights=weights, builder=str(entry[0]), config=entry[1], weights_read=read)
            nid = identity_id(ident)
            out["identities"][nid] = ident
            built[key] = nid
            del model
        rows = {}
        for target in r["targets"]:
            pc = cl5_ladder.resolved_platform(target, "declared", None, r["declared"], r["core_fields"])
            comp = computation_of(pc)
            cid = digest(comp)
            out["computations"][cid] = comp
            rows[target] = cid
        out["jobs"][job["id"]] = {"network": net, "network_id": built[key], "computations": rows,
                                  "out": job["out"]}
    return out


def _driver():
    ""
    drv = HERE.parents[3] / "12_tcad_readiness" / "e2e" / "driver"
    for p in (drv, drv / "members"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import inventory
    import loader
    return inventory, loader


def of_record(result: Mapping[str, Any]) -> Tuple[bool, str]:
    ""
    if result.get("status") != "complete":
        return False, f"status {result.get('status')!r}"
    if result.get("smoke"):
        return False, "a smoke row"
    if str(result.get("budget")) in NOT_OF_RECORD:
        return False, f"budget {result.get('budget')!r}: the footprint discovery row"
    k9 = result.get("K9") or {}
    if k9.get("correct") is None or k9.get("images") is None:
        return False, "no K9 count"
    return True, ""


def certified_of(result: Mapping[str, Any]) -> Dict[str, Any]:
    ""
    static = (result.get("static_program_check") or {}).get("theorem_1_failed")
    k45 = result.get("K4_K5_pipeline") or {}
    sampled = [((k45.get(k) or {}).get("verdict")) for k in ("K4_value_parity_gate", "K5_value_twin_certificate")]
    both = static is False and all(v == "PASS" for v in sampled)
    why = ("" if both else "static part " + ("established" if static is False else
                                             "not recorded" if static is None else "failed")
           + "; sampled run " + ", ".join(str(v) for v in sampled))
    return {"static": None if static is None else not static, "sampled": sampled, "both": both, "why": why}


def deployment_weights(result: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    ""
    import cl5_ladder
    k0 = result.get("K0") or {}
    if k0.get("sha256"):
        return f"checkpoint sha256 {k0['sha256']}", {"from": "the row's K0 record (the route (b) driver's "
                                                             "loader.load_checkpoint)",
                                                    **{f: k0.get(f) for f in cl5_ladder.K0_FIELDS}}
    inputs = result.get("inputs") or {}
    return (f"checkpoint sha256 {inputs.get('checkpoint_sha256')}",
            {"from": "the row's inputs (no K0 record)", "checkpoint": inputs.get("checkpoint"),
             "sha256": inputs.get("checkpoint_sha256")})


def resolve_row(row_dir: Path, inventory, loader, cache: Dict[str, Any]) -> Dict[str, Any]:
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline, merge_pipeline_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

    cfg = json.loads((row_dir / "config.json").read_text())
    result = json.loads((row_dir / "result.json").read_text())
    ok, why = of_record(result)
    if not ok:
        return {"row": row_dir.name, "candidate": False, "why": why}
    dp = dict(cfg["deployment_parameters"])
    dp.setdefault("seed", int(cfg.get("seed", 0)))
    with contextlib.redirect_stdout(io.StringIO()):
        DeploymentPipeline.apply_preset(cfg.get("pipeline_mode", "phased"), dp)
        config = merge_pipeline_config(dp, dict(cfg.get("platform_constraints") or {}))
        pc = build_platform_constraints_resolved(config)
    comp = computation_of(pc, config)
    member = inventory.member(result["member"])
    weights, read = deployment_weights(result)
    ckey = f"{member.id}|{weights}"
    if ckey not in cache:
        import torch
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            model = loader.build_member(member)
        cache[ckey] = identity_of(model, input_shape=(3, 32, 32), label=member.label,
                                  dataset=str(cfg.get("data_provider_name")),
                                  dataset_words=dataset_label(str(cfg.get("data_provider_name"))), weights=weights,
                                  builder=member.model_type, config=member.model_config, weights_read=read)
        del model
    ident = cache[ckey]
    k9 = result["K9"]
    return {"row": row_dir.name, "candidate": True, "member": result["member"],
            "target": (cfg.get("platform_constraints") or {}).get("target"), "network_id": identity_id(ident),
            "computation_id": digest(comp), "identity": ident, "computation": comp,
            "count": {"correct": int(k9["correct"]), "images": int(k9["images"])},
            "certified": certified_of(result), "config": str(row_dir / "config.json"),
            "result": str(row_dir / "result.json")}


def resolve_deployments(roots: Sequence[Path]) -> Dict[str, Any]:
    inventory, loader = _driver()
    out: Dict[str, Any] = {"rows": {}, "not_of_record": {}, "identities": {}, "computations": {}, "roots": [],
                           "roots_absent": []}
    cache: Dict[str, Any] = {}
    for root in roots:
        if not Path(root).is_dir():
            out["roots_absent"].append(str(root))
            continue
        out["roots"].append(str(root))
        for row_dir in sorted(p for p in Path(root).iterdir() if (p / "result.json").exists()
                              and (p / "config.json").exists()):
            rec = resolve_row(row_dir, inventory, loader, cache)
            if not rec["candidate"]:
                out["not_of_record"][rec["row"]] = rec["why"]
                continue
            if rec["row"] in out["rows"]:
                continue
            out["identities"][rec["network_id"]] = rec.pop("identity")
            out["computations"][rec["computation_id"]] = rec.pop("computation")
            out["rows"][rec["row"]] = rec
    return out


def _tree(wt: Path) -> Dict[str, Any]:
    import cl5_pool
    return {"worktree": str(wt), "code_state": cl5_pool.code_state(str(wt)), "loader": LOADER}


def write(out: Path, doc: Dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("worktree", type=Path)
    j.add_argument("run_dir", type=Path)
    j.add_argument("out", type=Path)
    d = sub.add_parser("deployments")
    d.add_argument("worktree", type=Path)
    d.add_argument("out", type=Path)
    d.add_argument("--rows", type=Path, action="append", required=True)
    a = ap.parse_args(argv)
    import cl5_ladder
    cl5_ladder.load_ladder(a.worktree)
    if a.mode == "jobs":
        doc = {"format": FORMAT_JOBS, "tree": _tree(a.worktree), "run_dir": str(a.run_dir),
               **resolve_jobs(a.worktree, a.run_dir)}
        write(a.out, doc)
        print(f"resolved {len(doc['jobs'])} jobs: {len(doc['identities'])} networks, {len(doc['computations'])} "
              f"computations -> {a.out}")
    else:
        doc = {"format": FORMAT_DEPLOYMENTS, "tree": _tree(a.worktree), **resolve_deployments(a.rows)}
        write(a.out, doc)
        print(f"resolved {len(doc['rows'])} deployments of record ({len(doc['not_of_record'])} rows not of record): "
              f"{len(doc['identities'])} networks, {len(doc['computations'])} computations -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
