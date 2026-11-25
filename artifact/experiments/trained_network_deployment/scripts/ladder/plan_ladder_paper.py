""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import re
import resource
import signal
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
CO = DRIVER = None
T = None


def refuse(name: str, text: str):
    print(f"REFUSED: {name}: {text}", flush=True)
    raise SystemExit(2)


def paper_toolchain(co: Path, driver: Path, lock: Path) -> dict:
    ""
    if (co / ".git").exists():
        refuse("NOT_AN_EXPORT", f"{co} holds .git: the paper version's planner runs on the export of record")
    sys.path.insert(0, str(driver))
    import export_tree
    import head_names
    ver = export_tree.verify(co)
    if not ver.get("export_verified") or ver.get("trees_changed"):
        refuse("EXPORT_NOT_VERIFIED", f"{co} is off its record {export_tree.RECORD}: {ver.get('trees_changed')}")
    raw = lock.read_bytes()
    sets = head_names.lock_sets(json.loads(raw))
    names = head_names.of_sets(sets, f"the sets of {lock}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(names["head"])) or names["head"] == names["base_commit"]:
        refuse("HEAD_NAMES", f"{lock} names no form: head {names['head']}")
    if names["base_commit"] != ver["commit"]:
        refuse("HEAD_BASE", f"{lock}'s sets are cut on {names['base_commit']}, the export {co} on {ver['commit']}")
    return {"path": str(co), **ver, "head": names["head"], "base_commit": names["base_commit"],
            "head_lock": {"path": str(lock), "sha256": hashlib.sha256(raw).hexdigest()}}


def import_toolchain(co: Path, driver: Path):
    ""
    global T
    for p in (str(co / "src"), str(driver), str(driver / "members")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import types
    import torch
    import mimarsinan.mapping.verification.layout_verification_scheduling as LVS
    from mimarsinan.common.refusal import ToolchainRefusal
    from mimarsinan.config_schema.runtime import run_config
    from mimarsinan.deployment_record.introspection.scaling_ladder import (
        LadderNetwork, packed_tiles, shape_census,
    )
    from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
    from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
    from mimarsinan.mapping.platform.packaging_contract import deployed_conversion_of
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.pipelining.core.platform_constraints_resolver import (
        build_platform_constraints_resolved,
    )
    from mimarsinan.pipelining.pipeline_steps.config.torch_mapping_step import convert_pipeline_model
    T = types.SimpleNamespace(**{k: v for k, v in locals().items() if k not in ("co", "driver", "p", "types")})
    T.PLAN = LVS.plan_program_passes
    LVS.plan_program_passes = _capture
    return T


INPUT_SHAPE, CLASSES = (3, 32, 32), 100
DRIVER_FILES = ("common.py", "inventory.py", "loader.py", "members/e2e_members.py", "members/screen_models.py",
                "members/headline_models.py")


_PLANS = []


def _capture(*args, **kwargs):
    plan = T.PLAN(*args, **kwargs)
    _PLANS.append(plan)
    return plan


class AboveWall(BaseException):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_member(result: dict):
    ""
    import inventory
    import loader
    member = inventory.member(result["member"])
    path = (result.get("K0") or {}).get("checkpoint") or result["inputs"]["checkpoint"]
    digest = (result.get("K0") or {}).get("sha256") or result["inputs"]["checkpoint_sha256"]
    own = {c.path: c for c in (member.float_ckpt, *member.qat.values()) if c is not None}
    if path not in own or own[path].sha256 != digest:
        raise SystemExit(f"the row's checkpoint {path} ({digest}) is not one of {member.id}'s inventory checkpoints")
    model, _split, k0 = loader.load_checkpoint(member, own[path])
    k0 = {k: v for k, v in k0.items() if k not in ("lane_sites", "grid_owner_types", "grids_without_builder_module")}
    return model, k0, member


def flat_config(cfg: dict, count: int) -> dict:
    ""
    dp = copy.deepcopy(cfg["deployment_parameters"])
    pc = copy.deepcopy(cfg["platform_constraints"])
    cores = pc["target_overrides"]["cores"]["value"]
    if len(cores) != 1:
        raise SystemExit(f"one core type expected, the row declares {len(cores)}")
    cores[0]["count"] = int(count)
    with contextlib.redirect_stdout(io.StringIO()):
        config = dict(T.run_config(dp, pc))
    config.setdefault("input_shape", INPUT_SHAPE)
    config.setdefault("num_classes", CLASSES)
    config.setdefault("device", "cpu")
    return config


def refusal_name(error: str) -> str:
    ""
    head = (error or "").strip().split(":", 1)[0].strip()
    return head if head and " " not in head else "LadderScheduleRefusal"


def plan_point(census, platform: dict) -> dict:
    ""
    core_types = T.core_types_from_declaration(platform["cores"])
    kwargs = T.ChipCapabilities.from_platform_constraints(platform).layout_kwargs()
    planned = {} if kwargs.get("residency_plan") is None else {"program": census.program}
    specs = census.residency_softcores if planned and census.residency_softcores is not None else census.softcores
    _PLANS.clear()
    stats, error = T.LVS.compute_mapping_stats(specs, core_types, census=census.wire_census, **planned,
                                             runs=census.runs, **kwargs)
    out = {"layout_kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                             for k, v in kwargs.items()},
           "plans_built": len(_PLANS)}
    if not stats.feasible:
        out.update(verdict=refusal_name(error), refusal={"name": refusal_name(error), "reason": (error or "")[:400]})
        return out
    passes = int(stats.schedule_pass_count)
    out.update(verdict="accepted", passes=passes, sync_count=int(stats.schedule_sync_count),
               max_cores_per_pass=int(stats.max_cores_per_pass))
    plan = _PLANS[-1] if _PLANS else None
    if passes > 0:
        if plan is None or int(plan.total_pass_count) != passes:
            raise SystemExit(f"the captured plan's total {getattr(plan, 'total_pass_count', None)} is not the "
                             f"stats' schedule_pass_count {passes}")
        per = {int(k): int(v) for k, v in plan.per_segment_passes.items()}
        out.update(scheduled=True, segments=len(per), most_divided_segment_passes=max(per.values()),
                   per_segment_passes_histogram={str(v): sum(1 for x in per.values() if x == v)
                                                 for v in sorted(set(per.values()))},
                   residency_composition=plan.residency is not None)
    else:
        out.update(scheduled=False, segments=int(census.segments), most_divided_segment_passes=1,
                   unscheduled_note="the unscheduled pack holds (the fit rule): compute_mapping_stats returns the "
                                    "flat pack, schedule_pass_count 0; every segment runs once")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--row-dir", required=True)
    ap.add_argument("--F", type=int, required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--point-wall-s", type=float, default=1200.0)
    ap.add_argument("--stop-after-refusals", type=int, default=2)
    ap.add_argument("--toolchain", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    global CO, DRIVER
    CO, DRIVER = Path(args.toolchain).resolve(), Path(args.driver).resolve()
    toolchain = paper_toolchain(CO, DRIVER, Path(args.lock).resolve())
    row = Path(args.row_dir)
    cfg = json.loads((row / "config.json").read_text())
    result = json.loads((row / "result.json").read_text())
    header = {
        "format": "ladder-t1-plan/1", "key": args.key, "row": row.name, "row_dir": str(row),
        "row_config_sha256": sha256(row / "config.json"), "row_result_sha256": sha256(row / "result.json"),
        "F": args.F, "ds": args.ds, "point_wall_s": args.point_wall_s,
        "script_sha256": sha256(Path(__file__)), "toolchain": toolchain,
        "driver": {"path": str(DRIVER), "sha256": {f: sha256(DRIVER / f) for f in DRIVER_FILES if (DRIVER / f).is_file()}},
        "residency_plan": (cfg.get("platform_constraints") or {}).get("residency_plan"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if args.check_only:
        Path(args.out).write_text(json.dumps({"header": header, "points": []}, indent=1, default=str))
        print(f"wrote {args.out} (check only: {toolchain['head']} on {toolchain['base_commit']})", flush=True)
        return 0
    import_toolchain(CO, DRIVER)
    header["torch"] = T.torch.__version__
    print(json.dumps({k: header[k] for k in ("key", "row", "F", "residency_plan")}), flush=True)
    T.torch.manual_seed(0)
    model, k0, member = load_member(result)
    header["K0"] = k0
    header["K0_equals_row"] = all(k0.get(f) == (result.get("K0") or {}).get(f) for f in
                                  ("checkpoint", "sha256", "missing", "unexpected", "shape_mismatched", "module_keys",
                                   "lanes", "grids"))
    config_F = flat_config(cfg, args.F)
    plan = T.DeploymentPlan.resolve(config_F)
    conversion = T.deployed_conversion_of(plan, config_F)
    packaging = conversion.packaging
    t = time.time()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        flow = T.convert_pipeline_model(model, config_F, conversion).eval()
    header["convert_wall_s"] = time.time() - t
    platform_F = T.build_platform_constraints_resolved(config_F)
    lowering = None
    if any(ct.get("operand") for ct in platform_F["cores"]):
        from mimarsinan.mapping.support.operand_lowering import lower_contractions
        with contextlib.redirect_stdout(io.StringIO()):
            report = lower_contractions(flow.get_mapper_repr(), platform_F, owner=flow, input_shape=INPUT_SHAPE)
        lowering = {"lowered": [str(p.name) for p in report.lowered], "host": [str(p.name) for p in report.host]}
    header["operand_lowering"] = lowering
    header["packaging"] = str(getattr(packaging, "name", packaging))[:200]
    header["decompose_leaf_modules"] = bool(conversion.decompose_leaf_modules)
    network = T.LadderNetwork(name=args.key, mapper_repr=flow.get_mapper_repr(), resolution=INPUT_SHAPE,
                            total_params=float(sum(p.numel() for p in flow.parameters())))
    t = time.time()
    census = T.shape_census(network, platform_F, wire_census=True)
    F_packed = T.packed_tiles(census, platform_F)
    header.update(layout_wall_s=time.time() - t, softcores=len(census.softcores), census_segments=census.segments,
                  F_packed=F_packed, F_packed_equals_F=F_packed == args.F)
    print(json.dumps({"F_packed": F_packed, "F": args.F, "softcores": len(census.softcores),
                      "segments": census.segments, "convert_wall_s": round(header["convert_wall_s"], 1),
                      "layout_wall_s": round(header["layout_wall_s"], 1)}), flush=True)
    points, refusals_in_row = [], 0
    for token in args.ds.split(","):
        d = float(token)
        count = math.ceil(args.F / d)
        point = {"d": d if d != int(d) else int(d), "share": 1.0 / d, "cores": count}
        start = time.time()

        def expire(signum, frame):
            raise AboveWall()

        previous = signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, args.point_wall_s)
        try:
            config = flat_config(cfg, count)
            platform = T.build_platform_constraints_resolved(config)
            declared = sum(int(c.get("count", 1)) for c in platform["cores"])
            if declared != count:
                raise SystemExit(f"declared {declared} cores, wanted {count}")
            point.update(plan_point(census, platform))
        except AboveWall:
            point.update(verdict="above_wall", above_wall={"wall_s": args.point_wall_s})
        except T.ToolchainRefusal as refused:
            point.update(verdict=type(refused).__name__,
                         refusal={"name": type(refused).__name__, "reason": str(refused).splitlines()[0][:400]
                                  if str(refused) else ""})
        except ValueError as refused:
            if not type(refused).__module__.startswith("mimarsinan."):
                raise
            point.update(verdict=type(refused).__name__,
                         refusal={"name": type(refused).__name__, "reason": str(refused).splitlines()[0][:400],
                                  "toolchain_refusal": False})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)
        point["wall_s"] = round(time.time() - start, 3)
        point["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        points.append(point)
        print(json.dumps({k: point.get(k) for k in ("d", "cores", "verdict", "passes", "most_divided_segment_passes",
                                                    "segments", "wall_s")}), flush=True)
        refusals_in_row = refusals_in_row + 1 if point["verdict"] not in ("accepted", "above_wall") else 0
        if refusals_in_row >= args.stop_after_refusals:
            print(f"ladder ends: {refusals_in_row} consecutive refusals", flush=True)
            break
    header["driver_sha256_at_end"] = {f: sha256(DRIVER / f) for f in DRIVER_FILES if (DRIVER / f).is_file()}
    import export_tree
    end = export_tree.verify(CO)
    header["toolchain"]["export_verified_at_end"] = bool(end.get("export_verified")) and not end.get("trees_changed")
    header["driver_unchanged"] = header["driver_sha256_at_end"] == header["driver"]["sha256"]
    header["wall_s"] = time.time() - t0
    header["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(args.out).write_text(json.dumps({"header": header, "points": points}, indent=1, default=str))
    print(f"wrote {args.out} ({len(points)} points) in {header['wall_s']:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
