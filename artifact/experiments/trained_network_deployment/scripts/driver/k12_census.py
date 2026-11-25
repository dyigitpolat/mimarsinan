""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import common

HELPER = "tests/unit/mapping/residency/executed/certified.py"
CARRY = "carry per boundary"


def export_root() -> Path:
    ""
    import mimarsinan

    return Path(mimarsinan.__file__).resolve().parents[2]


def load_helper(root: Path):
    ""
    path = Path(root) / HELPER
    if not path.is_file():
        return None, path
    name = "e2e_cl3_certified"
    if name in sys.modules:
        return sys.modules[name], path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path


def _last_model(work: Path) -> str:
    stems = sorted(Path(work).glob("*.model.pt"), key=lambda p: p.stat().st_mtime)
    return stems[-1].name[: -len(".pt")]


def planes_of(helper, *args, **kwargs):
    ""
    if "candidate_refusal" in inspect.signature(helper.planes).parameters:
        kwargs["candidate_refusal"] = True
    return helper.planes(*args, **kwargs)


def _rows(helper, census) -> dict:
    return {"Q": len(census.passes), "passes": [list(map(_plain, r)) for r in helper.pass_rows(census)],
            "carry_per_boundary": {str(k): list(v) for k, v in helper.carried(census).items()},
            "programmed_bytes": int(sum(p.payload_bytes for p in census.passes)),
            "written_cores": int(sum(p.cores - p.resident_cores for p in census.passes))}


def _plain(v):
    return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)


STEP_FIELDS = ("step_rule", "step_index", "rotating_tiles", "passes", "carried_elements", "written_tiles",
               "total_tiles", "core_budget")


def composition_rows(comp) -> list[dict]:
    ""
    segment_of_pass = list(comp.pass_segments())
    rows = []
    for ordinal in sorted(set(segment_of_pass)):
        mine = [q for q, s in enumerate(segment_of_pass) if s == ordinal]
        cuts = [comp.facts.phases[q][1] for q in mine[:-1]]
        rows.append({"step_rule": _plain(comp.selection.rule), "step_index": _plain(comp.selection.step_index),
                     "rotating_tiles": int(comp.r), "passes": len(mine),
                     "carried_elements": 2 * sum(int(comp.units.live[c - 1]) for c in cuts),
                     "written_tiles": int(comp.facts.written_tiles), "total_tiles": int(comp.plan.F),
                     "core_budget": int(comp.budget)})
    return rows


def declared_grid_rows(executed) -> list[dict] | None:
    ""
    if executed is None:
        return None
    return [{"segment_index": _plain(r.get("segment_index")), **{f: _plain(r.get(f)) for f in STEP_FIELDS}}
            for r in executed]


def step_mismatches(executed, comp, view: str) -> list[str]:
    ""
    if executed is None:
        return [f"the build executed no residency plan at the declared grid, so the {view} step is compared with none"]
    census = composition_rows(comp)
    out = []
    if len(executed) != len(census):
        out.append(f"{len(executed)} executed plan segments at the declared grid against {len(census)} in the {view}")
    for i, (x, y) in enumerate(zip(executed, census)):
        for f in STEP_FIELDS:
            if x.get(f) != y[f]:
                out.append(f"segment {x.get('segment_index', i)}: {f} {x.get(f)} at the declared grid against {y[f]} "
                           f"in the {view}")
    return out


def census(work: Path, declared_plan: dict, *, root: Path | None = None, element_view: bool | None = None,
           uncertified_model: bool = False) -> dict:
    ""
    root = export_root() if root is None else Path(root)
    helper, path = load_helper(root)
    out = {"helper": HELPER, "export": str(root), "declared": dict(declared_plan)}
    if helper is None:
        out.update(passed=False, refused=f"K12 refused: CL3's helper {HELPER} is absent from the export {root} (the "
                                         "head of this run does not carry the exploration unit's W03 tests)")
        return out
    out["helper_sha256"] = common.sha256_file(path)
    try:
        from mimarsinan.deployment_record.schema.record import load_deployment_record
        from mimarsinan.pipelining.cache.load_store_strategies import (PickleLoadStoreStrategy,
                                                                       TorchModelLoadStoreStrategy)
        from mimarsinan.search.problems.joint.candidate_fragments import StageSemantics

        work = Path(work)
        record = load_deployment_record(str(work / "deployment_record.json"))
        hybrid = PickleLoadStoreStrategy("Hard Core Mapping.hard_core_mapping").load(str(work))
        ir = PickleLoadStoreStrategy("Soft Core Mapping.ir_graph").load(str(work))
        platform = dict(record.identity.platform)
        spiking = platform.get("core_semantics") != "mvm"
        timesteps = int(record.timing.s_global) if spiking else None
        model_name = "Readout Law.model" if (work / "Readout Law.model.pt").exists() else _last_model(work)
        if uncertified_model:
            import torch

            obj = torch.load(work / f"{model_name}.pt", map_location="cpu", weights_only=False)
            model = obj[0] if isinstance(obj, tuple) else obj
            out["model_loader"] = f"torch.load of {model_name}.pt (uncertified, validation only)"
        else:
            model = TorchModelLoadStoreStrategy(model_name).load(str(work))
            out["model_loader"] = f"the cache's certified loader ({model_name})"
        mapper_repr = model.get_mapper_repr()
        if hasattr(mapper_repr, "assign_perceptron_indices"):
            mapper_repr.assign_perceptron_indices()
        walk = helper.deferred_walk(mapper_repr, platform["cores"],
                                    allow_coalescing=bool(platform.get("allow_coalescing")))
        bits = None if spiking else int(platform["activation_bits"])
        view = (bits is not None and bits % 8 != 0) if element_view is None else bool(element_view)
        surface = {**platform, "activation_bits": 8} if view else platform
        planes = planes_of(helper, hybrid, ir, walk, surface, declared_plan, timesteps=timesteps,
                           semantics=StageSemantics(), sealed_record=None if view else record)
        if view:
            weight_bits = int(platform["weight_bits"])
            written, reason = helper.plane_write_census(helper.mapping_program_passes(hybrid), platform,
                                                        weight_bits=weight_bits, timesteps=timesteps,
                                                        label="record at the declared grid")
            if written is None:
                raise ValueError(f"the record plane at the declared grid: {reason}")
            sealed, reason = helper.plane_write_census(
                helper.record_program_passes(record), record.identity.platform,
                weight_bits=record.utilization.crossbar.weight_bits, timesteps=int(record.timing.s_global),
                label="sealed record")
            if sealed is None:
                raise ValueError(f"the sealed record's plane: {reason}")
            record_declared, sealed_census = written.census, sealed.census
        else:
            record_declared, sealed_census = planes.record, planes.sealed
        comp = planes.composition
        declared_step = declared_grid_rows(helper.executed_plan(hybrid))
        view_name = "element view (8-bit surface)" if view else "census at the declared grid"
        mismatch = step_mismatches(declared_step, comp, view_name)
        sealed_vs_record = None if sealed_census is None else [
            d for d in helper.differences(sealed_census, record_declared) if not d.startswith(CARRY)]
        out.update({
            "element_view": view,
            "element_view_reason": (f"activation_bits {bits} is not a whole number of bytes: every plane read at one "
                                    "byte per carried element (carries compared in elements)") if view else None,
            "declared_grid_step": declared_step, "census_step": composition_rows(comp), "census_view": view_name,
            "step_mismatch": mismatch,
            "executed_step": [_plain(v) for v in comp.step], "r": _plain(comp.r), "C": _plain(comp.budget),
            "F": _plain(comp.plan.F), "steps": [[_plain(v) for v in s] for s in comp.plan.steps],
            "stateless": [_plain(v) for v in comp.plan.stateless],
            "record_marks_equal_executed": helper.record_marks(record.schedule) == helper.executed_marks(hybrid),
            "record_plan_equal_executed": helper.record_plan(record.schedule) == helper.executed_plan(hybrid),
            "record_vs_plan": helper.differences(planes.record, planes.plan),
            "record_vs_candidate": None if planes.candidate is None else helper.differences(planes.record,
                                                                                             planes.candidate),
            "candidate_refusal": getattr(planes, "candidate_refusal", None),
            "sealed_vs_record": sealed_vs_record,
            "planes": {"record": _rows(helper, planes.record), "plan": _rows(helper, planes.plan),
                       "candidate": None if planes.candidate is None else _rows(helper, planes.candidate),
                       **({"record_declared_grid": _rows(helper, record_declared)} if view else {})},
            "pass_count": int(record.schedule.pass_count)})
    except Exception as exc:
        out.update(passed=False, refused=f"K12 refused: {type(exc).__name__}: {exc}"[:2000])
        return out
    out["census_equal"] = not out["record_vs_plan"]
    if out["step_mismatch"]:
        out.update(passed=False, refused=("K12 refused: the census composes another step than the build executed at the "
                                          f"declared grid ({out['census_view']}): " + "; ".join(out["step_mismatch"][:6]))[:2000])
        return out
    out["passed"] = bool(out["census_equal"] and out["record_marks_equal_executed"] and out["record_plan_equal_executed"]
                         and out["sealed_vs_record"] == [] and out["pass_count"] > 1)
    return out
