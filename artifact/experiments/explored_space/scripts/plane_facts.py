""
from __future__ import annotations

import contextlib
import copy
import functools
import importlib.util
import io
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

WT = os.environ.get("PLANE_FACTS_WORKTREE") or sys.argv[1]
os.environ["PLANE_FACTS_WORKTREE"] = WT
sys.path.insert(0, os.path.join(WT, "src"))
sys.path.insert(0, os.path.join(WT, "tests"))
sys.path.insert(0, os.path.join(WT, "scripts"))

HERE = Path(__file__).resolve().parent


def _load_cl5_jobs():
    ""
    spec = importlib.util.spec_from_file_location("plane_facts_cl5_jobs", HERE / "cl5_jobs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


J = _load_cl5_jobs()

MEMBERS = tuple(name for name, _, _ in J.PRICED)

DEPARTURE = {"allow_coalescing": True}
DEPARTURE_LABEL = "allow_coalescing declared (the tier-0 cell's labelled departure, as CL5 prices it)"
SAMPLED_FRACTION = 0.125
PROVISIONAL = ("provisional: toolchain output of one run on the worktree named in the header; random weights; "
               "packing and value-path capabilities in flight change these numbers until the integrated head reruns")


class PairBudget(Exception):
    ""


def refusal(exc: BaseException) -> dict:
    return {"name": type(exc).__name__, "reason": str(exc).splitlines()[0][:400] if str(exc) else "",
            "where": traceback.extract_tb(exc.__traceback__)[-1].name if exc.__traceback__ else ""}


def pair_sets(bundles) -> dict:
    ""
    sets = J.bundle_sets(bundles)
    return {"targets": tuple(sets["physics"]), "spiking": tuple(sets["spiking"]), "departs": tuple(sets["no_coalescing"])}


@functools.lru_cache(maxsize=1)
def tree_pair_sets() -> dict:
    ""
    return pair_sets(J.tree_bundles(Path(WT)))


def with_departure(run, departs: bool):
    ""
    from mimarsinan.common.refusal import refusal_in_chain

    stage = {"at": None}

    def at(name):
        stage["at"] = name

    try:
        return run({}, at), {}, None
    except Exception as exc:
        if not departs or refusal_in_chain(exc) is None:
            raise
        refused = {**refusal(exc), "stage": stage["at"]}
    try:
        return run(dict(DEPARTURE), at), dict(DEPARTURE), refused
    except Exception as exc:
        exc.as_declared_refusal = refused
        raise


def skipped_pairs(previous: dict, state: dict) -> set:
    ""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import cl5_head
    if not cl5_head.same_state(previous.get("code_state"), state):
        raise SystemExit("--skip: the previous rows were written on another code state")
    return {(r["member"], r["target"]) for r in previous["rows"]}


def code_state() -> dict:
    if os.environ.get("CL5_HEAD_SET"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import cl5_head
        return cl5_head.code_state(WT)
    import hashlib
    import subprocess

    diff = subprocess.run(["git", "-C", WT, "diff", "--ignore-submodules=all", "75998141", "--", ".",
                           ":(exclude)nevresim", ":(exclude)datasets"], capture_output=True).stdout
    return {"worktree": WT, "base": "75998141", "diff_sha256_16": hashlib.sha256(diff).hexdigest()[:16]}


def ir_of(network, chip):
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.latency.ir import IRLatency
    from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params

    pmap = resolve_platform_mapping_params(list(chip["cores"]), allow_coalescing=bool(chip.get("allow_coalescing")))
    bits = int(chip.get("weight_bits") or 8)
    if hasattr(network.mapper_repr, "assign_perceptron_indices"):
        network.mapper_repr.assign_perceptron_indices()
    with contextlib.redirect_stdout(io.StringIO()):
        ir = IRMapping(q_max=float(2 ** (bits - 1) - 1), firing_mode="Default", max_axons=pmap.effective_max_axons,
                       max_neurons=pmap.effective_max_neurons, allow_coalescing=pmap.allow_coalescing,
                       hardware_bias=pmap.hardware_bias).map(network.mapper_repr)
        IRLatency(ir).calculate()
    return ir


def _payload_banks(ir) -> frozenset:
    ""
    if not getattr(ir, "weight_banks", None):
        return frozenset()
    from mimarsinan.mapping.ir.dynamic_bank import dynamic_bank_ids

    return frozenset(dynamic_bank_ids(ir))


def without_trained_scale(ir) -> None:
    ""
    import torch
    from mimarsinan.mapping.ir import NeuralCore

    dynamic = _payload_banks(ir)
    for bank_id, bank in getattr(ir, "weight_banks", {}).items():
        if bank_id not in dynamic:
            bank.parameter_scale = torch.tensor(0.0)
    for node in ir.nodes:
        if isinstance(node, NeuralCore) and not (node.has_weight_bank() and node.weight_bank_id in dynamic):
            node.parameter_scale = torch.tensor(0.0)
            if getattr(node, "bias_scale", None) is not None:
                node.bias_scale = None


def weight_pattern(ir) -> dict:
    ""
    import numpy as np
    from mimarsinan.mapping.ir import NeuralCore
    from mimarsinan.mapping.packing.softcore.sign_split import mixed_rows

    dynamic = _payload_banks(ir)
    matrices = [b.core_matrix for bank_id, b in getattr(ir, "weight_banks", {}).items() if bank_id not in dynamic]
    matrices += [n.core_matrix for n in ir.nodes
                 if isinstance(n, NeuralCore) and not n.has_weight_bank() and n.core_matrix is not None]
    arrays = [np.asarray(m) for m in matrices if m is not None and np.size(m)]
    return {"cells": sum(int(a.size) for a in arrays), "nonzero": sum(int(np.count_nonzero(a)) for a in arrays),
            "mixed_sign_rows": sum(int(mixed_rows(a).sum()) for a in arrays if a.ndim == 2)}


def executed_lanes(ir, mapper_repr, chip):
    ""
    from mimarsinan.mapping.lanes.declaration import lanes_executed

    declaration = lanes_executed(chip)
    if declaration is None:
        return ir, None
    from mimarsinan.mapping.lanes.install import lane_core_geometry
    from mimarsinan.mapping.lanes.ir.fusion import fuse_ir_lanes
    from mimarsinan.mapping.lanes.ir.reduction_banks import declared_reduction_banks
    from mimarsinan.mapping.lanes.plan import plan_lanes

    from mimarsinan.mapping.ir import ComputeOp


    payloads = [n.params.get("module") for n in ir.nodes if isinstance(n, ComputeOp) and n.params.get("module")]
    fused = copy.deepcopy(ir, {id(m): m for m in payloads})
    max_axons, max_neurons = lane_core_geometry(chip)
    fusion = fuse_ir_lanes(fused, plan_lanes(mapper_repr, declaration),
                           first_owner=len(list(mapper_repr.get_perceptrons())), max_axons=max_axons,
                           max_neurons=max_neurons, reduction_banks=declared_reduction_banks(chip))
    return fused, {"classes": sorted(declaration.classes), **fusion.as_dict()}


def executed_ir(ir, chip, *, quantize=None):
    ""
    from mimarsinan.mapping.platform.platform_constraints import PER_AXON, pool_sign_granularity

    if pool_sign_granularity(chip["cores"]) != PER_AXON:
        return ir, {"weights": "as converted (seeded float weights)"}
    if quantize is None:
        from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph as quantize
    bits = int(chip.get("weight_bits") or 8)
    built = copy.deepcopy(ir)
    without_trained_scale(built)
    quantize(built, bits, weight_quantization=True)
    converted, codes = weight_pattern(ir), weight_pattern(built)
    return built, {"weights": f"on the declared weight grid (weight_bits {bits}) by the quantizer's rule for weights "
                              "with no trained scale (one grid per perceptron at floor(q_max / peak)): the pool "
                              "declares per-axon weight sign, whose split needs integer weights",
                   "cells": codes["cells"], "codes_nonzero": codes["nonzero"],
                   "mixed_sign_rows": codes["mixed_sign_rows"], "float_nonzero": converted["nonzero"],
                   "float_mixed_sign_rows": converted["mixed_sign_rows"]}


def pair(member, target, sets=None):
    import scaling_ladder as ladder
    import mimarsinan.deployment_record.introspection.scaling_ladder as core
    from mimarsinan.deployment_record.design_point.endurance import priced_at
    from mimarsinan.deployment_record.design_point.pricing import candidate_facts, executed_facts
    from mimarsinan.deployment_record.design_point.rows import plane_agreement
    from mimarsinan.deployment_record.design_point.timeline import program_writes, writes_summary
    from mimarsinan.deployment_record.platform_physics import PlatformPhysics
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

    sets = sets or tree_pair_sets()
    packaging = "spiking" if target in sets["spiking"] else "value"
    out = {"member": member, "target": target, "packaging": packaging, "departure": None}

    def chain(declared, at):
        at("declaration")
        network = ladder.build_network(member, None, packaging)
        chip = ladder.resolved_platform(target, "declared", None, declared, {})
        at("census")
        census = core.shape_census(network, chip)
        at("ir")
        ir = ir_of(network, chip)
        at("packed_tiles")
        return network, chip, census, ir, core.packed_tiles(census, chip)

    (network, chip, census, ir, tiles), declared, refused = with_departure(chain, departs=target in sets["departs"])
    if refused is not None:
        out["as_declared_refusal"] = refused
        out["departure"] = DEPARTURE_LABEL
    out["packed_tiles_F"] = tiles
    build_ir, out["executed_ir"] = executed_ir(ir, chip)
    build_ir, lanes = executed_lanes(build_ir, network.mapper_repr, chip)
    if lanes is not None:
        out["executed_ir"]["lanes"] = lanes

    def platform_at(fraction):
        return {**ladder.resolved_platform(target, fraction, tiles, declared, {}), "inference_batch_K": 1,
                "inference_rate_hz": ladder.RATE_HZ, "min_lifetime_hours": ladder.LIFETIME_HOURS}


    fits = []
    for fraction in [f for f in ladder.BUDGET_FRACTIONS if f < 1.0]:
        try:
            platform = platform_at(fraction)
        except Exception as exc:
            fits.append({"budget": fraction, "declaration_refusal": refusal(exc)})
            continue
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                build_hybrid_mapping_for_pipeline(build_ir, {**platform, "allow_scheduling": False})
            holds = True
        except Exception as exc:
            holds, why = False, refusal(exc)
        entry = {"budget": fraction, "cores": sum(int(c["count"]) for c in platform["cores"]),
                 "single_pool_holds": holds, "wall_s": round(time.time() - t0, 1)}
        if not holds:
            entry["refusal"] = why
        fits.append(entry)
        if not holds:
            break
    out["fit_test"] = fits
    out["builder_holds_below_F"] = [f["budget"] for f in fits if f.get("single_pool_holds")]

    try:
        platform = platform_at(SAMPLED_FRACTION)
    except Exception as exc:
        out["sampled"] = {"budget": SAMPLED_FRACTION, "declaration_refusal": refusal(exc)}
        return out
    priced = core.price_program(network, census, platform, ladder.program_facts)
    point = core.ladder_row(network, census, priced, platform, ladder.READING)
    sampled = {"budget": SAMPLED_FRACTION, "cores": point["cores"], "x_packed": point.get("x_packed"),
               "schedule_reading": point.get("schedule_reading")}
    if point.get("refusal"):
        sampled["candidate_refusal"] = point["refusal"]
        out["sampled"] = sampled
        return out
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            hybrid = build_hybrid_mapping_for_pipeline(build_ir, platform)
    except Exception as exc:
        sampled.update(build_wall_s=round(time.time() - t0, 1), executed_refusal=refusal(exc),
                       candidate=candidate_facts(point),
                       plane_refusal={"name": "ExecutedRefusal", "fields": ["the executed build"],
                                      "reason": "the shape-only plane priced this point; the builder refuses it "
                                                f"({type(exc).__name__}): the W04 plane-parity class"})
        out["sampled"] = sampled
        return out
    sampled["build_wall_s"] = round(time.time() - t0, 1)
    stages = [s for s in hybrid.stages if getattr(s, "kind", None) == "neural"]
    writes = program_writes(hybrid, platform, weight_bits=int(platform["weight_bits"]))
    summary = writes_summary(writes)
    payload = platform.get("platform_physics_resolved")
    executed_refusal = "the platform declares no physics"
    if payload:
        pricing = priced_at(writes, PlatformPhysics.from_dict(payload), platform, 1)
        band = next((t.band for t in pricing.terms if t.name == "energy_write_mj" and t.band is not None), None)
        summary["energy_write_mj"] = None if band is None else [band.low, band.nominal, band.high]
        executed_refusal = next((r.reason for r in pricing.refusals if r.name == "energy_write_mj"), None)
    law = point.get("write_law") or {}
    candidate_refusal = ((law.get("refusals") or {}).get("energy_write_mj")
                         or (point.get("headline_refusals") or {}).get("energy_per_inference_mj"))
    executed = executed_facts({"program": {"neural_passes": len(stages)}, "writes": summary}, point["cores"])
    candidate = candidate_facts(point)
    executed["energy_write_mj_refusal"] = None if executed["energy_write_mj"] is not None else executed_refusal
    candidate["energy_write_mj_refusal"] = None if candidate["energy_write_mj"] is not None else candidate_refusal
    sampled["planes"] = plane_agreement(executed, candidate)
    if not sampled["planes"]["agree"]:
        sampled["plane_refusal"] = {"name": "PlaneDisagreement",
                                    "fields": [f["field"] for f in sampled["planes"]["fields"] if not f["agree"]]}
    out["sampled"] = sampled
    return out


def _alarm(_signum, _frame):
    raise PairBudget("the pair ran past its per-pair budget")


def main() -> int:
    out_path = sys.argv[2]
    argv = sys.argv[3:]
    sets = tree_pair_sets()
    members = argv[argv.index("--members") + 1].split(",") if "--members" in argv else MEMBERS
    targets = argv[argv.index("--targets") + 1].split(",") if "--targets" in argv else sets["targets"]
    budget = int(argv[argv.index("--pair-budget-s") + 1]) if "--pair-budget-s" in argv else 1800
    state, rows = code_state(), []

    done = set()
    if "--skip" in argv:
        done = skipped_pairs(json.load(open(argv[argv.index("--skip") + 1])), state)
    signal.signal(signal.SIGALRM, _alarm)
    for member in members:
        for target in targets:
            if (member, target) in done:
                continue
            t0 = time.time()
            signal.alarm(budget)
            try:
                row = pair(member, target)
            except PairBudget as exc:
                row = {"member": member, "target": target, "above_budget": {"name": "above_budget",
                                                                           "reason": f"{exc} ({budget} s)"}}
            except Exception as exc:
                row = {"member": member, "target": target, "refusal": refusal(exc)}
                if getattr(exc, "as_declared_refusal", None) is not None:
                    row.update(departure=DEPARTURE_LABEL, as_declared_refusal=exc.as_declared_refusal)
            finally:
                signal.alarm(0)
            row["wall_s"] = round(time.time() - t0, 1)
            row["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("member", "target", "builder_holds_below_F", "wall_s")}
                             | {"agree": ((row.get("sampled") or {}).get("planes") or {}).get("agree"),
                                "refusal": (row.get("refusal") or row.get("above_budget") or {}).get("name")}),
                  flush=True)
            with open(out_path, "w") as handle:
                json.dump({"class": "MEASURED pass counts, bytes and fit tests; ESTIMATE energies; single run",
                           "provisional": PROVISIONAL, "code_state": state,
                           "pair_sets": {k: list(v) for k, v in sets.items()}, "rows": rows}, handle, indent=1,
                          default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
