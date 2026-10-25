#!/usr/bin/env python3
"""Residency plan per member (E85, G5a): the step function on toolchain units, programmed bytes, Q and carry.

Per deployment config or tier-0 template: the model at random weights (seed 0), converted and walked at the
config's platform geometry (the IR walk's layout record, wire census and program graph, what the shape-only plane
plans from), the toolchain units and live sets, and ``residency_steps`` at the config's core budget (core counts
scaled by ``--scale``, or per member by ``--member-scale name=F``). One row per step and one for the stateless plan,
each planned by the shape-only plane's program (``shape_only_program(...).compose``: the step's phases packed with the
layout packer, pinned by the one rule), which applies every refusal rule of the builder by the same predicate
(``mapping/support/residency_rules.py``: the segmentation rule, the carry rule under the pass transfer the pipeline's
builder receives, None until CL3 forwards one, the fit on the declared pool), so a row the builder refuses is
refused by name with the builder's message, on chain and non-chain graphs alike. The one check the shape-only plane
cannot run is the fit of the builder's own hard-core packing (the packers differ, pre-existing; no row of the
round-2 and round-3 sweeps differed): ``--build`` runs it.

- r; W = F - C + r, the step's planned tile writes (units are tile groups, W03); W_exec, the tiles of the cores the
  pinned rule leaves to write once each phase is packed as one pass (either side of W: joint packing saves tiles,
  packing on the declared counts can cost one);
- programmed bytes per period, defined once: the occupied-cell bytes (``params_bytes`` at ``weight_bits``) of every
  core a pass writes; beside it, on each unit's own cores (its tile group, the layer packed alone); W x TB, TB one
  full tile of the largest declared core type, is printed as a bound only;
- Q and the carried elements (both directions at every cut);
- per row, ``period_census``: S0's ``PassPeriodCensus`` of the step (``units_period_census``, each unit's own cores,
  the carry at one byte per element), labelled with the row's label below.

Index 0 is the minimum-r step (``floor``, ``steps[0]``), not the byte floor: W = F - C + r is monotone in r, but the
programmed bytes are not (a step of fewer, larger phases saves more tiles by joint packing; t0_03 at x2: index 3
writes 31% fewer bytes than the floor). Per member the script also prints the step of minimum programmed bytes; the
pricer (CL3) and the ladder (CL5) take their argmin over the census bytes, never over W.

The composition fixes the pass partition (layers per pass), r, Q and the carry on both planes; per-core cells, bytes
and pins are each plane's own packing, which agree only where the layout packer and the builder's hard-core packer
pack a phase alike. So with ``--build`` every row is also built (the executed program of that step, and the key-absent
U1 program at the same budget), and the row's bytes are the EXECUTED program's (non-pinned cores, the record's rule),
the planner's printed beside, with every per-pass difference and its cause (``schedule_residency_pass``) flagged.

Labels are keyed by (member, budget, step): ``member@scale:floor``, ``member@scale:index<i>`` (i >= 1) and
``member@scale:stateless``. A row named in ``--executed`` (optionally ``/segments`` for a step whose passes are its
segments, or ``/u1`` for a cell certified against the U1 run only) reads "executed and certified (step <rule>)" with
that qualifier; a refused row reads "refused by name: <refusal>"; every other row reads "planned and priced by the
composition the builder executes (certified on <members>, CL3)", <members> those certified against the identity run,
then apart those certified against U1 only ("executed, certified against U1 only on <members>").

usage: python scripts/residency_plan.py CONFIG.json [...] [--scale F] [--member-scale name=F,...]
       [--executed name@F:rule[/segments|/u1],...] [--build] [--json OUT]
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.noc.wire_census import census_of_walk
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.packing.schedule_residency_plan import SCHEDULE_RESIDENCY_PASS
from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
from mimarsinan.mapping.support.residency_rules import ProgramGraph
from mimarsinan.mapping.support.schedule.residency_composition import ResidencySelection, shape_only_program
from mimarsinan.mapping.support.schedule.residency_units import plan_period_facts, units_period_census
from mimarsinan.mapping.support.schedule.schedule_budget import effective_core_budget
from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline, merge_pipeline_config
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
from mimarsinan.pipelining.session import _resolve_platform_constraints
from mimarsinan.torch_mapping.converter import convert_torch_model

PROVIDER_FACTS = {"MNIST_DataProvider": ((1, 28, 28), 10), "CIFAR10_DataProvider": ((3, 32, 32), 10),
                  "ImageNet_DataProvider": ((3, 224, 224), 1000)}
#: The census sizes a carried element at one byte; the pricer sizes it in the run's domain (CL3).
CARRY_BYTES_PER_ELEMENT = 1
QUALIFIERS = {
    "": "executed and certified (step {rule})",
    "segments": "executed and certified, segment-level passes only (step {rule})",
    "u1": "executed; certified against the U1 run only (step {rule}; U1 differs from the identity run, "
          "pre-existing)",
}


def resolve(cfg: dict) -> dict:
    dp = dict(cfg["deployment_parameters"])
    dp.setdefault("seed", int(cfg.get("seed", 0)))
    DeploymentPipeline.apply_preset(cfg.get("pipeline_mode", "phased"), dp)
    pc = _resolve_platform_constraints(cfg, dp)
    config = merge_pipeline_config(dp, pc if isinstance(pc, dict) else {})
    shape, classes = PROVIDER_FACTS[cfg["data_provider_name"]]
    config.update(input_shape=shape, num_classes=classes, input_size=int(np.prod(shape)), device="cpu")
    return config


def walk(config: dict):
    """The IR, its layout record, wire census and program graph at the config's geometry (random weights, seed 0)."""
    plan = DeploymentPlan.resolve(config)
    packaging = packaging_contract_for(plan)
    placement = str(config.get("encoding_layer_placement", "subsume"))
    shape, classes = tuple(config["input_shape"]), int(config["num_classes"])
    builder = BUILDERS_REGISTRY[config["model_type"]]("cpu", shape, classes, config)
    torch.manual_seed(0)
    model = build_model(builder, config.get("model_config", {}), encoding_placement=placement,
                        packaging=packaging).eval()
    with torch.no_grad():
        model(torch.zeros(1, *shape))
    flow = model if hasattr(model, "get_mapper_repr") else convert_torch_model(
        model, shape, classes, packaging=packaging, encoding_layer_placement=placement)
    rep = flow.get_mapper_repr()
    rep.assign_perceptron_indices()
    mapping = IRMapping(q_max=1.0, firing_mode=str(config.get("firing_mode", "Default")),
                        max_axons=int(config["max_axons"]), max_neurons=int(config["max_neurons"]),
                        allow_coalescing=bool(config.get("allow_coalescing", False)),
                        hardware_bias=bool(config.get("has_bias", True)))
    mapping.collect_wire_census = True
    ir = mapping.map(rep)
    return ir, list(ir.layout_softcores), census_of_walk(mapping), ProgramGraph.of_walk(mapping)


def platform_of(config: dict, cores: list, residency_plan) -> dict:
    platform = {"cores": cores, "allow_scheduling": True,
                "allow_coalescing": bool(config.get("allow_coalescing", False)),
                "allow_neuron_splitting": bool(config.get("allow_neuron_splitting", False)),
                "max_schedule_passes": int(config.get("max_schedule_passes", 8) or 8)}
    if residency_plan is not None:
        platform["residency_plan"] = residency_plan
    return platform


def executed_row(mapping, bits: int) -> dict:
    """Per period, the executed program's programmed bytes (the cores the residency marks leave to write), W_exec, and
    every per-pass difference from the composition with its cause."""
    passes = [s for s in mapping.stages if s.kind == "neural"]
    census = [getattr(s, SCHEDULE_RESIDENCY_PASS) for s in passes]
    return {"executed_bytes": sum(params_bytes(c, bits) for p in census for i, c in enumerate(p.executed_cells)
                                  if i not in p.executed_resident),
            "executed_W_exec_tiles": sum(t for p in census for i, (_, t) in enumerate(p.executed_cores)
                                         if i not in p.executed_resident),
            "pass_differences": [{"pass": p.pass_index, "differences": list(p.differences())} for p in census
                                 if p.differences()],
            "planes_differ": any(p.plane_differences() for p in census)}


def u1_bytes(mapping, bits: int, config: dict) -> int:
    """Per period, the record's programmed bytes of the key-absent U1 program (0 on a resident pass)."""
    record = schedule_record_from_mapping(mapping, weight_bits=bits, params_reloaded=0,
                                          timesteps=int(config.get("simulation_steps", 1) or 1),
                                          pass_transfer=run_pass_transfer(config))
    return sum(int(s.params_bytes) for s in record.stages if hasattr(s, "cores"))


def member_plan(path: Path, scale: float, executed: dict, certified_on: str, build: bool) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        config = resolve(json.loads(path.read_text()))
        ir, specs, census, graph = walk(config)
    name = path.stem.split("_")[0] + "_" + path.stem.split("_")[1] if path.stem.startswith("t0_") else path.stem
    cores = [{**ct, "count": max(1, int(int(ct["count"]) * scale))} for ct in config["cores"]]
    types = core_types_from_declaration(cores)
    coalescing, splitting = (bool(config.get(key, False)) for key in ("allow_coalescing", "allow_neuron_splitting"))
    bits = int(config.get("weight_bits", 8))
    C = effective_core_budget(cores)
    tile_cells = max(t.max_axons * t.max_neurons for t in types)
    tile_bytes = params_bytes(tile_cells, bits)
    passes_cap = int(config.get("max_schedule_passes", 8) or 8)
    row = {"member": name, "scale": scale, "config": str(path), "C": C, "tile_bytes": tile_bytes, "rows": []}
    if build:
        with contextlib.redirect_stdout(io.StringIO()):
            u1 = build_hybrid_mapping_for_pipeline(ir, platform_of(config, cores, None), pipeline_config=config)
        row["u1_bytes_per_period"] = u1_bytes(u1, bits, config)
    try:
        program = shape_only_program(specs, types, census, graph,
                                     allow_coalescing=coalescing, allow_splitting=splitting,
                                     max_schedule_passes=passes_cap)
        units = program.units
        row.update({"units": list(units.tiles), "F": sum(units.tiles), "forced_cuts": list(units.forced)})
        plan = program.steps()
    except ResidencyPlanError as exc:
        row["refused"] = str(exc)
        return row

    priced = functools.partial(params_bytes, weight_bits=bits)
    selections = [("index 0", ResidencySelection("floor"))] + [
        (f"index {i}", ResidencySelection("index", i)) for i in range(1, len(plan.steps))] + [
        ("stateless", ResidencySelection("stateless"))]
    for label, selection in selections:
        key = f"{name}@{scale:g}:{selection.rule}{selection.index or ''}"
        out = {"step": label, "key": key}
        try:
            comp = program.compose(selection)
        except ResidencyPlanError as exc:
            out["refused"] = str(exc)
            out["label"] = f"refused by name: {exc}"
            row["rows"].append(out)
            continue
        step = comp.step
        layered = plan_period_facts(units.tiles, units.live, C, step, forced=units.forced, unit_cores=units.cores)
        out.update({"r": step[0], "W_tiles": comp.facts.written_tiles, "W_exec_tiles": comp.facts.written_core_tiles,
                    "planner_bytes": comp.facts.programmed_bytes(priced),
                    "programmed_bytes_unit_cores": layered.programmed_bytes(priced),
                    "full_tile_bound": comp.facts.written_tiles * tile_bytes, "Q": step[2],
                    "carry_elements": step[1]})
        out["programmed_bytes"], out["bytes_basis"] = out["planner_bytes"], "planner (layout packer)"
        if build:
            declared = {"step": selection.rule} if selection.index is None else {"step": "index",
                                                                                  "index": selection.index}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    mapping = build_hybrid_mapping_for_pipeline(ir, platform_of(config, cores, declared),
                                                                pipeline_config=config)
                out.update(executed_row(mapping, bits))
                out["programmed_bytes"], out["bytes_basis"] = out["executed_bytes"], "executed (residency marks)"
            except ResidencyPlanError as exc:
                out["build_refused"] = str(exc)
        out["label"] = (f"refused by name at build: {out['build_refused']}" if "build_refused" in out else
                        QUALIFIERS[executed[key]].format(rule=selection.rule) if key in executed else
                        f"planned and priced by the composition the builder executes (certified on {certified_on}, "
                        f"CL3)")
        out["period_census"] = units_period_census(
            units, C, step, bytes_of_cells=priced, tile_cells=tile_cells, bytes_per_element=CARRY_BYTES_PER_ELEMENT,
            label=f"{out['label']}; unit cores; carry at {CARRY_BYTES_PER_ELEMENT} B per element").to_dict()
        row["rows"].append(out)
    first = row["rows"][0]
    row["programmed_bytes_at_r_star"] = first.get("programmed_bytes")          # the minimum-r step's
    priced_rows = [r for r in row["rows"] if "programmed_bytes" in r and "build_refused" not in r]
    if priced_rows:
        least = min(priced_rows, key=lambda r: (r["programmed_bytes"], r["r"]))
        row["minimum_programmed_bytes"] = {k: least[k] for k in ("key", "step", "r", "programmed_bytes", "Q")}
    return row


def provenance(executed: dict) -> str:
    """The members a planned row cites: certified against the identity run (no qualifier, ``/segments``), then apart
    the members certified against the U1 run only (``/u1``)."""
    identity = sorted({key.split("@")[0] for key, q in executed.items() if q in ("", "segments")})
    u1 = sorted({key.split("@")[0] for key, q in executed.items() if q == "u1"} - set(identity))
    tail = f"; executed, certified against U1 only on {', '.join(u1)}" if u1 else ""
    return (", ".join(identity) or "none yet") + tail


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--member-scale", default="")
    parser.add_argument("--executed", default="")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    executed = dict(entry.partition("/")[::2] for entry in filter(None, args.executed.split(",")))
    scales = {k: float(v) for k, v in (e.split("=") for e in filter(None, args.member_scale.split(",")))}
    certified_on = provenance(executed)
    out = []
    for path in args.configs:
        stem = "_".join(path.stem.split("_")[:2]) if path.stem.startswith("t0_") else path.stem
        row = member_plan(path, scales.get(stem, args.scale), executed, certified_on, args.build)
        out.append(row)
        u1 = f", U1 {row['u1_bytes_per_period']} B/period" if "u1_bytes_per_period" in row else ""
        units = f"F {row['F']} tiles, " if "F" in row else ""
        print(f"== {row['member']} x{row['scale']:g}: {units}C {row['C']}, units {row.get('units', '-')}{u1}")
        if "refused" in row:
            print(f"   refused by name: {row['refused']}")
            continue
        for r in row["rows"]:
            if "refused" in r:
                print(f"   {r['step']:<10} {r['label']}")
                continue
            flag = ""
            if "executed_bytes" in r and r["executed_bytes"] != r["planner_bytes"]:
                flag = f"  [BYTES DIFFER: planner {r['planner_bytes']} B; per-pass causes below]"
            elif r.get("planes_differ"):
                flag = "  [cores or pins differ on the two planes, bytes equal; per-pass causes below]"
            w_exec = r.get("executed_W_exec_tiles", r["W_exec_tiles"])
            print(f"   {r['step']:<10} r {r['r']:>5}  W {r['W_tiles']:>6}  W_exec {w_exec:>6}  "
                  f"bytes {r['programmed_bytes']:>11} ({r['bytes_basis']}; unit cores "
                  f"{r['programmed_bytes_unit_cores']:>11}, full-tile bound {r['full_tile_bound']:>11})  "
                  f"Q {r['Q']:>3}  carry {r['carry_elements']:>8}  {r['label']}{flag}")
            for d in r.get("pass_differences", ()):
                for text in d["differences"]:
                    print(f"      pass {d['pass']}: {text}")
        least = row.get("minimum_programmed_bytes")
        if least:
            print(f"   minimum programmed bytes: {least['step']} (r {least['r']}, Q {least['Q']}) "
                  f"{least['programmed_bytes']} B; the minimum-r step (index 0) {row['programmed_bytes_at_r_star']} B")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
