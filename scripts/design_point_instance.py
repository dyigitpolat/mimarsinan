#!/usr/bin/env python
"""A design-point instance: one network deployed on the chip rows of a spec, each row built, certified on two
executors, censused per operator, timed per pass with its write events, judged for lifetime and K*, priced on the
exploration plane, or refused by name; written as one document (``deployment_record.design_point.schema``).

    python scripts/design_point_instance.py --spec SPEC.json --parts DIR --row NAME     # one row -> DIR/NAME.json
    python scripts/design_point_instance.py --spec SPEC.json --parts DIR --out DOC.json  # assemble and validate

The spec names the network (a scaling-ladder roster name, or a roster entry it declares), the budget fraction of the
packed tile count F, the declared batch and the batch grid, the integer inputs, the per-point wall budget, the
override blocks (host, backing stores) by path, and the rows. A row names a shipped target bundle with its declared
core fields and backing-store variant, or declares core types (role, the bundle each takes its geometry and write
technology from, an operand declaration); a declaration the toolchain does not build refuses by name before
anything is built. Per judged row: the ladder's priced point at the budget (the exploration plane, one variant per
override-block set, each batch of the grid), the integer program built the pipeline's way on the target's geometry
and permissions (random weights from the roster's factory), the program the pipeline's builder schedules at the
budget (U1), and the instance machinery's judged fields on it. Every number is a single run; provisional.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
sys.path.insert(0, HERE)

import scaling_ladder as ladder  # noqa: E402

from mimarsinan.deployment_record.design_point.chip_rows import core_type_record, declaration_refusals  # noqa: E402
from mimarsinan.deployment_record.design_point.executors import EXECUTORS  # noqa: E402
from mimarsinan.deployment_record.design_point.pricing import (  # noqa: E402
    candidate_facts, executed_facts, ladder_pricing,
)
from mimarsinan.deployment_record.design_point.rows import judged_row, plane_agreement  # noqa: E402
from mimarsinan.deployment_record.design_point.schema import SCHEMA_ID, SCHEMA_VERSION, validate_instance  # noqa: E402
from mimarsinan.deployment_record.platform_physics import PlatformPhysics  # noqa: E402
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks  # noqa: E402
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params  # noqa: E402
import mimarsinan.mapping.platform.target_bundles as bundle_module  # noqa: E402
from mimarsinan.mapping.platform.target_bundles import declared_departures, target_bundles  # noqa: E402
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline  # noqa: E402
from unit.chip_simulation.value_run.integer_programs import build_integer_program, integer_inputs  # noqa: E402

#: Declared on every built program beside the bundle's keys: the 8-bit value grid of the integer readout law.
BUILD_DECLARATIONS = {"activation_bits": 8, "core_semantics": "mvm"}
EVIDENCE = {"MEASURED": "toolchain output of a run on the named worktree (single run)",
            "ESTIMATE": "priced from declared constants and the measured census (a band per constant)",
            "SECONDARY": "a published constant read in a secondary source (NeuRRAM's endurance band)",
            "DECLARED": "an assumption of this spec (rate, floor, budget fraction, batch, backing-store choice)"}


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def register_network(spec: Mapping[str, Any]) -> str:
    """The ladder roster name of the spec's network; a spec that declares its roster entry adds it (for this run)."""
    network = spec["network"]
    entry = network.get("roster_entry")
    if entry is not None:
        builder, config, classes, shape = entry
        ladder.ROSTER[network["name"]] = (builder, dict(config), int(classes), tuple(shape))
    return str(network.get("roster", network["name"]))


def model_factory(spec: Mapping[str, Any]):
    """The roster entry's model factory (the ladder's: the same random weights for the ladder and the build)."""
    builder, config, classes = ladder.ROSTER[register_network(spec)][:3]
    shape = ladder.member_shape(register_network(spec), None)
    return lambda: ladder._builder(builder)(shape, classes, config).eval(), shape, int(classes)


def blocks_of(spec: Mapping[str, Any], names: List[str]) -> Optional[Dict[str, Any]]:
    """The merged override value of the host block and the named backing-store blocks."""
    blocks: Dict[str, Any] = {}
    if spec.get("host_block"):
        blocks["host"] = load_json(spec["host_block"]).get("platform_physics_overrides", load_json(spec["host_block"]))
    for name in names:
        data = load_json(spec["backing_store_blocks"][name])
        blocks["backing_store"] = data.get("platform_physics_overrides", data)
    return merge_override_blocks(blocks) if blocks else None


def row_variants(spec: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Label -> merged override blocks: the row's named store first, then each store printed beside it."""
    stores = [row["backing_store"]] + list(row.get("beside_stores", [])) if row.get("backing_store") else []
    if not stores:
        return {"host" if spec.get("host_block") else "shipped": blocks_of(spec, [])}
    return {("host+" if spec.get("host_block") else "") + s: blocks_of(spec, [s]) for s in stores}


def bundle_document(name: str) -> Dict[str, Any]:
    """The shipped bundle file (its display name and faithfulness note, which the loaded bundle does not keep)."""
    return load_json(os.path.join(os.path.dirname(bundle_module.__file__), "targets", f"{name}.json"))


def chip_block(spec: Mapping[str, Any], row: Mapping[str, Any], cores: Optional[int]) -> Dict[str, Any]:
    """The row's chip declaration in the instance's form."""
    bundles = target_bundles()
    if row.get("core_types"):
        types, departures = [], []
        for declared in row["core_types"]:
            bundle = bundles[declared["target"]]
            core = dict(bundle.values["cores"][0])
            role = declared.get("role", "any")
            types.append(core_type_record({**core, "role": role, "count": declared.get("count"),
                                           "write_technology": bundle.values.get("platform_physics_profile"),
                                           **({"operand": declared["operand"]} if "operand" in declared else {}),
                                           "source": f"geometry and write technology of bundle {bundle.name}"}))
            # [U-C C-5] each core type's bundle against the build's own declarations, by name
            departures += [f"{d['key']} declared ({role} core type, bundle {bundle.name})"
                           for d in declared_departures(bundle.name, BUILD_DECLARATIONS)]
        names = [ct["target"] for ct in row["core_types"]]
        return {"target": None, "display_name": row.get("display_name", " with ".join(names)), "core_types": types,
                "weight_bits": None, "faithfulness": {"class": "refused", "named": departures,
                                                      "note": row.get("note", "")},
                "claim_class": None, "arithmetic_class": row["arithmetic_class"], "backing_store": None,
                "physics_profile": None}
    bundle = bundles[row["target"]]
    document = bundle_document(bundle.name)
    values = bundle.values
    core = dict(values["cores"][0])
    fields = dict(row.get("core_fields", {}))
    store = row.get("backing_store")
    # [U-C C-5] the build's own declarations and the spec's core fields, where they depart from the bundle, by name
    departed = bundle.faithfulness.with_departures(
        f"{d['key']} declared" for d in declared_departures(bundle.name, BUILD_DECLARATIONS, fields))
    return {
        "target": bundle.name, "display_name": str(document.get("display_name", bundle.name)),
        "core_types": [core_type_record({**core, **fields, "role": "any", "count": cores,
                                         "write_technology": values.get("platform_physics_profile"),
                                         "source": f"bundle {bundle.name}" + (
                                             f"; core fields {fields} declared by the spec: "
                                             f"{row.get('core_fields_source', 'no source stated')}" if fields else "")})],
        "weight_bits": int(values["weight_bits"]),
        "faithfulness": {"class": departed.kind, "named": list(departed.named),
                         "note": str((document.get("faithfulness") or {}).get("note", ""))},
        "claim_class": bundle.claim_class, "arithmetic_class": row["arithmetic_class"],
        "backing_store": None if not store else {
            "variant": store, "constants": load_json(spec["backing_store_blocks"][store]),
            "chosen_by": row.get("store_chosen_by", "spec"), "beside": list(row.get("beside_stores", []))},
        "physics_profile": values.get("platform_physics_profile"),
        "permissions": {k: values[k] for k in ("allow_coalescing", "allow_neuron_splitting") if k in values},
        **({"readout": {"declared_law": str(spec.get("readout_law", "the integer program's readout law")),
                        "bundle_converter": str(row["bundle_converter"])}} if row.get("bundle_converter") else {}),
    }


def ladder_rows(spec, row, network, variants) -> List[Dict[str, Any]]:
    target, fields = row["target"], dict(row.get("core_fields", {}))
    chip = ladder.resolved_platform(target, "declared", None, None, fields)

    def platform_at(budget, packed, residency=None, overrides=None):
        plan = {} if residency is None else {"residency_plan": dict(residency)}
        return ladder.resolved_platform(target, budget, packed, plan, fields, overrides)

    return list(ladder.cell_rows(network, target, "value", chip, platform_at, budgets=[spec["budget_fraction"]],
                                 ks=spec["k_grid"], point_budget_s=spec.get("point_budget_s"), wire_census=True,
                                 variants=variants, arms=("absent",)))


def judged_part(spec: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    """Every judged field of one bundle row (or its stop by name)."""
    network = ladder.build_network(register_network(spec), None, "value")
    variants = row_variants(spec, row)
    rows = ladder_rows(spec, row, network, variants)
    named = next(iter(variants))
    at_one = next((r for r in rows if r.get("variant", named) == named and r.get("inference_batch_K") == 1), rows[0])
    if at_one.get("packed_tiles") is None or at_one.get("refusal") or at_one.get("above_budget"):
        mark = at_one.get("refusal") or at_one.get("above_budget") or {"name": "absent", "reason": "no packed tiles"}
        return {"admission": {"verdict": "above_cap" if at_one.get("above_budget") else "refused", "at_K": None,
                              "reasons": [], "refusals": [{"name": str(mark.get("name", "refused")),
                                                           "reason": str(mark.get("reason", mark)),
                                                           "stage": str(at_one.get("stage", "ladder"))}]},
                "pricing": ladder_pricing(rows, operand_cells=0), "chip": chip_block(spec, row, None)}
    tiles = int(at_one["packed_tiles"])
    fields = dict(row.get("core_fields", {}))
    platform = {**ladder.resolved_platform(row["target"], spec["budget_fraction"], tiles, dict(BUILD_DECLARATIONS),
                                           fields, variants[named]),
                "inference_batch_K": int(spec["declared_K"]), "inference_rate_hz": ladder.RATE_HZ,
                "min_lifetime_hours": ladder.LIFETIME_HOURS}
    cores = sum(int(c.get("count", 1)) for c in platform["cores"])
    factory, shape, classes = model_factory(spec)
    pmap = resolve_platform_mapping_params(list(platform["cores"]), allow_coalescing=bool(platform["allow_coalescing"]))
    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        program = build_integer_program(factory(), shape, num_classes=classes, max_axons=pmap.effective_max_axons,
                                        max_neurons=pmap.effective_max_neurons,
                                        allow_coalescing=bool(platform["allow_coalescing"]),
                                        weight_bits=int(platform["weight_bits"]), hardware_bias=pmap.hardware_bias)
        hybrid = build_hybrid_mapping_for_pipeline(program.ir, platform)
    built = time.time() - start
    inputs = spec["inputs"]
    x_int = integer_inputs(program, int(inputs["count"]), seed=int(inputs["seed"]), spread=int(inputs.get("spread", 3)))
    names = {i: p.name for i, p in enumerate(program.model.get_perceptrons())}
    judged = judged_row(nf=program.model, ir_graph=program.ir, hybrid=hybrid, x_int=x_int, platform=platform,
                        physics=PlatformPhysics.from_dict(platform["platform_physics_resolved"]),
                        k_grid=spec["k_grid"], weight_bits=int(platform["weight_bits"]), names=names,
                        x_fluid=at_one.get("x_fluid"), x_packed=at_one.get("x_packed"),
                        write_technology=target_bundles()[row["target"]].values.get("platform_physics_profile"))
    judged["program"].update(route="build_integer_program + build_hybrid_mapping_for_pipeline (U1), integer "
                                   "readout law", build_wall_s=round(built, 1))
    judged["budget"] = {"packed_tiles_F": tiles, "fraction": float(spec["budget_fraction"]), "cores": cores,
                        "rule": "max(1, floor(fraction x F)) cores, F the packed tile count of the ladder's packer on "
                                "the target's declared geometry", "x_fluid": at_one.get("x_fluid"),
                        "x_packed": at_one.get("x_packed"), "replication": at_one.get("replication")}
    judged["pricing"] = ladder_pricing(rows, operand_cells=judged["writes"]["operand_written_cells_per_inference"])
    judged["planes"] = plane_agreement(executed_facts(judged, cores), candidate_facts(at_one))
    judged["chip"] = chip_block(spec, row, cores)
    return judged


def row_part(spec: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    start = time.time()
    head = {k: row.get(k) for k in ("row", "role", "title", "beside_of")}
    if row.get("label"):
        head["label"] = row["label"]
    if row.get("core_types"):
        refusals = declaration_refusals([{**core_type_record({**dict(target_bundles()[ct["target"]].values["cores"][0]),
                                                               "role": ct.get("role", "any")}),
                                          "write_technology": target_bundles()[ct["target"]].values.get(
                                              "platform_physics_profile"), **({"operand": ct["operand"]}
                                                                             if "operand" in ct else {})}
                                         for ct in row["core_types"]])
        part = {**head, "chip": chip_block(spec, row, None),
                "admission": {"verdict": "refused" if refusals else "admitted", "at_K": None, "reasons": [],
                              "refusals": refusals}}
        if not refusals:
            raise ladder.LadderDeclarationError("a multi-core-type row with no refusal needs per-core-type pricing, "
                                                "which this script does not build")
        return part
    work, mark = ladder.attempt(spec.get("point_budget_s"), lambda clock: judged_part(spec, row))
    if work is None:
        part = {**head, "chip": chip_block(spec, row, None),
                "admission": {"verdict": "above_cap" if (mark or {}).get("above_budget") else "refused",
                              "at_K": None, "reasons": [],
                              "refusals": [{"name": "point_stopped", "reason": json.dumps(mark, default=str)[:400]}]}}
    else:
        part = {**head, **work}
    part["wall_s"] = round(time.time() - start, 1)
    part["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)
    return part


def provenance(spec_path: str, argv: List[str]) -> Dict[str, Any]:
    root = os.path.abspath(os.path.join(HERE, ".."))
    with open(os.path.abspath(__file__), "rb") as handle:
        script_sha = hashlib.sha256(handle.read()).hexdigest()[:16]

    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", root, *args], capture_output=True, check=False)
        return done.stdout.decode(errors="replace")

    base = git("merge-base", "HEAD", "HEAD").strip()[:8] or "unknown"
    diff = git("diff", "HEAD", "--", ".", ":(exclude)nevresim", ":(exclude)datasets")
    return {"script": os.path.relpath(os.path.abspath(__file__), root), "script_sha256_16": script_sha,
            "command": " ".join(["python", "scripts/design_point_instance.py", *argv]), "worktree": root,
            "base": base, "diff_sha256_16": hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else None,
            "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}


def assemble(spec: Mapping[str, Any], parts_dir: str, spec_path: str, argv: List[str]) -> Dict[str, Any]:
    network = spec["network"]
    rows = [load_json(os.path.join(parts_dir, f"{row['row']}.json")) for row in spec["rows"]]
    return {
        "schema": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "status": "provisional",
        "produced_by": provenance(spec_path, argv),
        "network": {"name": network["name"], "builder": network["builder"], "width": network.get("width"),
                    **({"width_note": network["width_note"]} if network.get("width_note") else {}),
                    "input_shape": list(network["input_shape"]), "classes": int(network["classes"]),
                    "weights": network["weights"], "block": network.get("block"),
                    "accuracy": {"value": None, "absent": network["accuracy_absent"]}},
        "declarations": {"inference_rate_hz": ladder.RATE_HZ, "min_lifetime_hours": ladder.LIFETIME_HOURS,
                         "declared_K": int(spec["declared_K"]), "k_grid": list(spec["k_grid"]),
                         "budget_rule": f"{spec['budget_fraction']} of the packed tile count F",
                         "inputs": {"kind": "integer", **spec["inputs"]}, "atol": 0.0,
                         "point_cap": {"wall_s": spec.get("point_budget_s"), "memory": spec.get("memory")},
                         "notes": list(spec.get("notes", []))},
        "executors": [dict(e) for e in EXECUTORS], "rows": rows, "evidence_classes": dict(EVIDENCE),
        "notes": ["provisional: every number is a single run on the named worktree, to be re-run on the integrated "
                  "head"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--parts", required=True, help="directory of the per-row parts")
    parser.add_argument("--row", help="compute this row into PARTS/ROW.json")
    parser.add_argument("--out", help="assemble the parts into this document and validate it")
    args = parser.parse_args(argv)
    spec = load_json(args.spec)
    os.makedirs(args.parts, exist_ok=True)
    if args.row:
        row = next(r for r in spec["rows"] if r["row"] == args.row)
        part = row_part(spec, row)
        with open(os.path.join(args.parts, f"{args.row}.json"), "w", encoding="utf-8") as handle:
            json.dump(part, handle, indent=1, default=str)
        print(json.dumps({"row": args.row, "verdict": part["admission"]["verdict"], "wall_s": part.get("wall_s")}))
    if args.out:
        document = assemble(spec, args.parts, args.spec, argv)
        problems = validate_instance(document)
        target = args.out if not problems else args.out + ".invalid.json"
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=1, default=str)
        print(json.dumps({"out": target, "rows": len(document["rows"]), "problems": problems[:20]}))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
