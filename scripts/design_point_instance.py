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

[E-6, targeted pass] A row that declares its core types with a ``declaration`` (a platform declaration file whose
core types each state their ``write_physics``, e.g. ``tests/fixtures/design_point/hybrid_neurram_sram_cim.json``) is
judged too: its platform is the declaration resolved (at the row's ``core_counts`` where stated, with the row's
``declare`` keys and each variant's override blocks), its network the row's own (``network``; a ``program``
``operand`` network is built the pipeline's way onto the declared operand core type), the IR's latencies tagged as
Soft Core Mapping tags them, its wear and lifetime per core type, and it is priced on the record plane (its executed
record per variant, ``design_point.pricing.executed_record``), since the exploration plane plans no operand
regions.
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
    candidate_facts, executed_facts, ladder_pricing, record_pricing,
)
from mimarsinan.deployment_record.design_point.pricing import executed_record  # noqa: E402
from mimarsinan.deployment_record.build.partition import compute_partition_record  # noqa: E402
from mimarsinan.config_schema.runtime import run_config  # noqa: E402
from mimarsinan.mapping.latency.ir import IRLatency  # noqa: E402
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved  # noqa: E402
from mimarsinan.deployment_record.design_point.rows import judged_row, plane_agreement  # noqa: E402
from mimarsinan.deployment_record.design_point.schema import SCHEMA_ID, SCHEMA_VERSION, validate_instance  # noqa: E402
from mimarsinan.deployment_record.platform_physics import PlatformPhysics  # noqa: E402
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks  # noqa: E402
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params  # noqa: E402
import mimarsinan.mapping.platform.target_bundles as bundle_module  # noqa: E402
from mimarsinan.mapping.platform.target_bundles import declared_departures, target_bundles  # noqa: E402
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline  # noqa: E402
from unit.chip_simulation.value_run.integer_programs import build_integer_program, integer_inputs  # noqa: E402
from unit.chip_simulation.value_run.operand_programs import LAW as OPERAND_LAW, build_operand_program  # noqa: E402

#: Declared on every built program beside the bundle's keys: the 8-bit value grid of the integer readout law.
BUILD_DECLARATIONS = {"activation_bits": 8, "core_semantics": "mvm"}
EVIDENCE = {"MEASURED": "toolchain output of a run on the named worktree (single run)",
            "ESTIMATE": "priced from declared constants and the measured census (a band per constant)",
            "SECONDARY": "a published constant read in a secondary source (NeuRRAM's endurance band)",
            "DECLARED": "an assumption of this spec (rate, floor, budget fraction, batch, backing-store choice)"}


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def register_network(spec: Mapping[str, Any], network: Optional[Mapping[str, Any]] = None) -> str:
    """The ladder roster name of the spec's network (or of ``network``, a row's own); a network that declares its
    roster entry adds it (for this run)."""
    network = spec["network"] if network is None else network
    entry = network.get("roster_entry")
    if entry is not None:
        builder, config, classes, shape = entry
        ladder.ROSTER[network["name"]] = (builder, dict(config), int(classes), tuple(shape))
    return str(network.get("roster", network["name"]))


def model_factory(spec: Mapping[str, Any], network: Optional[Mapping[str, Any]] = None):
    """The roster entry's model factory (the ladder's: the same random weights for the ladder and the build)."""
    name = register_network(spec, network)
    builder, config, classes = ladder.ROSTER[name][:3]
    shape = ladder.member_shape(name, None)
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


def declaration_of(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """[E-6, targeted pass] The row's platform declaration (its ``declaration`` file's ``platform_constraints``, a
    path relative to the worktree root), or None."""
    path = row.get("declaration")
    if not path:
        return None
    return dict(load_json(os.path.join(HERE, "..", path))["platform_constraints"])


def declared_core_types(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The row's core types in the instance's form, with their write technology: the declaration's core types
    (each with its ``write_physics``, at the row's ``core_counts`` where stated) where the row names one, else each
    named bundle's first core type under the bundle's physics profile."""
    declaration = declaration_of(row)
    out: List[Dict[str, Any]] = []
    if declaration is None:
        for declared in row["core_types"]:
            bundle = target_bundles()[declared["target"]]
            out.append({**dict(bundle.values["cores"][0]), "role": declared.get("role", "any"),
                        "count": declared.get("count"),
                        "write_technology": bundle.values.get("platform_physics_profile"),
                        **({"operand": declared["operand"]} if "operand" in declared else {}),
                        "source": f"geometry and write technology of bundle {bundle.name}"})
        return out
    cores = declaration_cores(row, declaration)
    if len(cores) != len(row["core_types"]):
        raise ladder.LadderDeclarationError(f"the row names {len(row['core_types'])} core types and its declaration "
                                            f"{row['declaration']} declares {len(cores)}")
    for declared, core in zip(row["core_types"], cores):
        out.append({**core, "role": declared.get("role", "any"),
                    "write_technology": core.get("write_physics"),
                    "source": f"core type of the declaration {row['declaration']}"})
    return out


def declaration_cores(row: Mapping[str, Any], declaration: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The declaration's core types (its target overrides), at the row's ``core_counts`` where stated."""
    cores = [dict(c) for c in declaration["target_overrides"]["cores"]["value"]]
    counts = row.get("core_counts")
    if counts is not None:
        if len(counts) != len(cores):
            raise ladder.LadderDeclarationError(f"core_counts {counts} names {len(counts)} counts for {len(cores)} "
                                                "declared core types")
        cores = [{**c, "count": int(n)} for c, n in zip(cores, counts)]
    return cores


def declared_platform(spec: Mapping[str, Any], row: Mapping[str, Any], overrides: Optional[Dict[str, Any]],
                      k: int) -> Dict[str, Any]:
    """The row's declaration resolved (``build_platform_constraints_resolved`` over the registry's run config): its
    core types at the row's counts, the row's ``declare`` keys, the instance's build declarations, the declared batch,
    rate and floor, and the variant's override blocks. [U-C C-5] A build declaration the declaration's target sets
    to another value (``declared_departures``) is a declared departure, stated in ``target_overrides`` with its
    note, so the record names it; the others are plain declarations."""
    declaration = declaration_of(row)
    assert declaration is not None
    cores = declaration_cores(row, declaration)
    overrides_block = dict(declaration.get("target_overrides") or {})
    overrides_block["cores"] = {**overrides_block["cores"], "value": cores}
    build = {k_: v for k_, v in BUILD_DECLARATIONS.items() if k_ != "core_semantics"}
    departing = {d["key"] for d in declared_departures(declaration["target"], build)} if declaration.get(
        "target") else set()
    overrides_block.update({k_: {"value": v, "note": f"the instance's build declares {k_}={v!r} (BUILD_DECLARATIONS: "
                                                     "the integer readout law's value grid)"}
                            for k_, v in build.items() if k_ in departing})
    declared = {**declaration, "target_overrides": overrides_block,
                **{k_: v for k_, v in build.items() if k_ not in departing},
                **dict(row.get("declare") or {}), "inference_batch_K": int(k), "inference_rate_hz": ladder.RATE_HZ,
                "min_lifetime_hours": ladder.LIFETIME_HOURS,
                **({"platform_physics_overrides": overrides} if overrides else {})}
    return build_platform_constraints_resolved(run_config({}, declared))


def chip_block(spec: Mapping[str, Any], row: Mapping[str, Any], cores: Optional[int],
               resolved: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The row's chip declaration in the instance's form (``resolved``: a declared row's resolved platform, once
    judged)."""
    bundles = target_bundles()
    if row.get("core_types"):
        types = [core_type_record({k: v for k, v in ct.items() if k not in ("write_physics", "compute_physics")})
                 for ct in declared_core_types(row)]
        # [U-C C-5] each core type's bundle against the build's own declarations, by name
        departures = [f"{d['key']} declared ({ct.get('role', 'any')} core type, bundle {ct['target']})"
                      for ct in row["core_types"] for d in declared_departures(ct["target"], BUILD_DECLARATIONS)]
        names = [ct["target"] for ct in row["core_types"]]
        judged = resolved is not None
        return {"target": None, "display_name": row.get("display_name", " with ".join(names)), "core_types": types,
                "weight_bits": None if not judged else int(resolved["weight_bits"]),
                "faithfulness": {"class": "refused" if not judged else str(resolved.get("target_faithfulness_class")),
                                 "named": departures, "note": row.get("note", "")},
                "claim_class": None, "arithmetic_class": row["arithmetic_class"], "backing_store": None,
                "physics_profile": None if not judged else resolved.get("platform_physics_profile")}
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


def declared_program(spec: Mapping[str, Any], row: Mapping[str, Any], platform: Mapping[str, Any]):
    """(the row's program on its resolved platform, its model, shape and classes): the row's network (or the spec's)
    built the pipeline's way; a ``program`` ``operand`` network lowers its products onto the declared operand core
    type (``build_operand_program``). The IR's latencies are tagged as Soft Core Mapping tags them (``IRLatency``)."""
    network = row.get("network") or spec["network"]
    factory, shape, classes = model_factory(spec, network)
    bits = int(platform["weight_bits"])
    if network.get("program", "integer") == "operand":
        program = build_operand_program(factory(), tuple(shape), num_classes=classes, weight_bits=bits,
                                        platform={**platform, "readout_law": dict(OPERAND_LAW),
                                                  "readout_ranges": "per_consumer_group"})
    else:
        pmap = resolve_platform_mapping_params(list(platform["cores"]),
                                               allow_coalescing=bool(platform.get("allow_coalescing", False)))
        program = build_integer_program(factory(), shape, num_classes=classes, max_axons=pmap.effective_max_axons,
                                        max_neurons=pmap.effective_max_neurons,
                                        allow_coalescing=bool(platform.get("allow_coalescing", False)),
                                        weight_bits=bits, hardware_bias=pmap.hardware_bias)
    IRLatency(program.ir).calculate()
    return program, network, tuple(shape), classes


def judged_declared_part(spec: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    """[E-6, targeted pass] Every judged field of a row declared by core types (its declaration's), priced on the
    record plane per variant at each batch of the grid."""
    variants = row_variants(spec, row)
    named = next(iter(variants))
    declared_k = int(row.get("declared_K", spec["declared_K"]))
    platform = declared_platform(spec, row, variants[named], declared_k)
    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        program, network, shape, classes = declared_program(spec, row, platform)
        hybrid = build_hybrid_mapping_for_pipeline(program.ir, platform)
    built = time.time() - start
    bits = int(platform["weight_bits"])
    partition = compute_partition_record(program.ir, program.model, shape, classes, encoding_placement="subsume")
    records = {label: executed_record(hybrid, declared_platform(spec, row, blocks, declared_k), weight_bits=bits,
                                      label=f"{row['row']}:{label}", partition=partition)
               for label, blocks in variants.items()}
    inputs = spec["inputs"]
    x_int = integer_inputs(program, int(inputs["count"]), seed=int(inputs["seed"]), spread=int(inputs.get("spread", 3)))
    names = {i: p.name for i, p in enumerate(program.model.get_perceptrons())}
    types = declared_core_types(row)
    judged = judged_row(nf=program.model, ir_graph=program.ir, hybrid=hybrid, x_int=x_int, platform=platform,
                        physics=PlatformPhysics.from_dict(platform["platform_physics_resolved"]),
                        k_grid=spec["k_grid"], weight_bits=bits, names=names,
                        roles={str(ct["write_physics"]): str(ct["role"]) for ct in types if ct.get("write_physics")})
    judged["program"].update(route=("build_operand_program" if network.get("program") == "operand"
                                    else "build_integer_program") + " + IRLatency + build_hybrid_mapping_for_pipeline "
                                   "on the row's resolved declaration, integer readout law", build_wall_s=round(built, 1))
    cores = sum(int(c.get("count", 1)) for c in platform["cores"])
    judged["budget"] = {"packed_tiles_F": None, "fraction": None, "cores": cores,
                        "rule": ("the row's declaration states its core counts (" + ", ".join(
                            f"{ct['role']} {ct.get('write_physics')}: {ct.get('count')}" for ct in types) + "); "
                                 "no footprint F: the ladder's packer plans no operand regions")}
    judged["pricing"] = record_pricing(records, spec["k_grid"])
    judged["chip"] = chip_block(spec, row, cores, resolved=platform)
    if row.get("network"):
        judged["network"] = network_block(row["network"])
    return judged


def network_block(network: Mapping[str, Any]) -> Dict[str, Any]:
    """A network declaration in the document's form (the document's network, or a row's own)."""
    return {"name": network["name"], "builder": network["builder"], "width": network.get("width"),
            **({"width_note": network["width_note"]} if network.get("width_note") else {}),
            "input_shape": list(network["input_shape"]), "classes": int(network["classes"]),
            "weights": network["weights"], "block": network.get("block"),
            "accuracy": {"value": None, "absent": network["accuracy_absent"]}}


def row_part(spec: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    start = time.time()
    head = {k: row.get(k) for k in ("row", "role", "title", "beside_of")}
    if row.get("label"):
        head["label"] = row["label"]
    judge = judged_part
    if row.get("core_types"):
        refusals = declaration_refusals(declared_core_types(row))
        if refusals or declaration_of(row) is None:
            return {**head, "chip": chip_block(spec, row, None),
                    "admission": {"verdict": "refused", "at_K": None, "reasons": [],
                                  "refusals": refusals or [{"name": "no_declaration", "stage": "declare",
                                                            "reason": "a row of core types names no declaration "
                                                                      "(a platform declaration file), so it is not "
                                                                      "built"}]}}
        judge = judged_declared_part
    work, mark = ladder.attempt(spec.get("point_budget_s"), lambda clock: judge(spec, row))
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
        "network": network_block(network),
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
