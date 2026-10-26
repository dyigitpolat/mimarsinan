"""The design-point instance document: one network deployed on several chip rows, each row certified, censused,
priced and judged, or refused by name.

Workload-free: an operator is placed on a core with a role (``static`` weights written from the model, ``operand``
weights written per inference from a run-time payload) or on the host; a pass states its static writes per period
and its operand writes per inference; a core type states its role and its write technology. A convolutional network
fills the static fields only; a block with dynamic-operand products (attention) fills the operand fields with the same
schema, so the document of either is read by one reader. A row that the toolchain cannot judge today (a declaration
whose capability is not built) carries its refusals by name and only the declaration fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from mimarsinan.deployment_record.design_point.wear import wear_problems
from mimarsinan.deployment_record.platform_physics.loader import COMBINE_RULES
from mimarsinan.deployment_record.design_point.schema_types import (
    ANY, BAND, BOOL, INT, NUMBER, STR, Enum, ListOf, MapOf, Nullable, Obj, errors, json_schema,
)

SCHEMA_ID = "mimarsinan.design_point_instance"
SCHEMA_VERSION = 1
#: Placements of an operator, and the roles a core-placed operator's matrix plays.
CORE, HOST = "core", "host"
STATIC, OPERAND, HOST_ROLE, WIRING = "static", "operand", "host", "wiring"
#: Admission verdicts of a row at its declared batch.
VERDICTS = ("admitted", "inadmissible", "straddles", "refused", "above_cap")
#: The energy split every priced variant states: the parts the headline at K is the sum of (dynamic, static power over
#: the priced latency, static writes, operand writes apart from static writes, host).
ENERGY_PARTS = ("dynamic", "static", "static_write", "operand_write", "host")

REFUSAL = Obj({"name": STR, "reason": STR}, {"stage": STR})
#: A declared constant as a block states it; ``combine`` is how it composes with the chip's own value (``COMBINE_RULES``).
CONSTANT = Obj({"nominal": NUMBER, "unit": STR, "evidence_kind": STR},
               {"low": NUMBER, "high": NUMBER, "citation": STR, "note": STR, "combine": Enum(COMBINE_RULES)})
CORE_TYPE = Obj(
    {"role": Enum(("any", STATIC, OPERAND)), "max_axons": INT, "max_neurons": INT, "count": Nullable(INT),
     "write_technology": Nullable(STR)},
    {"operand": Nullable(Obj({"payload_bits": INT, "write_source": Enum(("host", "on_chip"))})),
     "write_granularity": Nullable(STR), "source": STR},
    doc="one declared core type: its role, geometry, count at the row's budget and write technology (a physics "
        "profile name)")
CHIP = Obj(
    {"target": Nullable(STR), "display_name": STR, "core_types": ListOf(CORE_TYPE), "weight_bits": Nullable(INT),
     "faithfulness": Obj({"class": STR, "named": ListOf(STR)}, {"note": STR}), "claim_class": Nullable(STR),
     "arithmetic_class": STR,
     "backing_store": Nullable(Obj({"variant": STR, "constants": MapOf(CONSTANT), "chosen_by": STR},
                                   {"beside": ListOf(STR)}))},
    {"permissions": MapOf(ANY), "physics_profile": Nullable(STR), "notes": STR,
     # [fix_b, W24] the function the row's program computes at its readout beside the converter the bundle declares
     "readout": Obj({"declared_law": STR, "bundle_converter": STR}, {"note": STR})})
ADMISSION = Obj({"verdict": Enum(VERDICTS), "at_K": Nullable(INT), "reasons": ListOf(STR),
                 "refusals": ListOf(REFUSAL)})
BUDGET = Obj({"packed_tiles_F": INT, "fraction": NUMBER, "cores": INT, "rule": STR},
             {"x_fluid": Nullable(NUMBER), "x_packed": Nullable(NUMBER), "replication": Nullable(NUMBER)})
PROGRAM = Obj({"neural_passes": INT, "host_stages": INT, "segments": INT, "hard_cores_used": INT, "ir_cores": INT,
               "host_ops": INT}, {"route": STR, "build_wall_s": NUMBER})
OPERATOR = Obj(
    {"operator": STR, "kind": STR, "placement": Enum((CORE, HOST)), "role": Enum((STATIC, OPERAND, HOST_ROLE, WIRING)),
     "reason": STR, "ir_nodes": INT, "passes": ListOf(INT)},
    {"host_stage": Nullable(INT), "as_mapped_cells": INT, "output_elements": INT, "capability_classes": STR,
     "host_boundary": STR})
TIMELINE_ENTRY = Obj(
    {"position": INT, "kind": Enum(("pass", "host"))},
    {"pass": INT, "segment_index": INT, "cores": INT, "resident_cores": INT,
     "static_written_cells_per_period": INT, "operand_written_cells_per_inference": INT,
     "payload_bytes_per_period": INT, "write_units_per_period": Nullable(INT),
     "write_events_per_period": Nullable(INT), "operand_write_events_per_inference": Nullable(INT),
     "operators": ListOf(STR), "carry_out_bytes": INT, "carry_in_bytes": INT, "operator": STR, "op_kind": STR})
EDGE = Obj({"name": STR, "between": ListOf(STR, length=2), "mismatches": INT, "compared": INT})
CERTIFICATE = Obj({"atol": NUMBER, "inputs": INT, "edges": ListOf(EDGE), "exact": BOOL, "nonvacuous": BOOL,
                   "operators_observed": INT, "operators": INT},
                  {"operand_cells": Nullable(EDGE), "notes": ListOf(STR)})
ACTIVITY = Obj({"operator": STR, "placement": Enum((CORE, HOST)), "outputs": INT, "nonzero": MapOf(INT),
                "mismatches": INT}, {"nodes_observed": INT, "nodes": INT})
CORE_TYPE_WRITES = Obj({"role": Enum(("any", STATIC, OPERAND)), "write_technology": Nullable(STR),
                        "static_written_cells_per_period": INT, "operand_written_cells_per_inference": INT,
                        "max_writes_per_cell_per_inference": Nullable(NUMBER)})
WRITES = Obj(
    {"declared_K": INT, "static_written_cells_per_period": INT, "operand_written_cells_per_inference": INT,
     "payload_bytes_per_period": INT, "written_passes": INT, "most_written_core_writes_per_period": Nullable(INT),
     "max_writes_per_cell_per_inference": Nullable(NUMBER), "write_events_per_inference": Nullable(NUMBER),
     "write_granularity": Nullable(STR), "census_label": STR},
    {"energy_write_mj": Nullable(BAND), "latency_write_exposed_s": Nullable(BAND), "refusals": ListOf(REFUSAL),
     "per_core_type": ListOf(CORE_TYPE_WRITES)})
ENDURANCE = Obj({"cycles": BAND, "evidence_kind": STR, "source": STR})
#: [fix_b] One declared core type's wear: its role and write technology, the cells it writes, its most-written cell's
#: writes per inference at the row's K, its endurance band, the lifetime hours and the verdict they give.
CORE_TYPE_LIFETIME = Obj({"role": Enum(("any", STATIC, OPERAND)), "write_technology": Nullable(STR),
                          "max_writes_per_cell_per_inference": Nullable(NUMBER), "endurance": Nullable(ENDURANCE),
                          "hours": Nullable(BAND), "verdict": Nullable(STR)}, {"notes": STR})
LIFETIME = Obj({"at_K": INT, "hours": Nullable(BAND), "verdict": Nullable(STR), "reason": STR,
                "endurance": Nullable(ENDURANCE)},
               {"refusals": ListOf(REFUSAL), "per_core_type": ListOf(CORE_TYPE_LIFETIME), "binding": Nullable(STR)})
K_VERDICT = Obj({"K": INT, "verdict": STR, "hours": Nullable(BAND), "per_corner": MapOf(STR)})
K_STAR = Obj({"per_corner": MapOf(Nullable(INT)), "basis": STR, "verdicts_by_K": ListOf(K_VERDICT),
              "boundary": STR, "checked": BOOL},
             {"no_K_admits": Nullable(STR), "min_batch": MapOf(ANY)})
PRICED_VARIANT = Obj(
    {"variant": STR, "K": INT, "energy_mj": Nullable(BAND), "energy_split_mj": MapOf(Nullable(BAND)),
     "latency_s": Nullable(BAND), "refusals": MapOf(Nullable(STR))},
    {"host_share": Nullable(NUMBER), "time_per_inference_s": Nullable(BAND), "request_latency_s": Nullable(BAND),
     "throughput_inferences_per_s": Nullable(BAND), "lifetime_verdict": Nullable(STR),
     "split_closes": Nullable(BOOL), "dynamic_shares_mj": MapOf(Nullable(BAND)),
     "energy_without_writes_mj": Nullable(BAND), "latency_without_writes_s": Nullable(BAND)},
    doc="one override-block variant at one batch K: the headlines at K (writes of a period over K included), their "
        "refusals by name, the energy split into the headline's parts and the steady headlines without writes")
PRICING = Obj({"plane": STR, "variants": ListOf(PRICED_VARIANT)}, {"refusal": Nullable(REFUSAL)})
PLANES = Obj({"agree": BOOL, "fields": ListOf(Obj({"field": STR, "executed": ANY, "candidate": ANY, "agree": BOOL}))},
             {"named": ListOf(Obj({"field": STR, "reason": STR}))})
ROW = Obj(
    {"row": STR, "role": Enum(("figure", "beside")), "title": STR, "beside_of": Nullable(STR), "chip": CHIP,
     "admission": ADMISSION},
    {"label": STR, "budget": BUDGET, "program": PROGRAM, "placement": ListOf(OPERATOR),
     "timeline": ListOf(TIMELINE_ENTRY), "certificate": CERTIFICATE, "activity": ListOf(ACTIVITY), "writes": WRITES,
     "lifetime": LIFETIME, "k_star": K_STAR, "pricing": PRICING, "planes": PLANES, "wall_s": NUMBER,
     "peak_rss_gib": NUMBER})
#: The fields every judged row (any verdict but ``refused`` and ``above_cap``) carries.
JUDGED_FIELDS = ("budget", "program", "placement", "timeline", "certificate", "activity", "writes", "lifetime",
                 "k_star")
NETWORK = Obj(
    {"name": STR, "builder": STR, "width": Nullable(NUMBER), "input_shape": ListOf(INT), "classes": INT,
     "weights": STR, "block": Nullable(STR),
     "accuracy": Obj({"value": Nullable(NUMBER), "absent": Nullable(STR)}, {"split": STR, "evidence": STR})},
    {"width_note": STR, "parameters": INT})
DECLARATIONS = Obj(
    {"inference_rate_hz": NUMBER, "min_lifetime_hours": NUMBER, "declared_K": INT, "k_grid": ListOf(INT),
     "budget_rule": STR, "inputs": Obj({"kind": STR, "count": INT, "seed": INT}, {"spread": INT}), "atol": NUMBER,
     "point_cap": Obj({"wall_s": Nullable(NUMBER), "memory": Nullable(STR)})},
    {"notes": ListOf(STR)})
INSTANCE = Obj(
    {"schema": Enum((SCHEMA_ID,)), "schema_version": Enum((SCHEMA_VERSION,)),
     "status": Enum(("provisional", "of record")),
     "produced_by": Obj({"script": STR, "command": STR, "worktree": STR, "base": STR,
                         "diff_sha256_16": Nullable(STR), "date": STR}, {"script_sha256_16": STR, "wall_s": NUMBER}),
     "network": NETWORK, "declarations": DECLARATIONS,
     "executors": ListOf(Obj({"name": STR, "role": Enum(("executor", "reference")), "description": STR})),
     "rows": ListOf(ROW)},
    {"notes": ListOf(STR), "evidence_classes": MapOf(STR)},
    doc="one network on several chip rows, each certified, censused, priced and judged, or refused by name")


def validate_instance(document: Mapping[str, Any]) -> List[str]:
    """Every violation: the closed field rules, then the row rules (a judged row carries every judged field, a
    refused row names a refusal, the energy split names each part, every executor named in a census is declared)."""
    out = errors(document, INSTANCE)
    if out:
        return out
    executors = {e["name"] for e in document["executors"]}
    for index, row in enumerate(document["rows"]):
        path = f"$.rows[{index}]"
        verdict = row["admission"]["verdict"]
        if verdict == "refused" and not row["admission"]["refusals"]:
            out.append(f"{path}: a refused row names its refusals")
        if verdict not in ("refused", "above_cap"):
            out.extend(f"{path}: a judged row carries {name!r}" for name in JUDGED_FIELDS if name not in row)
        for v, priced in enumerate(row.get("pricing", {}).get("variants", [])):
            missing = [p for p in ENERGY_PARTS if p not in priced["energy_split_mj"]]
            if missing:
                out.append(f"{path}.pricing.variants[{v}]: energy split lacks {missing}")
        for a, census in enumerate(row.get("activity", [])):
            unknown = sorted(set(census["nonzero"]) - executors)
            if unknown:
                out.append(f"{path}.activity[{a}]: undeclared executors {unknown}")
        if verdict not in ("refused", "above_cap"):
            out.extend(f"{path}: {p}" for p in wear_problems(row, document["declarations"]))
        planes = row.get("planes")
        if planes is not None and not planes["agree"]:
            named = {n["field"] for n in planes.get("named", [])}
            unnamed = [f["field"] for f in planes["fields"] if not f["agree"] and f["field"] not in named]
            if unnamed:
                out.append(f"{path}.planes: the executed and the priced program disagree on {unnamed}, unnamed")
    return out


def instance_json_schema() -> Dict[str, Any]:
    """The document rules as JSON Schema (the closed field rules; the row rules are ``validate_instance``'s)."""
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": f"{SCHEMA_ID}/v{SCHEMA_VERSION}",
            **json_schema(INSTANCE)}
