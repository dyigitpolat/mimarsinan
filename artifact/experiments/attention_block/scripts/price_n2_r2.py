""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import import_gate_checkpoint as ig
import deploy_gate as dg
from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.deployment_record.build.write_census import core_capacity_of, period_census
from mimarsinan.deployment_record.cost.absolute.writes import price_write_terms
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.resolve import (
    merge_override_blocks, resolve_platform_physics,
)
from mimarsinan.deployment_record.quantities.write_law import placed_wear, write_policy_of
from mimarsinan.mapping.capability.ladder import HOST_PAYLOAD_WRITE, capability_level
from mimarsinan.mapping.capability.pricing_inputs import level_period_census
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.operand import operand_mappers
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.platform.target_bundles import technology_blocks
from mimarsinan.mapping.platform.weight_cells import (
    CELLS_PER_WEIGHT, WEIGHT_CELL_ROWS, technology_block_errors,
)
from mimarsinan.mapping.support.operand_write import OperandWrite
from mimarsinan.models.families.vit_lite import ViTLiteCCT
from mimarsinan.torch_mapping.converter import convert_torch_model

BUNDLES = ("sram_cim_macro", "generic_estimated_22nm", "neurram", "isaac_like", "hermes_64")
SOURCES = ("host", "on_chip")
KS = (1, 16, 64)
RATE_HZ, FLOOR_H = 30.0, 43_800.0


BLOCK_REFUSAL = "weight_cell_block"

STORE_BLOCK = "backing_store"

STORE_FIELDS = ("store", "store_role", "store_block_sha256")


class StoreDeclarationRefused(Exception):
    ""

    def __init__(self, name: str, detail: str):
        super().__init__(f"{name}: {detail}")
        self.name, self.detail = name, detail


def band(b):
    return None if b is None else {"low": b.low, "nominal": b.nominal, "high": b.high, "basis": b.basis}


def technology_block(technology, blocks=None):
    ""
    entries = (technology_blocks() if blocks is None else blocks).get(technology, ())
    stated = sorted({(int(cells), int(rows)) for _description, cells, rows in entries})
    if len(stated) > 1:
        raise ValueError(f"{BLOCK_REFUSAL}: the chip descriptions binding {technology!r} state {len(stated)} weight-cell "
                         f"blocks {list(entries)}: no one block to state on the cores priced under it")
    return stated[0] if stated else None


def with_block(cores, technology, blocks=None):
    ""
    block = technology_block(technology, blocks)
    return [dict(ct) if block is None or CELLS_PER_WEIGHT in ct
            else dict(ct, **{CELLS_PER_WEIGHT: block[0], WEIGHT_CELL_ROWS: block[1]}) for ct in cores]


def block_refusals(cores, profile) -> "list[dict[str, Any]]":
    ""
    errors = technology_block_errors(list(cores), technology_blocks(), profile)
    return [{"name": BLOCK_REFUSAL, "reason": "; ".join(errors)}] if errors else []


def platform_for(cores_declared, source):
    cores = []
    for ct in cores_declared:
        ct = dict(ct, write_granularity="cell")
        if "operand" in ct:
            ct["operand"] = dict(ct["operand"], write_source=source)
        cores.append(ct)
    return {"cores": cores, "inference_rate_hz": RATE_HZ, "min_lifetime_hours": FLOOR_H}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_block(path) -> dict:
    ""
    data = json.loads(Path(path).read_text())
    return data.get("platform_physics_overrides", data)


def store_declaration(spec_path, rows, block_sha256, spec_sha256) -> dict:
    ""
    spec_path = Path(spec_path)
    got = sha256_file(spec_path)
    if got != spec_sha256:
        raise StoreDeclarationRefused("SPEC_DIGEST", f"{spec_path} is {got}, not the named {spec_sha256}")
    spec = json.loads(spec_path.read_text())
    by_row = {r.get("row"): r for r in spec.get("rows", [])}
    paths = spec.get("backing_store_blocks") or {}
    declared: dict = {}
    used = set()
    for name in rows:
        row = by_row.get(name)
        if row is None:
            raise StoreDeclarationRefused("SPEC_ROW_ABSENT", f"{spec_path} holds no row {name!r} ({sorted(by_row)})")
        bundle = row.get("target")
        if bundle not in BUNDLES:
            raise StoreDeclarationRefused("BUNDLE_NOT_PRICED", f"row {name!r} targets {bundle!r}, which this script "
                                                               f"does not price ({', '.join(BUNDLES)})")
        if bundle in declared:
            raise StoreDeclarationRefused("BUNDLE_DECLARED_TWICE", f"row {name!r} declares {bundle!r}, which row "
                                                                   f"{declared[bundle][0]['row']!r} declares too")
        if not row.get("backing_store"):
            raise StoreDeclarationRefused("ROW_DECLARES_NO_STORE", f"row {name!r} ({bundle}) names no backing_store")
        stores = [row["backing_store"]] + list(row.get("beside_stores") or [])
        if len(set(stores)) != len(stores):
            raise StoreDeclarationRefused("STORE_NAMED_TWICE", f"row {name!r} names a store twice: {stores}")
        variants = []
        for i, store in enumerate(stores):
            path = paths.get(store)
            if path is None:
                raise StoreDeclarationRefused("STORE_BLOCK_ABSENT", f"{spec_path} names no block file for store "
                                                                    f"{store!r} (backing_store_blocks)")
            want = block_sha256.get(store)
            if want is None:
                raise StoreDeclarationRefused("STORE_BLOCK_UNNAMED", f"store {store!r} ({path}) is named by no sha256 "
                                                                     f"(--block-sha256 {store}=SHA)")
            have = sha256_file(path)
            if have != want:
                raise StoreDeclarationRefused("STORE_BLOCK_DIGEST", f"store {store!r}: {path} is {have}, not the "
                                                                    f"named {want}")
            used.add(store)
            variants.append({"store": store, "role": "figure" if i == 0 else "beside", "row": name, "block": str(path),
                             "sha256": have, "blocks": {STORE_BLOCK: load_block(path)}})
        declared[bundle] = variants
    unused = sorted(set(block_sha256) - used)
    if unused:
        raise StoreDeclarationRefused("STORE_BLOCK_UNUSED", f"digests named for {unused}, no store of rows {list(rows)}")
    return declared


def declaration_record(stores, spec=None) -> "dict | None":
    ""
    if not stores:
        return None
    return {"spec": spec, "bundles": {b: [{k: v for k, v in var.items() if k != "blocks"} for var in vs]
                                      for b, vs in stores.items()}}


def physics_for(bundle, blocks=None):
    ""
    if not blocks:
        return get_platform_physics(bundle)
    return resolve_platform_physics(bundle, merge_override_blocks(blocks))


def variants_of(bundle, stores):
    ""
    declared = (stores or {}).get(bundle)
    return list(declared) if declared else [None]


def store_fields(var) -> dict:
    if var is None:
        return dict.fromkeys(STORE_FIELDS)
    return {"store": var["store"], "store_role": var["role"], "store_block_sha256": var["sha256"]}


def price_program(packed, cores_declared, product_macs: int, declarations=None, stores=None) -> list:
    ""
    if declarations is None:
        declarations = {bundle: with_block(cores_declared, bundle) for bundle in BUNDLES}
    physics_of = {(bundle, None if var is None else var["store"]): physics_for(bundle, None if var is None
                                                                                else var["blocks"])
                  for bundle in BUNDLES for var in variants_of(bundle, stores)}
    passes = mapping_program_passes(packed)
    rows = []
    for source in (SOURCES if any("operand" in c for c in cores_declared) else ("none",)):
        base = platform_for(cores_declared, source if source != "none" else "host")
        capacity = core_capacity_of(base)
        census = period_census(passes, weight_bits=8, capacity=capacity)
        for k in KS:
            for bundle in BUNDLES:
                for var in variants_of(bundle, stores):
                    tag = store_fields(var)
                    platform = dict(platform_for(declarations[bundle], source if source != "none" else "host"),
                                    inference_batch_K=k)
                    refused = block_refusals(platform["cores"], bundle)
                    if refused:
                        rows.append({"write_source": source, "K": k, "bundle": bundle, "terms": {},
                                     "energy_write_mJ": None, "exposed_write_s": None, "verdicts": [],
                                     "refusals": refused, "product_mac_energy": None,
                                     "cores_priced": platform["cores"], **tag})
                        continue
                    policy = write_policy_of(platform)
                    physics = physics_of[(bundle, tag["store"])]
                    priced = price_write_terms(census.census, physics, policy, written_extents=census.written_extents,
                                               most_written=census.most_written_core,
                                               pass_core_types=census.pass_core_types,
                                               producer_slots=census.producer_slots)
                    terms = {t.name: {"unit": t.unit, **band(t.band)} for t in priced.terms}
                    mac = None
                    if physics.has("e_mac"):
                        declared = physics.constants["e_mac"]
                        si = physics.band("e_mac")
                        mac = {"e_mac_declared": {"low": declared.low, "nominal": declared.nominal,
                                                  "high": declared.high, "unit": declared.unit},
                               "product_macs": product_macs,
                               "product_mac_energy_mJ": {"low": si.low * product_macs * 1e3,
                                                         "nominal": si.nominal * product_macs * 1e3,
                                                         "high": si.high * product_macs * 1e3}}
                    rows.append({
                        "write_source": source, "K": k, "bundle": bundle, "terms": terms,
                        "energy_write_mJ": band(priced.energy), "exposed_write_s": band(priced.exposed),
                        "verdicts": [{"name": v.name, "verdict": v.verdict, "reason": v.reason,
                                      "evidence": v.evidence} for v in priced.verdicts],
                        "refusals": [{"name": r.name, "reason": r.reason} for r in priced.refusals],
                        "product_mac_energy": mac, "cores_priced": platform["cores"], **tag,
                    })
    census_k1 = period_census(passes, weight_bits=8, capacity=core_capacity_of(platform_for(cores_declared, "host")))
    summary = {
        "producer_cells_per_inference": int(sum(p.written_cells_producer for p in census_k1.census.passes)),
        "backing_store_cells_per_period": int(sum(p.written_cells_backing_store for p in census_k1.census.passes)),
        "passes": len(census_k1.census.passes), "most_written_core_period_writes": census_k1.most_written_core,
        "producer_slots": census_k1.producer_slots,
        "wear_per_inference": {k: (None if census_k1.most_written_core is None or census_k1.producer_slots is None
                                   else placed_wear(k, census_k1.most_written_core, census_k1.producer_slots))
                               for k in KS},
    }
    return rows, summary


def perceptron_names(model_repr) -> dict:
    ""
    out = {}
    for node in model_repr.execution_order():
        index, perceptron = getattr(node, "perceptron_index", None), getattr(node, "perceptron", None)
        if index is not None and perceptron is not None:
            out[int(index)] = str(perceptron.name)
    for mapper in operand_mappers(model_repr):
        for op in mapper.banks:
            out[int(op.perceptron_index)] = str(op.name)
    return out


def timeline(packed, ir, model_repr) -> list:
    ""
    from mimarsinan.mapping.capability.op_kinds import op_kind
    names = perceptron_names(model_repr)
    by_node = {int(n.id): names.get(int(n.perceptron_index)) for n in ir.get_neural_cores()
               if n.perceptron_index is not None}
    banks = getattr(packed, "dynamic_bank_bindings", {}) or {}
    out = []
    for i, stage in enumerate(packed.stages):
        if stage.kind == "compute":
            op = stage.compute_op
            out.append({"stage": i, "kind": "host", "name": str(op.name), "op_type": str(op.op_type),
                        "op_kind": op_kind(op)})
            continue
        hcm = stage.hard_core_mapping
        static_tiles = operand_tiles = regions = cells = 0
        held = set()
        for placements in hcm.soft_core_placements_per_hard_core:
            ops = [p for p in placements if p.get("weight_bank_id") in banks]
            regions += len(ops)
            cells += sum(int(p["axons"]) * int(p["neurons"]) for p in ops)
            if ops:
                operand_tiles += 1
            elif placements:
                static_tiles += 1
            held |= {by_node.get(int(p["ir_node_id"])) for p in placements}
        out.append({"stage": i, "kind": "neural", "segment": stage.schedule_segment_index,
                    "pass": stage.schedule_pass_index, "cores": len(hcm.cores), "static_tiles": static_tiles,
                    "operand_tiles": operand_tiles, "operand_regions_written": regions, "operand_cells_written": cells,
                    "weights_resident": bool(stage.schedule_weights_resident),
                    "perceptrons": sorted(n for n in held if n)})
    return out


def c3_census(form: str) -> dict:
    ""
    model = ViTLiteCCT(num_classes=100, attention=form).eval()
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(copy.deepcopy(model), (3, 32, 32), 100, device="cpu", packaging=MVM_PACKAGING)
        repr_ = flow.get_mapper_repr()
        repr_.assign_perceptron_indices()
        ir = IRMapping(q_max=127, firing_mode="Default").map(repr_)


    planned = level_period_census(ir, capability_level("C3"), value_bits=8, payload_write=HOST_PAYLOAD_WRITE,
                                  bytes_of_cells=int)
    return {"c3_written_cells_producer": int(sum(p.written_cells_producer for p in planned.passes))}


def add_store_arguments(parser: argparse.ArgumentParser) -> None:
    ""
    parser.add_argument("--spec", help="CL7's spec (fig1_rows.json) the stores are read from")
    parser.add_argument("--spec-sha256", help="the spec's sha256 (refused by name when the file differs)")
    parser.add_argument("--store-row", action="append", default=[], help="a spec row whose target's stores to price")
    parser.add_argument("--block-sha256", action="append", default=[], help="NAME=SHA: a store block named by digest")


def declaration_from_args(args) -> tuple:
    ""
    if not args.spec:
        if args.spec_sha256 or args.store_row or args.block_sha256:
            raise StoreDeclarationRefused("STORE_ARGUMENTS_PARTIAL", "--spec-sha256, --store-row or --block-sha256 "
                                                                     "named without --spec")
        return None, None
    if not args.spec_sha256 or not args.store_row:
        raise StoreDeclarationRefused("STORE_ARGUMENTS_PARTIAL", "--spec needs --spec-sha256 and at least one "
                                                                 "--store-row")
    digests: dict = {}
    for item in args.block_sha256:
        name, sep, sha = item.partition("=")
        if not sep or not name or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise StoreDeclarationRefused("BLOCK_SHA256_FORM", f"{item!r} is not NAME=<64 hex digits>")
        if name in digests:
            raise StoreDeclarationRefused("BLOCK_SHA256_TWICE", f"block {name!r} is named twice")
        digests[name] = sha
    stores = store_declaration(args.spec, list(args.store_row), digests, args.spec_sha256)
    return stores, declaration_record(stores, {"path": str(args.spec), "sha256": args.spec_sha256,
                                               "rows": list(args.store_row)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="N2 acceptance 5, the executed declaration (revision 2)")
    parser.add_argument("out")
    parser.add_argument("threads", nargs="?", type=int, default=1)
    add_store_arguments(parser)
    args = parser.parse_args(argv)
    try:
        stores, record = declaration_from_args(args)
    except StoreDeclarationRefused as refused:
        print(f"REFUSED {refused}", file=sys.stderr, flush=True)
        return 2
    out = Path(args.out)
    import torch
    torch.set_num_threads(args.threads)
    data = ig.load_gate_data()
    result = {"rate_hz": RATE_HZ, "min_lifetime_hours": FLOOR_H,
              "script": {"name": "price_n2_r2.py", "revision": 2}, "stores_declared": record, "programs": {}}
    for key in ("softmax.O8", "relu.O8", "softmax.T"):
        program = ig.build_gate_program(key, data)
        _job, _form, arm, bits = ig.JOBS[key]
        packed, packing = dg.build_packed(program, bits if arm == "operand" else None)
        writes = sum(n.params["module"].payload_cells() for n in program.ir.get_compute_ops()
                     if isinstance(n.params.get("module"), OperandWrite))
        product_macs = sum(int(m.lowering.form.banks) * int(m.lowering.form.m) * int(m.lowering.form.k)
                           * int(m.lowering.form.n) for m in operand_mappers(program.model.get_mapper_repr()))
        rows, summary = price_program(packed, packing["cores_declared"], product_macs, stores=stores)
        entry = {"packing": packing, "census": summary, "operand_write_cells_per_inference": int(writes),
                 "timeline": timeline(packed, program.ir, program.model.get_mapper_repr()),
                 "product_macs": int(product_macs),
                 "bank_cells_per_inference": int(sum(b.cells for b in program.ir.dynamic_bank_bindings.values())),
                 "rows": rows}
        if arm == "operand":
            entry["g7"] = c3_census(_form)
            entry["g7"]["equals_operand_write_cells"] = entry["g7"]["c3_written_cells_producer"] == int(writes)
            entry["g7"]["equals_placed_region_cells"] = (entry["g7"]["c3_written_cells_producer"]
                                                         == summary["producer_cells_per_inference"])
        result["programs"][key] = entry
        out.write_text(json.dumps(result, indent=1, default=str))
        print(json.dumps({"key": key, **summary}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
