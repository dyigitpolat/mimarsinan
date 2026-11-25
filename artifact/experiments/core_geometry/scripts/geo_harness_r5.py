""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

RA = Path("records/research_artifacts")
G = RA / "13_geometry_study"
TR = RA / "12_tcad_readiness"
E2E = TR / "e2e"
OP_DIR = E2E / "w48" / "own_pick"
RR = OP_DIR / "study_chips_release"
RR_SUMS = OP_DIR / "release_run" / "pricing_scripts.sha256"
WORKTREE = Path("/work/worktrees/final-ro-R5")
WORKTREE_HEAD = "e5f6bb37d6285c6ac218d2558682a7e84fcb4c84"
PLAN = G / "plan_r5.json"
PLAN_SHA = "4dbdd6bc2185b5065f4f1b232766806be5ac5df0a272aa44d012bcf57e0a82eb"
LIVE_CLASSES = 78
OUT_ROOT = G / "sweep_r5"
LABEL_R5 = "CONFIRMATORY (pre-registered, EXPLORATORY study): {what} of the confirmatory run (plan_r5.json, RA/13_geometry_study); not a record"
BLOCK_LABEL = "CONFIRMATORY (pre-registered, EXPLORATORY study): a routed block"
HEADLINE_RESULT = RA / "06_build" / "headline_v46" / "headline_result.json"

FROZEN = {
    "12_tcad_readiness/e2e/w48/w48_pairs.py": (
        "w48_pairs", RA / "11_wave2/final_units/uw48m/install_backup_20261106T221303/12_tcad_readiness/e2e/w48/"
                     "w48_pairs.py"),
    "12_tcad_readiness/e2e/driver/declarations.py": (
        "declarations", RA / "11_wave2/final_units/uw48r/scripts_frozen/12_tcad_readiness/e2e/driver/declarations.py"),
}
OWN_PICK_REL = "12_tcad_readiness/e2e/w48/own_pick/own_pick.py"
CHIP = "study_compass"
VARIANTS = ("study:declared_read+edge_host", "study:declared_read+edge_host_low", "study:declared_read+edge_host_high")
KS = (1, 16, 64)
POINT_BUDGET_S = 1800.0
DR_TILE = "COMPASS 256x256x1"

DECLARING_PART = "R18-w100__256x64__{arm}.json"

BLOCK_KEYS = {"area_per_cell": ("area_per_cell_um2", "um^2"),
              "area_per_cell_per_weight_bit": ("area_per_cell_per_weight_bit_um2", "um^2"),
              "area_per_row_driver": ("area_per_row_driver_um2", "um^2"),
              "area_per_neuron_logic": ("area_per_neuron_logic_um2", "um^2"),
              "area_per_tile_fixed": ("area_per_tile_fixed_um2", "um^2"),
              "e_mac": ("e_mac_pj_per_weight_mac", "pJ"),
              "e_write_per_cell": ("e_write_per_cell_fj", "fJ"),
              "p_static_per_core": ("p_static_per_core_uw", "uW"),
              "conductance_levels": ("conductance_levels", "levels")}
GEOMETRY_NOTE = ("CONFIRMATORY (pre-registered, EXPLORATORY study) core geometry study (RA/13_geometry_study "
                 "plan_r5.json): the class's per-geometry block")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_readings as GR


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_frozen(name: str, frozen: Path, digest: str, live: Path) -> types.ModuleType:
    ""
    src = frozen.read_bytes()
    if hashlib.sha256(src).hexdigest() != digest:
        raise SystemExit(f"REFUSED: FROZEN_DIGEST: {frozen} is not RR's {digest}")
    mod = types.ModuleType(name)
    mod.__file__ = str(live)
    sys.modules[name] = mod
    exec(compile(src, str(live), "exec"), mod.__dict__)
    return mod


def bootstrap() -> Tuple[Any, Any, Dict[str, Any]]:
    ""
    if sha256_file(PLAN) != PLAN_SHA:
        raise SystemExit(f"REFUSED: PLAN_DIGEST: {PLAN} is not plan_r5.json {PLAN_SHA}")
    entries = []
    listed = {}
    for line in RR_SUMS.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        listed[rel] = digest
        if rel in FROZEN:
            src = FROZEN[rel][1]
            entries.append({"path": rel, "sha256": digest, "served_from": str(src.relative_to(RA)),
                            "equal": sha256_file(src) == digest, "live_equal": sha256_file(RA / rel) == digest})
        else:
            entries.append({"path": rel, "sha256": digest, "equal": sha256_file(RA / rel) == digest})
    bad = [e["path"] for e in entries if not e["equal"]]
    if bad:
        raise SystemExit(f"REFUSED: SCRIPT_DIGEST: {bad} differ from RR's pricing set ({RR_SUMS})")
    sys.path.insert(0, str(E2E / "driver"))
    for rel, (name, src) in FROZEN.items():
        _load_frozen(name, src, listed[rel], RA / rel)
    if sha256_file(RA / OWN_PICK_REL) != listed[OWN_PICK_REL]:
        raise SystemExit("REFUSED: SCRIPT_DIGEST: own_pick.py")
    spec = importlib.util.spec_from_file_location("own_pick", RA / OWN_PICK_REL)
    op = importlib.util.module_from_spec(spec)
    sys.modules["own_pick"] = op
    spec.loader.exec_module(op)
    W = sys.modules["w48_pairs"]
    if op.W is not W:
        raise SystemExit("REFUSED: FROZEN_IMPORT: own_pick did not take the frozen w48_pairs")
    start = {"plan_r5_sha256": PLAN_SHA, "pricing_set": {"path": str(RR_SUMS.relative_to(RA)),
                                                         "sha256": sha256_file(RR_SUMS), "entries": entries},
             "worktree": str(WORKTREE), "harness_sha256": sha256_file(Path(__file__)),
             "readings_sha256": sha256_file(Path(GR.__file__))}
    return W, op, start


def tree_check() -> Dict[str, Any]:
    import subprocess
    head = subprocess.run(["git", "-C", str(WORKTREE), "log", "-1", "--format=%H", "--ignore-submodules=all"],
                          capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(WORKTREE), "status", "--porcelain", "--ignore-submodules=all"],
                            capture_output=True, text=True).stdout.strip()
    if head != WORKTREE_HEAD or status:
        raise SystemExit(f"REFUSED: TREE: {WORKTREE} HEAD {head} ({len(status.splitlines())} status entries); the "
                         f"study prices on {WORKTREE_HEAD}, clean")
    return {"head": head, "status_entries": 0}


def plan() -> Dict[str, Any]:
    return json.loads(PLAN.read_text())


def class_of(p: Mapping[str, Any], R: int, C: int, s: int, k: int) -> Dict[str, Any]:
    got = [c for c in p["grid"]["classes"] if (c["R"], c["C_cells"], c["s"], c["cells_per_weight"]) == (R, C, s, k)]
    if len(got) != 1 or got[0]["pruned"]:
        raise SystemExit(f"REFUSED: CLASS: no live class R {R} C {C} s {s} k {k} in plan_r5.json")
    return got[0]


def load_routed_block(path: Path, digest: str) -> Dict[str, Any]:
    ""
    if sha256_file(path) != digest:
        raise SystemExit(f"REFUSED: ROUTED_BLOCK_DIGEST: {path} is not {digest}")
    block = json.loads(path.read_text())
    if not str(block.get("confirmatory", "")).startswith(BLOCK_LABEL):
        raise SystemExit(f"REFUSED: ROUTED_BLOCK_LABEL: {path}")
    return block


def overlay_routed(p: Dict[str, Any], block: Mapping[str, Any], digest: str) -> Dict[str, Any]:
    ""
    key = (block["R"], block["C_cells"], block["s"])
    q = dict(p)
    grid = dict(p["grid"])
    rows = [g for g in grid["geometry_table"] if (g["R"], g["C_cells"], g["s"]) != key]
    if len(rows) != len(grid["geometry_table"]) - 1:
        raise SystemExit(f"REFUSED: ROUTED_BLOCK: {key} is not one geometry of plan_r5.json")
    classes = [c for c in grid["classes"] if (c["R"], c["C_cells"], c["s"]) != key]
    mine = [c for c in grid["classes"] if (c["R"], c["C_cells"], c["s"]) == key]
    routed = [dict(c, routed_block_sha256=digest) for c in block["classes"]]
    if sorted(c["cells_per_weight"] for c in mine) != sorted(c["cells_per_weight"] for c in routed):
        raise SystemExit(f"REFUSED: ROUTED_BLOCK: {key}'s classes differ from the plan's")
    grid["geometry_table"] = rows + [dict(block["geometry_row"], routed_block_sha256=digest)]
    grid["classes"] = classes + routed
    q["grid"] = grid
    return q


def geometry_of(p: Mapping[str, Any], R: int, C: int, s: int) -> Dict[str, Any]:
    return next(g for g in p["grid"]["geometry_table"] if (g["R"], g["C_cells"], g["s"]) == (R, C, s))


def members_of(p: Mapping[str, Any], arch: str, bit_classes: Sequence[int]) -> List[str]:
    return [m["label"] for m in p["members"] if m["arch"] == arch and m["b_class"] in bit_classes]


def arm_declarations(W: Any, arm: str) -> Dict[str, Any]:
    ""
    part = RR / "parts" / DECLARING_PART.format(arm=arm)
    dec = json.loads(part.read_text())["declarations_of"]
    rel = dec.get("config") or f"12_tcad_readiness/e2e/routeb/rows/{dec['row']}/config.json"
    if sha256_file(RA / rel) != dec["config_sha256"]:
        raise SystemExit(f"REFUSED: CONFIG_DIGEST: {rel} is not the config RR priced ({dec['config_sha256']})")
    return {"row": dec["row"], "config": rel, "config_sha256": dec["config_sha256"], "declare": dec.get("declare"),
            "declaring_part": str(part.relative_to(RA)), "declaring_part_sha256": sha256_file(part)}


def _write_once(path: Path, body: str) -> str:
    digest = hashlib.sha256(body.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != digest:
            raise SystemExit(f"REFUSED: DERIVED_CONFIG: {path} exists with other bytes")
    else:
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(body)
        os.replace(tmp, path)
    return digest


def derived_config(arm_dec: Mapping[str, Any], cls: Mapping[str, Any], folder: Path,
                   row_values: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    ""
    config = json.loads((RA / arm_dec["config"]).read_text())
    to = config["platform_constraints"]["target_overrides"]
    lanes = int(cls["constants"]["lanes_per_core"])
    wb = int(cls["weight_bits_declared"])
    to["onchip_arithmetic"]["value"]["lanes_per_core"] = lanes
    to["onchip_arithmetic"]["note"] = (f"{to['onchip_arithmetic'].get('note', '')}; EXPLORATORY core geometry study: "
                                       f"lanes_per_core {lanes} = max(1, ceil(C / 64)) (G-C6; the bytes depend only on the row, the lanes and "
                                       f"the weight bits, the three the file name states)")
    to["weight_bits"]["value"] = wb
    to["weight_bits"]["note"] = (f"{to['weight_bits'].get('note', '')}; EXPLORATORY core geometry study: weight_bits "
                                 f"{wb} = min(4, cells_per_weight x s)")
    label = ("CONFIRMATORY (pre-registered, EXPLORATORY study): a derived config of the confirmatory run "
             f"(plan_r5.json, RA/13_geometry_study), from {arm_dec['config']} ({arm_dec['config_sha256']}); not a record")
    out: Dict[str, Any] = {"changed": {"onchip_arithmetic.lanes_per_core": lanes, "weight_bits": wb}}
    chip_body = json.dumps({"exploratory": label + "; the study chip's form", **config}, indent=1) + "\n"
    chip_path = folder / f"{arm_dec['row']}__L{lanes}_wb{wb}__chip.json"
    out["chip"] = (chip_path, _write_once(chip_path, chip_body))
    row_config = json.loads(json.dumps(config))
    rto = row_config["platform_constraints"]["target_overrides"]
    dropped = sorted(key for key in ("onchip_arithmetic", "weight_bits")
                     if row_values is not None and key in row_values and rto[key]["value"] == row_values[key])
    for key in dropped:
        rto.pop(key)
    out["dropped_on_row"] = dropped
    row_body = json.dumps({"exploratory": label + ("; the row's form" + (f", {dropped} left to the row's target "
                                                                        f"{config['platform_constraints']['target']}"
                                                                        if dropped else "")), **row_config},
                          indent=1) + "\n"
    row_path = folder / f"{arm_dec['row']}__L{lanes}_wb{wb}__row.json"
    out["row"] = (row_path, _write_once(row_path, row_body))
    return out


def declared_read(W: Any) -> Dict[str, Any]:
    ""
    result = json.loads(HEADLINE_RESULT.read_text())
    cells = {name: {"declared_read": {"s_au": cell["declared_read"]["s_au"], "combo": cell["declared_read"]["combo"]}}
             for name, cell in result["measured"]["cells"].items()}
    return W.declared_read({"cells": cells})


def geometry_block(cls: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    ""
    c = cls["constants"]
    block: Dict[str, Dict[str, Any]] = {}
    for key, (field, unit) in BLOCK_KEYS.items():
        v = c[field]
        low, nom, high = (v if isinstance(v, list) else [v, v, v])
        block[key] = {"low": float(low), "nominal": float(nom), "high": float(high), "unit": unit,
                      "evidence_kind": "derived",
                      "derivation": (f"plan_r5.json grid.classes R {cls['R']} C {cls['C_cells']} s {cls['s']} k "
                                     f"{cls['cells_per_weight']}: {field}"
                                     + (f" (routed block {cls['routed_block_sha256']})"
                                        if cls.get("routed_block_sha256") else "")),
                      "note": GEOMETRY_NOTE}
    return block


class GeometryChip:
    ""

    def __init__(self, W: Any, tc: Any, cls: Mapping[str, Any]):
        from mimarsinan.deployment_record.platform_physics import get_platform_physics
        self.W, self.tc, self.cls = W, tc, cls
        values = tc.chip_values(CHIP)
        physics = get_platform_physics(values["platform_physics_profile"])
        constants = {k: v.to_dict() for k, v in physics.constants.items()}
        self.geometry = geometry_block(cls)


        own = {**constants, **{k: {**constants.get(k, {}), **{f: v[f] for f in ("low", "nominal", "high", "unit")}}
                               for k, v in self.geometry.items()}}
        tile_mm2 = float(W.load_price_measured().estimators().K.PV.TILE_MM2[DR_TILE])
        self.declared_read = W.declared_read_block(own, tile_mm2=tile_mm2, lanes_per_core=1, read=declared_read(W))
        self.geometry_kept = {k: v for k, v in self.geometry.items() if k not in self.declared_read}

    def overrides(self, variant: str) -> Dict[str, Any]:
        kind, blocks = self.W.chip_variant(variant)
        parts = blocks.split("+")
        if kind != "study" or self.W.DECLARED_READ not in parts:
            raise SystemExit(f"REFUSED: VARIANT: {variant} is not a declared-read study variant")
        named: Dict[str, Any] = {self.W.DECLARED_READ: self.declared_read, "geometry": self.geometry_kept}
        named.update(self.tc._named_blocks([p for p in parts if p != self.W.DECLARED_READ]))
        return self.tc.merge(named)

    def digest(self) -> str:
        return sha256_json({"declared_read": self.declared_read, "geometry": self.geometry_kept})


def stored_quantities(tc: Any, network: Any, census: Any, scheduled: Any, platform: Mapping[str, Any],
                      priced: Any) -> Dict[str, Any]:
    ""
    from mimarsinan.deployment_record.introspection.scaling_ladder_lanes import host_census
    from mimarsinan.deployment_record.objectives import CandidateStaticView, declared_core_capacity
    from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
    from mimarsinan.deployment_record.platform_physics import PlatformPhysics
    from mimarsinan.deployment_record.cost.absolute.formulas import carry_buffer_static_watts
    from mimarsinan.deployment_record.quantities.write_law import write_quantities
    from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
    facts = scheduled.facts
    host = host_census(network, census)
    context = candidate_context_from_platform(
        platform, latency_steps=facts.latency_steps, programming=facts.programming, carry=facts.carry,
        passes=facts.passes, **{k: host.get(k) for k in ("host_macs", "onchip_macs", "host_params",
                                                          "onchip_params", "compute_op_count")},
        lane_ops=census.lane_ops, lane_walked=census.lane_ops is not None, host_boundary=census.host_boundary)
    physics = PlatformPhysics.from_dict(platform["platform_physics_resolved"])
    q = CandidateStaticView(layout=scheduled.stats, chip_param_capacity=declared_core_capacity(dict(platform)),
                            total_params=network.total_params, host_side_segment_count=census.host_segments,
                            physics=physics, quantity_context=context, noc_fragments=scheduled.noc).quantities

    def val(key: str) -> Optional[float]:
        return float(q.get(key).value) if q.has(key) else None

    out: Dict[str, Any] = {k: val(k) for k in ("synaptic_events", "cores_physical", "tiles", "cells_physical",
                                               "neurons_physical", "axons_physical", "lanes_per_tile", "weight_bits",
                                               "carry_buffer_capacity_bytes", "macs", "timesteps")}


    out["latency_steps"] = val("latency_steps")
    out["latency_steps_schedule"] = facts.latency_steps
    t_cycle = physics.band("t_cycle")
    steps = float(out["latency_steps"] or 0)
    out["t_cycle_s"] = [t_cycle.low, t_cycle.nominal, t_cycle.high]
    out["compute_time_s"] = [t_cycle.low * steps, t_cycle.nominal * steps, t_cycle.high * steps]
    watts = []
    if physics.has("p_static_per_core") and q.has("cores_physical"):
        b = physics.band("p_static_per_core")
        n = q.get("cores_physical").value
        watts.append([b.low * n, b.nominal * n, b.high * n])
    if physics.has("p_static_global"):
        b = physics.band("p_static_global")
        watts.append([b.low, b.nominal, b.high])
    buf = carry_buffer_static_watts(q, physics)
    if buf is not None:
        watts.append([buf.low, buf.nominal, buf.high])
    out["static_power_w"] = [sum(w[i] for w in watts) for i in range(3)] if watts else None
    writes = context.writes
    if writes is not None and writes.census is not None:
        wi = WriteInputs.of(writes)
        wq = write_quantities(wi.census, wi.policy, written_extents=wi.written_extents, most_written=wi.most_written,
                              producer_slots=wi.producer_slots)
        k = wi.policy.batch_k
        out["period_rewrite_bytes"] = float(wq["period_rewrite_bytes"].value)
        out["runtime_written_cells"] = float(wq["runtime_written_cells"].value)
        out["written_weights"] = (sum(p.written_cells_backing_store for p in wi.census.passes) / k
                                  + sum(p.written_cells_producer for p in wi.census.passes))
        out["write_overlap_mode"] = wi.policy.overlap_mode
        out["batch_k"] = k
    else:
        out.update(period_rewrite_bytes=None, runtime_written_cells=None, written_weights=None,
                   write_overlap_mode=str(dict(platform.get("write_overlap") or {}).get("mode", "serial")),
                   batch_k=platform.get("inference_batch_K"),
                   writes_unavailable=None if writes is None else writes.unavailable)
    return out


def price_rows(W: Any, OP: Any, tc: Any, network: Any, census: Any, tiles: int, path: Path,
               declare: Optional[Mapping[str, Any]], shape: Mapping[str, Any], counts: Sequence[int],
               variants: Sequence[str], chips: Sequence[Optional[str]], overrides: Mapping[Tuple, Any],
               lows: Mapping[Optional[str], Any], ks: Sequence[int], budget_s: Optional[float],
               quantities: bool, log=None, chip_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    ""
    rows: List[Dict[str, Any]] = []
    for count in counts:
        t0 = time.time()
        cores = [{**shape, "count": int(count)}]
        point = tc.platform(path, cores=cores, declare=declare)
        scheduled, mark = tc.schedule(network, census, point, tiles, budget_s)
        row: Dict[str, Any] = {"cores": int(count)}
        if scheduled is None or scheduled.refusal is not None:
            row["refusal"] = mark or dict(scheduled.refusal)
            if quantities:
                row["refusal_name"] = GR.refusal_name(row["refusal"])
                row["wall_s"] = round(time.time() - t0, 3)
            rows.append(row)
            continue
        if quantities:
            row["quantities"] = {}
        for variant in variants:
            kind, blocks = W.chip_variant(variant)
            for chip in (chips if kind == "study" else [None]):
                name = variant if chip is None else f"{chip}:{blocks}"
                if W.PORT_LOW in blocks.split("+") and lows[chip] is None:
                    row[name] = {"absent": "the chip states no t_program_per_byte band"}
                    continue
                for k in ks:
                    own = tc.platform(path, cores=cores, overrides=overrides[(chip, blocks)], k=k, declare=declare)
                    platform = own if chip is None else W.physics_swap(own, tc.platform(
                        chip_path or path, cores=cores, overrides=overrides[(chip, blocks)], k=k, declare=declare,
                        chip=chip))
                    priced, pmark = tc.price_candidate(network, census, scheduled, platform, budget_s)
                    if priced is None:
                        row[f"{name}|K{k}"] = pmark
                        if quantities:
                            row["quantities"][f"{name}|K{k}"] = {"refusal_name": GR.refusal_name(pmark)}
                        continue
                    axes = W.plane_axes(W.candidate_axes(priced), variant)
                    axes["store_traffic"] = tc.store_traffic(scheduled, platform, priced)
                    row[f"{name}|K{k}"] = OP.design_row(axes)
                    if quantities:
                        row["quantities"][f"{name}|K{k}"] = stored_quantities(tc, network, census, scheduled,
                                                                              platform, priced)
        if quantities:
            row["wall_s"] = round(time.time() - t0, 3)
        rows.append(row)
        if log:
            log(f"count {count} {'refused ' + GR.refusal_name(row['refusal']) if 'refusal' in row else 'priced'} "
                f"{time.time() - t0:.1f} s")
    return rows


def price_part(W: Any, OP: Any, tc: Any, p: Mapping[str, Any], R: int, C: int, s: int, k: int, arch_id: str,
               arm: str, configs: Path, counts: Optional[Sequence[int]] = None,
               variants: Sequence[str] = VARIANTS, ks: Sequence[int] = KS, log=None) -> Dict[str, Any]:
    ""
    t_all = time.time()
    cls = class_of(p, R, C, s, k)
    geo = geometry_of(p, R, C, s)
    if arch_id not in cls["architectures"]:
        raise SystemExit(f"REFUSED: ARCH: {arch_id} has no member at the class's bit classes {cls['member_bit_classes']}")
    rules = json.loads((W.HEADLINE / "headline_rules.json").read_text())
    arch = {a["id"]: a for a in rules["architectures"]}[arch_id]
    dec = arm_declarations(W, arm)
    row_target = json.loads((RA / dec["config"]).read_text())["platform_constraints"]["target"]
    cfg = derived_config(dec, cls, configs, tc.chip_values(row_target))
    (path, cfg_sha), (chip_path, chip_sha) = cfg["row"], cfg["chip"]
    declare = dec["declare"]
    if declare and declare.get("carry_buffer_bytes"):
        OP.check_buffer(tc.worktree, [CHIP], int(declare["carry_buffer_bytes"]))
    chip = GeometryChip(W, tc, cls)
    counts = list(geo["ladder_counts"] if counts is None else counts)
    out: Dict[str, Any] = {
        "confirmatory": LABEL_R5.format(what="a priced part"),
        "routed_block_sha256": cls.get("routed_block_sha256"),
        "class": {"R": R, "C_cells": C, "s": s, "cells_per_weight": k, "weight_columns": cls["weight_columns"],
                  "idle_column_factor": cls["idle_column_factor"], "weight_bits": cls["weight_bits_declared"],
                  "lanes_per_core": cls["constants"]["lanes_per_core"], "member_bit_classes": cls["member_bit_classes"]},
        "arch": arch_id, "depth": arch["depth"], "width": arch["width"], "arm": arm,
        "members": members_of(p, arch_id, cls["member_bit_classes"]),
        "declarations_of": {**dec, "derived_config_row": str(path.relative_to(G)), "derived_config_row_sha256": cfg_sha,
                            "derived_config_chip": str(chip_path.relative_to(G)),
                            "derived_config_chip_sha256": chip_sha, "derived_changes": cfg["changed"],
                            "left_to_the_row_target": cfg["dropped_on_row"], "row_target": row_target},
        "chip": CHIP, "geometry_block": {"declared_read": chip.declared_read, "geometry": chip.geometry_kept},
        "geometry_block_sha256": chip.digest(), "variants": list(variants), "ks": list(ks), "counts": counts,
        "point_budget_s": POINT_BUDGET_S}
    network = tc.network(arch["depth"], float(arch["width"]))
    base = tc.platform(path, declare=declare)
    shape = {kk: v for kk, v in base["cores"][0].items() if kk != "count"}
    shape.update(max_axons=int(R), max_neurons=int(cls["weight_columns"]))
    if k > 1:
        shape.update(cells_per_weight=int(k), weight_cell_rows=1)
    out["core_shape"] = shape
    t0 = time.time()
    geo_point = tc.platform(path, cores=[{**shape, "count": counts[0]}], declare=declare)
    census = tc.LD.shape_census(network, geo_point)
    tiles = tc.LD.packed_tiles(census, geo_point)
    out["layout"] = {"key": tc.lane_key(geo_point), "softcores": len(census.softcores), "segments": census.segments,
                     "host_ops": tc.LD.host_census(network, census).get("compute_op_count"), "packed_tiles": tiles,
                     "wall_s": round(time.time() - t0, 1)}
    overrides = {(CHIP, W.chip_variant(v)[1]): chip.overrides(v) for v in variants}
    out["fixed_area_mm2_by_variant"] = {
        f"{CHIP}:{W.chip_variant(v)[1]}": tc.fixed_area_mm2(tc.platform(chip_path, declare=declare, chip=CHIP,
                                                                         overrides=overrides[(CHIP, W.chip_variant(v)[1])]))
        for v in variants}
    out["rows"] = price_rows(W, OP, tc, network, census, tiles, path, declare, shape, counts, variants, [CHIP],
                             overrides, {CHIP: None}, ks, POINT_BUDGET_S, True, log=log, chip_path=chip_path)
    for row in out["rows"]:
        row["rung_tags"] = GR.rung_tags(geo, row["cores"])
    out["wall_s"] = round(time.time() - t_all, 1)
    return out


def h1(W: Any, OP: Any, tc: Any, log) -> Dict[str, Any]:
    ""
    rr_path = RR / "parts" / "R10-w050__256x64__pipe+P+B.json"
    rr = json.loads(rr_path.read_text())
    arch = {a["id"]: a for a in json.loads((W.HEADLINE / "headline_rules.json").read_text())["architectures"]}["R10-w050"]


    dec = rr["declarations_of"]
    path = RA / dec["config"]
    if sha256_file(path) != dec["config_sha256"]:
        raise SystemExit(f"REFUSED: CONFIG_DIGEST: {path} is not the config RR priced")
    if OP.p_variant_difference(path, RA / f"12_tcad_readiness/e2e/routeb/rows/{dec['row']}/config.json") != {
            "onchip_arithmetic": (["L", "E"], OP.POOL_CLASSES)}:
        raise SystemExit("REFUSED: P_VARIANT_DECLARATION: the P variant differs from its row beyond the pool lane")
    spec = OP.arm_spec("pipe+P+B")
    declare = dec["declare"]
    if declare != {"max_schedule_passes": W.ONE_COPY_BUDGET, **spec["declare"]}:
        raise SystemExit(f"REFUSED: DECLARATIONS: RR's declare {declare} is not the arm's")
    got = {"row": dec["row"], "member": dec["member"]}
    variants = OP.declared_variants("pipe+P+B")
    network = tc.network(arch["depth"], float(arch["width"]))
    base = tc.platform(path, declare=declare)
    shape = {k: v for k, v in base["cores"][0].items() if k != "count"}
    census = tc.LD.shape_census(network, base)
    tiles = tc.LD.packed_tiles(census, base)
    chips = list(OP.PART_CHIPS.get("256x64", ())) + [None]
    lows = {}
    fixed = {}
    for chip in chips:
        cb = tc.platform(path, declare=declare, chip=chip)
        lows[chip] = OP.port_low_block(tc, cb)
        fixed[chip or W.GENERIC_CHIP] = tc.fixed_area_mm2(cb)
    overrides = OP.overrides_table(tc, chips, lows)
    by_variant = {f"{chip}:{W.chip_variant(v)[1]}": tc.fixed_area_mm2(tc.platform(
        path, declare=declare, chip=chip, overrides=overrides[(chip, W.chip_variant(v)[1])]))
        for v in variants if W.chip_variant(v)[0] == "study" for chip in chips[:-1]}
    rows = price_rows(W, OP, tc, network, census, tiles, path, declare, shape, rr["caps"], variants, chips[:-1],
                      overrides, lows, rr["ks"], 1800.0, False, log=log)

    def canon(x):
        return json.loads(json.dumps(x, sort_keys=True, default=str))
    mine = canon({"rows": rows, "fixed_area_mm2_by_variant": by_variant, "fixed_area_mm2": fixed,
                  "variants": list(variants), "declarations_of": {"row": got["row"], "member": got["member"],
                                                                  "declare": declare,
                                                                  "config_sha256": W.sha256(path)},
                  "layout_packed_tiles": tiles, "layout_softcores": len(census.softcores)})
    theirs = canon({"rows": rr["rows"], "fixed_area_mm2_by_variant": rr["fixed_area_mm2_by_variant"],
                    "fixed_area_mm2": rr["fixed_area_mm2"], "variants": rr["variants"],
                    "declarations_of": {k: rr["declarations_of"][k] for k in ("row", "member", "declare",
                                                                              "config_sha256")},
                    "layout_packed_tiles": rr["layout"]["packed_tiles"], "layout_softcores": rr["layout"]["softcores"]})
    diffs = []
    for key in mine:
        if mine[key] != theirs[key]:
            diffs.append(key)
    row_diffs = []
    for a, b in zip(mine["rows"], theirs["rows"]):
        if a != b:
            keys = sorted(kk for kk in set(a) | set(b) if a.get(kk) != b.get(kk))
            row_diffs.append({"cores": a.get("cores"), "keys": keys[:10], "n_keys": len(keys)})
    values = sum(1 for r in mine["rows"] for kk, v in r.items() if isinstance(v, dict))
    out = {"check": "H1", "rr_part": str(rr_path.relative_to(RA)), "rr_part_sha256": sha256_file(rr_path),
           "counts": rr["caps"], "variants": len(variants), "chips": chips[:-1], "ks": rr["ks"],
           "priced_or_marked_entries_compared": values, "equal": not diffs, "differing_fields": diffs,
           "row_diffs": row_diffs[:20]}
    if diffs:
        out["refusal"] = "HARNESS_REPRODUCTION"
    return out


def h2(W: Any, OP: Any, tc: Any, p: Mapping[str, Any], configs: Path, log) -> Dict[str, Any]:
    ""


    cases = [("192x48x4 (analog of 256x64x4)", [((192, 48, 4, 1), (172, 171)), ((192, 48, 4, 1), (204, 203)),
                                                ((192, 48, 4, 1), (244, 243)), ((192, 48, 4, 1), (288, 289)),
                                                ((192, 48, 4, 1), (344, 345))]),
             ("192x192x1 k4 (analog of 256x256x1 k4)", [((192, 192, 1, 4), (120, 121)), ((192, 192, 1, 4), (144, 143)),
                                                         ((192, 192, 1, 4), (172, 171))])]
    idle_classes = [c for c in p["grid"]["classes"] if c["idle_cell_columns"] > 0]
    arch = "R10-w050"
    checks, attempts = [], []
    for label, tries in cases:
        for (R, C, s, k), counts in tries:
            geo = geometry_of(p, R, C, s)
            got = []
            for arm in p["arms"]:
                part = price_part(W, OP, tc, p, R, C, s, k, arch, arm, configs, counts=counts,
                                  variants=VARIANTS[:1], ks=(1,), log=log)
                fixed = part["fixed_area_mm2_by_variant"][f"{CHIP}:{W.chip_variant(VARIANTS[0])[1]}"]
                for row in part["rows"]:
                    n = row["cores"]
                    form = GR.area_form_um2(n, geo["core_per_core_um2"], geo["noc_tile_um2"],
                                            arm in GR.BUFFERED_ARMS) * 1e-6
                    if "refusal" in row:
                        got.append({"class": [R, C, s, k], "arm": arm, "cores": n, "refusal": row["refusal_name"]})
                        continue
                    d = row[f"{CHIP}:{W.chip_variant(VARIANTS[0])[1]}|K1"]
                    tile_only = d["area_mm2"][1] - fixed[1]
                    rel = tile_only / form - 1
                    qq = row["quantities"][f"{CHIP}:{W.chip_variant(VARIANTS[0])[1]}|K1"]
                    got.append({"class": [R, C, s, k], "arm": arm, "cores": n, "tiles": qq["tiles"],
                                "noc_tiles_form": GR.noc_tiles(n), "priced_mm2": tile_only, "form_mm2": form,
                                "rel": rel, "ok": abs(rel) <= 1e-9 and qq["tiles"] == GR.noc_tiles(n)})
            priced_pair = all(sum(1 for c in got if c["arm"] == arm and "ok" in c) == 2 for arm in p["arms"])
            attempts.append({"case": label, "class": [R, C, s, k], "counts": list(counts), "priced_pair_on_every_arm":
                             priced_pair, "refused": [c for c in got if "refusal" in c]})
            if priced_pair:
                checks.extend(got)
                break
        else:
            attempts.append({"case": label, "none_priced": True})
    ok = (all(c.get("ok") for c in checks) and not idle_classes
          and all(any(a.get("case") == lab and a.get("priced_pair_on_every_arm") for a in attempts) for lab, _ in cases))
    out = {"check": "H2", "arch": arch, "cases": checks, "attempts": attempts, "equal": ok,
           "idle_column_case": ("vacuous: no class of plan_r5.json's held-out grid has idle cell columns "
                                f"({len(idle_classes)} of {len(p['grid']['classes'])})"),
           "max_abs_rel": max((abs(c["rel"]) for c in checks if "rel" in c), default=None)}
    if not ok:
        out["refusal"] = "AREA_IDENTITY"
    return out


def h2_routed(W: Any, OP: Any, tc: Any, p: Mapping[str, Any], block: Mapping[str, Any], configs: Path,
              log) -> Dict[str, Any]:
    ""
    R, C, s = block["R"], block["C_cells"], block["s"]
    geo = geometry_of(p, R, C, s)
    live = [c for c in p["grid"]["classes"] if (c["R"], c["C_cells"], c["s"]) == (R, C, s) and not c["pruned"]]
    if not live:
        return {"check": "H2 routed block", "geometry": f"{R}x{C}x{s}", "equal": True,
                "note": "the geometry has no live class: nothing is priced from the block"}
    pick_cls = next((c for c in live if "R10-w050" in c["architectures"]), live[0])
    arch = "R10-w050" if "R10-w050" in pick_cls["architectures"] else pick_cls["architectures"][0]
    k = pick_cls["cells_per_weight"]
    pairs = []
    for n in sorted(set(geo["ladder_main_by_j"]), reverse=True):
        m4 = 4 * (n // 4)
        if m4 >= 2 and (m4, m4 - 1) not in pairs:
            pairs.append((m4, m4 - 1))
    attempts, checks = [], []
    for counts in pairs[:4]:
        got = []
        for arm in p["arms"]:
            part = price_part(W, OP, tc, p, R, C, s, k, arch, arm, configs, counts=list(counts),
                              variants=VARIANTS[:1], ks=(1,), log=log)
            fixed = part["fixed_area_mm2_by_variant"][f"{CHIP}:{W.chip_variant(VARIANTS[0])[1]}"]
            for row in part["rows"]:
                n = row["cores"]
                form = GR.area_form_um2(n, geo["core_per_core_um2"], geo["noc_tile_um2"],
                                        arm in GR.BUFFERED_ARMS) * 1e-6
                if "refusal" in row:
                    got.append({"arm": arm, "cores": n, "refusal": row["refusal_name"]})
                    continue
                d = row[f"{CHIP}:{W.chip_variant(VARIANTS[0])[1]}|K1"]
                tile_only = d["area_mm2"][1] - fixed[1]
                rel = tile_only / form - 1
                qq = row["quantities"][f"{CHIP}:{W.chip_variant(VARIANTS[0])[1]}|K1"]
                got.append({"arm": arm, "cores": n, "tiles": qq["tiles"], "noc_tiles_form": GR.noc_tiles(n),
                            "priced_mm2": tile_only, "form_mm2": form, "rel": rel,
                            "ok": abs(rel) <= 1e-9 and qq["tiles"] == GR.noc_tiles(n)})
        priced_pair = all(sum(1 for c in got if c["arm"] == arm and "ok" in c) == 2 for arm in p["arms"])
        attempts.append({"counts": list(counts), "priced_pair_on_every_arm": priced_pair,
                         "refused": [c for c in got if "refusal" in c]})
        if priced_pair:
            checks = got
            break
    ok = bool(checks) and all(c.get("ok") for c in checks)
    out = {"check": "H2 routed block", "geometry": f"{R}x{C}x{s}", "class_k": k, "arch": arch,
           "routed_core_per_core_um2": geo["core_per_core_um2"], "cases": checks, "attempts": attempts,
           "equal": ok, "max_abs_rel": max((abs(c["rel"]) for c in checks if "rel" in c), default=None)}
    if not ok:
        out["refusal"] = "AREA_IDENTITY"
    return out


def h3(W: Any, OP: Any, tc: Any, p: Mapping[str, Any], configs: Path, log) -> Dict[str, Any]:
    ""
    counts = (85, 101, 121, 144, 172)
    a = price_part(W, OP, tc, p, 192, 192, 1, 4, "R10-w050", "pipe+P+B", configs, counts=counts,
                   variants=VARIANTS[:1], log=log)
    b = price_part(W, OP, tc, p, 192, 48, 4, 1, "R10-w050", "pipe+P+B", configs, counts=counts,
                   variants=VARIANTS[:1], log=log)
    checks = []
    for ra, rb in zip(a["rows"], b["rows"]):
        if "refusal" in ra or "refusal" in rb:
            checks.append({"cores": ra["cores"], "refusal": [ra.get("refusal_name"), rb.get("refusal_name")],
                           "both_refused": "refusal" in ra and "refusal" in rb})
            continue
        for key in ra["quantities"]:
            qa, qb = ra["quantities"][key], rb["quantities"][key]
            ev = qa["synaptic_events"] == qb["synaptic_events"]
            wr = (qa["period_rewrite_bytes"] == qb["period_rewrite_bytes"]
                  and math.isclose(qa["runtime_written_cells"], qa["written_weights"] * 4, rel_tol=1e-12, abs_tol=1e-9)
                  and math.isclose(qb["runtime_written_cells"], qb["written_weights"] * 1, rel_tol=1e-12, abs_tol=1e-9)
                  and math.isclose(qa["written_weights"], qb["written_weights"], rel_tol=1e-12, abs_tol=1e-9))
            checks.append({"cores": ra["cores"], "key": key, "events": [qa["synaptic_events"], qb["synaptic_events"]],
                           "period_rewrite_bytes": [qa["period_rewrite_bytes"], qb["period_rewrite_bytes"]],
                           "runtime_written_cells": [qa["runtime_written_cells"], qb["runtime_written_cells"]],
                           "written_weights": [qa["written_weights"], qb["written_weights"]],
                           "CPW_EVENTS": ev, "CPW_WRITES": wr})
    ev_ok = all(c.get("CPW_EVENTS", c.get("both_refused")) for c in checks)
    wr_ok = all(c.get("CPW_WRITES", c.get("both_refused")) for c in checks)
    written = sum(1 for c in checks if (c.get("written_weights") or [0])[0])
    out = {"check": "H3", "cases": checks, "events_equal": ev_ok, "writes_identity": wr_ok,
           "designs_with_written_weights": written}
    if not ev_ok:
        out["refusal"] = "CPW_EVENTS"
    elif not wr_ok:
        out["refusal"] = "CPW_WRITES"
    return out


def h4(W: Any, OP: Any, tc: Any, log) -> Dict[str, Any]:
    ""
    dec = arm_declarations(W, "pipe+bufS+P+B")
    path = RA / dec["config"]
    declare = dec["declare"]
    arch = {a["id"]: a for a in json.loads((W.HEADLINE / "headline_rules.json").read_text())["architectures"]}["R18-w071"]
    network = tc.network(arch["depth"], float(arch["width"]))
    base = tc.platform(path, declare=declare)
    shape = {k: v for k, v in base["cores"][0].items() if k != "count"}
    shape.update(max_axons=128, max_neurons=64)
    geo = tc.platform(path, cores=[{**shape, "count": 86}], declare=declare)
    census = tc.LD.shape_census(network, geo)
    tiles = tc.LD.packed_tiles(census, geo)
    cases = []
    for count in (86, 102, 121, 144, 171):
        point = tc.platform(path, cores=[{**shape, "count": count}], declare=declare)
        scheduled, mark = tc.schedule(network, census, point, tiles, POINT_BUDGET_S)
        if scheduled is None or scheduled.refusal is not None:
            ref = mark or dict(scheduled.refusal)
            shape_kind = "refusal.name" if "name" in ref else "refusal.refusal.name" if "refusal" in ref else "other"
            try:
                name = GR.refusal_name(ref)
            except ValueError as e:
                name = str(e)
            cases.append({"cores": count, "shape": shape_kind, "name": name})
        else:
            cases.append({"cores": count, "scheduled": True})
    shapes_ok = True
    for bad in ({"above_budget": {"stage": "schedule"}}, {"refusal": {}}, None, "text"):
        try:
            GR.refusal_name(bad)
            shapes_ok = False
        except ValueError:
            pass
    named = [c for c in cases if "name" in c]
    ok = bool(named) and all(not c["name"].startswith("REFUSED") for c in named) and shapes_ok
    out = {"check": "H4", "cases": cases, "other_shapes_refuse": shapes_ok, "named": len(named), "ok": ok}
    if not ok:
        out["refusal"] = "REFUSAL_SHAPE"
    return out


def selftest(W: Any, OP: Any, tc: Any, p: Mapping[str, Any], configs: Path) -> Dict[str, Any]:
    ""
    from mimarsinan.deployment_record.platform_physics import PlatformPhysics
    from mimarsinan.deployment_record.platform_physics.resolve import OverrideBlockConflictError
    from mimarsinan.deployment_record.units import to_canonical
    results = []

    def check(name, cond, detail=""):
        results.append({"test": name, "ok": bool(cond), "detail": detail})
    live = [c for c in p["grid"]["classes"] if not c["pruned"]]
    check(f"classes_live_{LIVE_CLASSES}", len(live) == LIVE_CLASSES, len(live))

    for cls in (class_of(p, 192, 48, 4, 1), class_of(p, 192, 192, 1, 4), class_of(p, 96, 96, 1, 3),
                class_of(p, 96, 96, 2, 2), class_of(p, 384, 192, 3, 2)):
        tag = f"{cls['R']}x{cls['C_cells']}x{cls['s']} k{cls['cells_per_weight']}"
        chip = GeometryChip(W, tc, cls)
        c = cls["constants"]
        dr = chip.declared_read
        check(f"dr_lane_15015 {tag}", abs(dr["area_per_lane"]["nominal"] - GR.LANE_UM2) < 1e-9,
              dr["area_per_lane"]["nominal"])
        check(f"dr_write_f2_class_nominal {tag}",
              abs(dr["e_write_per_cell"]["nominal"] - 2 * c["e_write_per_cell_fj"][1]) < 1e-12,
              dr["e_write_per_cell"]["nominal"])
        check(f"dr_points {tag}", all(v["low"] == v["nominal"] == v["high"] for v in dr.values()))
        check(f"geometry_minus_dr {tag}", set(chip.geometry_kept) == set(BLOCK_KEYS) - {"e_write_per_cell"},
              sorted(chip.geometry_kept))
        for v in VARIANTS:
            ov = chip.overrides(v)
            dec = arm_declarations(W, "pipe+P+B")
            row_target = json.loads((RA / dec["config"]).read_text())["platform_constraints"]["target"]
            cfg = derived_config(dec, cls, configs, tc.chip_values(row_target))
            path = cfg["chip"][0]
            cores = [{"max_axons": cls["R"], "max_neurons": cls["weight_columns"], "has_bias": True, "count": 100,
                      **({"cells_per_weight": cls["cells_per_weight"], "weight_cell_rows": 1}
                         if cls["cells_per_weight"] > 1 else {})}]
            pt = tc.platform(path, cores=cores, overrides=ov, k=1, declare=dec["declare"], chip=CHIP)
            own = tc.platform(cfg["row"][0], cores=cores, overrides=ov, k=1, declare=dec["declare"])
            try:
                W.physics_swap(own, pt)
                check(f"physics_swap {tag} {v.split('+')[-1]}", True)
            except SystemExit as e:
                check(f"physics_swap {tag} {v.split('+')[-1]}", False, str(e)[:300])
            own_lanes = (own.get("onchip_arithmetic") or {}).get("lanes_per_core")
            check(f"row lanes_per_core {tag}", own_lanes is not None and int(own_lanes) == int(c["lanes_per_core"]),
                  [own.get("onchip_arithmetic"), cfg["dropped_on_row"]])
            phys = PlatformPhysics.from_dict(pt["platform_physics_resolved"])
            for key, (field, unit) in BLOCK_KEYS.items():
                if key == "e_write_per_cell":
                    continue
                want = c[field] if isinstance(c[field], list) else [c[field]] * 3
                band = phys.band(key)
                got_c = [band.low, band.nominal, band.high]
                scale = to_canonical(1.0, unit)
                check(f"resolved {key} {tag} {v.split('+')[-1]}",
                      all(abs(g - w * scale) <= 1e-12 * max(abs(w * scale), 1e-300) for g, w in zip(got_c, want)),
                      [got_c, want, unit])
            lanes_got = (pt.get("onchip_arithmetic") or {}).get("lanes_per_core")
            check(f"resolved lanes_per_core {tag}", lanes_got is not None and int(lanes_got) == int(c["lanes_per_core"]),
                  pt.get("onchip_arithmetic"))
            check(f"resolved weight_bits {tag}", int(pt.get("weight_bits")) == int(cls["weight_bits_declared"]),
                  pt.get("weight_bits"))

        for arm in ("B5u", "pipe+P+B"):
            dec = arm_declarations(W, arm)
            row_target = json.loads((RA / dec["config"]).read_text())["platform_constraints"]["target"]
            row_values = tc.chip_values(row_target)
            cfg = derived_config(dec, cls, configs, row_values)
            a = json.loads((RA / dec["config"]).read_text())
            fa = OP._flat(a)
            allowed = {"/exploratory", "/platform_constraints/target_overrides/onchip_arithmetic/note",
                       "/platform_constraints/target_overrides/onchip_arithmetic/value/lanes_per_core",
                       "/platform_constraints/target_overrides/weight_bits/note",
                       "/platform_constraints/target_overrides/weight_bits/value"}
            for form in ("chip", "row"):
                fb = OP._flat(json.loads(cfg[form][0].read_text()))
                diff = sorted(kk for kk in set(fa) | set(fb) if fa.get(kk) != fb.get(kk))
                extra = set(diff) - allowed
                if form == "row":
                    extra = {kk for kk in extra if not any(kk.startswith(f"/platform_constraints/target_overrides/{d}/")
                                                           for d in cfg["dropped_on_row"])}
                check(f"derived_config_keys {tag} {arm} {form}", not extra, diff)
            for key in cfg["dropped_on_row"]:
                check(f"left_to_row_target_equal {tag} {arm} {key}",
                      json.loads(cfg["chip"][0].read_text())["platform_constraints"]["target_overrides"][key]["value"]
                      == row_values[key])

    names = set()
    try:
        for cls in live:
            for arm in p["arms"]:
                dec = arm_declarations(W, arm)
                row_target = json.loads((RA / dec["config"]).read_text())["platform_constraints"]["target"]
                cfg = derived_config(dec, cls, configs, tc.chip_values(row_target))
                names.update({cfg["row"][0].name, cfg["chip"][0].name})
        check("derived_configs_every_class_one_name_one_body", True, sorted(names))
    except SystemExit as e:
        check("derived_configs_every_class_one_name_one_body", False, str(e)[:300])

    cls = class_of(p, 192, 48, 4, 1)
    chip = GeometryChip(W, tc, cls)
    try:
        tc.merge({"a": {"e_mac": chip.geometry["e_mac"]}, "b": {"e_mac": chip.geometry["e_mac"]}})
        check("merge_refuses_duplicate", False)
    except OverrideBlockConflictError:
        check("merge_refuses_duplicate", True)
    try:
        tc.merge({"declared_read": chip.declared_read, "geometry": chip.geometry})
        check("merge_refuses_unfiltered_geometry", False)
    except OverrideBlockConflictError:
        check("merge_refuses_unfiltered_geometry", True)
    ok = all(r["ok"] for r in results)
    return {"check": "selftest", "tests": len(results), "failed": [r for r in results if not r["ok"]], "ok": ok,
            "results": results}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    pp = sub.add_parser("part")
    pp.add_argument("--class", dest="cls", required=True, help="R,C,s,k")
    pp.add_argument("--arch", required=True)
    pp.add_argument("--arm", required=True)
    pp.add_argument("--out", required=True, type=Path)
    pp.add_argument("--configs", type=Path, default=OUT_ROOT / "configs")
    pp.add_argument("--counts", default="")
    pp.add_argument("--routed-block", type=Path, default=None)
    pp.add_argument("--routed-block-sha256", default="")
    for m in ("h1", "h2", "h3", "h4", "selftest", "h2routed"):
        x = sub.add_parser(m)
        x.add_argument("--out", required=True, type=Path)
        x.add_argument("--configs", type=Path, default=OUT_ROOT / "configs")
        if m == "h2routed":
            x.add_argument("--routed-block", type=Path, required=True)
            x.add_argument("--routed-block-sha256", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    W, OP, start = bootstrap()
    start["tree"] = tree_check()
    tc = W.Toolchain(WORKTREE)
    p = plan()
    block = None
    if getattr(a, "routed_block", None) is not None:
        block = load_routed_block(a.routed_block, a.routed_block_sha256)
        p = overlay_routed(p, block, a.routed_block_sha256)
        start["routed_block"] = {"path": str(a.routed_block), "sha256": a.routed_block_sha256,
                                 "geometry": f"{block['R']}x{block['C_cells']}x{block['s']}"}

    def log(line: str) -> None:
        print(f"{time.time() - t0:8.1f} {line}", flush=True)
    if a.mode == "part":
        R, C, s, k = (int(x) for x in a.cls.split(","))
        counts = [int(x) for x in a.counts.split(",") if x] or None
        out = price_part(W, OP, tc, p, R, C, s, k, a.arch, a.arm, a.configs, counts=counts, log=log)
    elif a.mode == "h1":
        out = h1(W, OP, tc, log)
    elif a.mode == "h2":
        out = h2(W, OP, tc, p, a.configs, log)
    elif a.mode == "h3":
        out = h3(W, OP, tc, p, a.configs, log)
    elif a.mode == "h4":
        out = h4(W, OP, tc, log)
    elif a.mode == "h2routed":
        out = h2_routed(W, OP, tc, p, block, a.configs, log)
    else:
        out = selftest(W, OP, tc, p, a.configs)
    out["start"] = start
    out["process_wall_s"] = round(time.time() - t0, 1)
    out.setdefault("confirmatory", LABEL_R5.format(what="an output of the pricing harness"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ordered = {"confirmatory": out["confirmatory"], **{k: out[k] for k in sorted(out) if k != "confirmatory"}}
    body = json.dumps(ordered, default=str, separators=(",", ":")) + "\n"
    tmp = a.out.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(body)
    os.replace(tmp, a.out)
    refused = out.get("refusal")
    print(f"wrote {a.out} ({out.get('process_wall_s')} s){' REFUSED: ' + refused if refused else ''}")
    return 3 if refused or (a.mode == "selftest" and not out["ok"]) or (a.mode == "h4" and not out["ok"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
