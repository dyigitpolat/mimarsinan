""
import hashlib
import json
import math
import sys

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import UNJUDGED, CoreGeometry, declared_geometry
from mimarsinan.deployment_record.platform_physics import PHYSICS_CONSTANTS, apply_overrides, get_platform_physics
from mimarsinan.deployment_record.platform_physics.converter import declared_adc_bits
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.mapping.platform.weight_cells import array_totals
from mimarsinan.transformations.quantization_bounds import CORE_WEIGHT_BITS

TAG = sys.argv[1]
INPUTS = json.load(open(sys.argv[2]))
TERM = "chip_area_mm2"

POINTERS = {"row": "/row", "status": "/status", "target": "/record/identity/platform/target",
            "cores": "/record/identity/platform/cores", "weight_bits": "/record/crossbar/weight_bits",
            "platform_weight_bits": "/record/identity/platform/weight_bits",
            "tile_grid_rows": "/record/identity/platform/tile_grid_rows_resolved",
            "tile_grid_cols": "/record/identity/platform/tile_grid_cols_resolved",
            "adc_bits": "/record/identity/platform/readout_law/adc_bits"}
_PHYSICS = {}


def pointer(doc, path):
    cur = doc
    for part in path.strip("/").split("/"):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def physics_of(name):
    ""
    if name not in _PHYSICS:
        p = get_platform_physics(name)
        missing = [k for c in p.undeclared_external() for k in p.missing(c.constants)]
        if missing:
            p = apply_overrides(p, {k: {"nominal": 1.0, "unit": PHYSICS_CONSTANTS[k].display_unit,
                                        "evidence_kind": "estimated", "derivation": "W9 stand-in store",
                                        "note": "W9 stand-in store (no area)"} for k in missing})
        _PHYSICS[name] = (p, missing)
    return _PHYSICS[name]


def priced(name, quantities, geometry, readout_bits):
    p, missing = physics_of(name)
    pr = price_absolute(quantities, p, geometry=geometry, readout_bits=readout_bits)
    term = {t.name: t for t in pr.terms}.get(TERM)
    refusal = {r.name: r.reason for r in pr.refusals}.get(TERM)
    flags = [{"kind": f.kind, "value": f.value, "reason": f.reason} for f in getattr(pr, "flags", ()) if f.name == TERM]
    out = {"refusal": refusal, "flags": flags, "stand_in_constants": missing}
    if term is not None and term.band is not None:
        out.update(nominal=term.band.nominal, low=term.band.low, high=term.band.high,
                   basis=getattr(term.band, "basis", None))
    return out


def record_census(doc):
    ""
    cores = pointer(doc, POINTERS["cores"]) or []
    values, withheld = {}, {}
    widths = sorted({int(c[CORE_WEIGHT_BITS]) for c in cores if c.get(CORE_WEIGHT_BITS) is not None})
    if len(widths) > 1:
        withheld["weight_bits"] = (f"the core types carry weight widths {widths}: no one width prices a cell "
                                   "(per-type per-bit pricing is not built, WI-5g)")
    elif pointer(doc, POINTERS["weight_bits"]) is not None:
        values["weight_bits"] = QuantityValue(float(pointer(doc, POINTERS["weight_bits"])), "static")
    axons, neurons, cells = array_totals(cores)
    values["cores_physical"] = QuantityValue(float(sum(int(c.get("count", 1)) for c in cores)), "static")
    values["axons_physical"] = QuantityValue(float(axons), "static")
    values["neurons_physical"] = QuantityValue(float(neurons), "static")
    values["cells_physical"] = QuantityValue(float(cells), "static")
    rows, cols = (int(pointer(doc, POINTERS[k]) or 0) for k in ("tile_grid_rows", "tile_grid_cols"))
    if rows > 0 and cols > 0:
        values["tiles"] = QuantityValue(float(rows * cols), "static")
    return cores, Quantities(values, withheld=withheld), widths


def price_record(item):
    doc = json.load(open(item["path"]))
    read = {k: pointer(doc, p) for k, p in POINTERS.items()}
    cores, q, widths = record_census(doc)
    geometry = declared_geometry(cores)
    platform = pointer(doc, "/record/identity/platform") or {}
    readout = declared_adc_bits(platform)
    name = read["target"]
    out = {"row": read["row"], "status": read["status"], "description": name, "source": item["path"],
           "source_sha256": sha(item["path"]), "pointers": POINTERS, "read": {k: v for k, v in read.items()
                                                                              if k != "cores"},
           "core_types": cores, "census": {k: q.get(k).value for k in ("cores_physical", "axons_physical",
                                                                       "neurons_physical", "cells_physical",
                                                                       "weight_bits", "tiles") if q.has(k)},
           "census_withheld": {k: q.withheld_reason(k) for k in ("weight_bits",) if q.withheld_reason(k)},
           "readout_bits": readout, "geometry": "declared core types" if geometry is not None else "UNJUDGED"}
    out.update(priced(name, q, UNJUDGED if geometry is None else geometry, readout))
    if len(widths) > 1:
        parts = []
        for core in cores:
            vals = {"weight_bits": QuantityValue(float(core[CORE_WEIGHT_BITS]), "static")}
            a, n, c = array_totals([core])
            vals.update(cores_physical=QuantityValue(float(core.get("count", 1)), "static"),
                        axons_physical=QuantityValue(float(a), "static"),
                        neurons_physical=QuantityValue(float(n), "static"),
                        cells_physical=QuantityValue(float(c), "static"))
            parts.append(priced(name, Quantities(vals), declared_geometry([core]), readout))
        if all(p.get("nominal") is not None for p in parts):
            out["per_type_reading"] = {"nominal": sum(p["nominal"] for p in parts),
                                       "low": sum(p["low"] for p in parts), "high": sum(p["high"] for p in parts),
                                       "parts": [{k: p.get(k) for k in ("nominal", "low", "high")} for p in parts],
                                       "what": "a reading beside the price: each core type priced alone at its own "
                                               "weight width (lanes out), summed; not the unit's price"}
        else:
            out["per_type_reading"] = {"refused": [p.get("refusal") for p in parts]}
    return out


def explore(cfgs, cpt):
    rows = []
    for c in cfgs:
        name, n, w = c["target"], c["cores"], c["weight_bits"]
        p, _missing = physics_of(name)
        g = p.validity.geometry
        vals = {"cores_physical": n, "axons_physical": n * g.axons, "neurons_physical": n * g.neurons,
                "cells_physical": n * g.axons * g.neurons, "weight_bits": w,
                "tiles": n if cpt.get(name) is None else math.ceil(n / cpt[name])}
        q = Quantities({k: QuantityValue(float(x), "static") for k, x in vals.items() if x is not None})
        got = priced(name, q, [CoreGeometry(g.axons, g.neurons, n)], None)
        rows.append({**{k: c[k] for k in ("target", "network", "budget", "cores", "passes", "weight_bits",
                                          "softcores", "points")},
                     "census": vals, "core": [g.axons, g.neurons], **got})
    return rows


def identities():
    out = {}
    for name, n_pub in (("truenorth", 4096), ("loihi", 128), ("hermes_64", 64), ("neurram", 48), ("odin", 1),
                        ("sram_cim_macro", 1)):
        p, _m = physics_of(name)
        g = p.validity.geometry
        vals = {"cores_physical": n_pub, "axons_physical": n_pub * g.axons, "neurons_physical": n_pub * g.neurons,
                "cells_physical": n_pub * g.axons * g.neurons, "weight_bits": g.cell_bits, "tiles": n_pub}
        q = Quantities({k: QuantityValue(float(x), "static") for k, x in vals.items() if x is not None})
        out[name] = {"cores": n_pub, **priced(name, q, [CoreGeometry(g.axons, g.neurons, n_pub)], None)}
    return out


def main():
    out = {"format": "uarea-w9-prices/1", "tag": TAG, "term": TERM}
    out["routeb"] = [price_record(i) for i in INPUTS["routeb"]]
    out["ladder"] = [price_record(i) for i in INPUTS["ladder"]]
    cfg = json.load(open(INPUTS["explore"]["configs"]))
    cpt = {k: x.get("cpt") for k, x in json.load(open(INPUTS["explore"]["config_area"]))["laws"].items()}
    out["explore_cpt"] = cpt
    out["explore"] = explore(cfg["configurations"], cpt)
    out["identities"] = identities()
    json.dump(out, open(sys.argv[3], "w"), indent=1, default=str)
    for s in ("routeb", "ladder", "explore"):
        rows = out[s]
        print(TAG, s, len(rows), "priced", sum(1 for r in rows if r.get("nominal") is not None),
              "refused", sum(1 for r in rows if r.get("refusal")))


if __name__ == "__main__":
    main()
