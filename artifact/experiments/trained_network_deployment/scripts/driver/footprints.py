""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import common
import configs
import inventory

FOOTPRINTS = common.RECORDS / "footprints"
RESOLVED = "Model Configuration.platform_constraints_resolved.json"
UTILIZATION = "crossbar_utilization.json"
PACKING_KEYS = ("cores_geometry", "allow_coalescing", "allow_neuron_splitting", "core_semantics", "weight_bits",
                "pruning")


class FootprintRefusal(SystemExit):
    ""


def record_path(member: str, geometry: str) -> Path:
    return FOOTPRINTS / f"{member}.{geometry}.json"


def packing_declaration(resolved: dict) -> dict:
    ""
    cores = resolved.get("cores") or []
    return {"cores_geometry": [{k: c.get(k) for k in ("max_axons", "max_neurons", "has_bias")} for c in cores],
            **{k: resolved.get(k) for k in PACKING_KEYS if k != "cores_geometry"}}


def measure_work(work: Path, geometry: str) -> dict:
    ""
    work = Path(work)
    resolved_file, util_file = work / RESOLVED, work / UTILIZATION
    for f in (resolved_file, util_file):
        if not f.is_file():
            raise FootprintRefusal(f"REFUSED: {f} is missing")
    resolved = json.loads(resolved_file.read_text())
    axons, neurons = configs.GEOMETRIES[geometry]
    cores = resolved.get("cores") or []
    if len(cores) != 1 or (cores[0].get("max_axons"), cores[0].get("max_neurons")) != (axons, neurons):
        raise FootprintRefusal(f"REFUSED: {work} declares cores {cores}, not one {geometry} type")
    if not cores[0].get("has_bias"):
        raise FootprintRefusal(f"REFUSED: {work} declares cores without on-chip bias")
    if resolved.get("allow_scheduling"):
        raise FootprintRefusal(f"REFUSED: {work} is a scheduled pack, not a resident one")
    util = json.loads(util_file.read_text())
    phys_axons, cells = int(util["axons_physical"]), int(util["cells_physical"])
    if phys_axons % axons or cells % (axons * neurons) or phys_axons // axons != cells // (axons * neurons):
        raise FootprintRefusal(f"REFUSED: {work}: axons_physical {phys_axons} and cells_physical {cells} do not "
                               f"give one whole count of {geometry} cores")
    footprint = phys_axons // axons
    if footprint > int(cores[0]["count"]):
        raise FootprintRefusal(f"REFUSED: {work}: {footprint} physical cores exceed the declared {cores[0]['count']}")
    return {"F": footprint, "listed_cores": int(util["cores_allocated"]), "source_declared_count": int(cores[0]["count"]),
            "packing": packing_declaration(resolved), "source_work": str(work),
            "files": {RESOLVED: common.sha256_file(resolved_file), UTILIZATION: common.sha256_file(util_file)}}


def write(member: str, geometry: str, work: Path, note: str = "") -> dict:
    if member not in inventory.MEMBERS:
        raise FootprintRefusal(f"REFUSED: {member} is no member of the inventory")
    rec = {"format": "e2e-footprint/1", "member": member, "geometry": geometry, **measure_work(work, geometry),
           "note": note, "driver_code_listing_sha256": common.listing_sha256(common.driver_code_sha256())}
    common.atomic_write_json(record_path(member, geometry), rec)
    return rec


def read(member: str, geometry: str) -> dict | None:
    ""
    path = record_path(member, geometry)
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    work = Path(rec["source_work"])
    for name, digest in rec["files"].items():
        if not (work / name).is_file() or common.sha256_file(work / name) != digest:
            raise FootprintRefusal(f"REFUSED: {path}: source file {work / name} does not verify")
    rec["record_sha256"] = common.sha256_file(path)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--geometry", required=True, choices=sorted(configs.GEOMETRIES))
    ap.add_argument("--work", required=True, help="a pipeline working directory holding a resident pack")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    rec = write(a.member, a.geometry, Path(a.work), a.note)
    print(json.dumps({k: rec[k] for k in ("member", "geometry", "F", "listed_cores", "source_declared_count")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
