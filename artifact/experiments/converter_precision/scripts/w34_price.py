""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import w34_points as P

PROFILES = ("isaac_like", "neurram")


def view(pricing) -> Dict[str, Any]:
    terms = [t.to_dict() if hasattr(t, "to_dict") else str(t) for t in pricing.terms]
    refusals = [r.to_dict() if hasattr(r, "to_dict") else str(r) for r in pricing.refusals]
    conv = [t for t in terms if isinstance(t, dict) and any("adc" in json.dumps(v, default=str).lower()
                                                             for v in t.values())]
    return {"terms": {t["name"]: t.get("value") for t in terms if isinstance(t, dict) and "name" in t},
            "refusals": refusals, "converter_terms": conv}


def price_record(raw: bytes) -> Dict[str, Any]:
    from mimarsinan.deployment_record.cost.absolute import report as R
    from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
    from mimarsinan.deployment_record.platform_physics import get_platform_physics
    from mimarsinan.deployment_record.quantities.from_record import from_record
    from mimarsinan.deployment_record.schema import load_deployment_record

    with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as handle:
        handle.write(raw)
    try:
        record = load_deployment_record(handle.name)
    finally:
        Path(handle.name).unlink()
    q = from_record(record)
    bits = R.record_readout_bits(record)
    geometry = R.record_geometry(record)
    out: Dict[str, Any] = {"record_sha256": hashlib.sha256(raw).hexdigest(), "readout_bits": bits}
    own = R.absolute_pricing_for_record(record)
    out["own"] = None if own is None else view(own)
    for profile in PROFILES:
        try:
            p = price_absolute(q, get_platform_physics(profile), writes=R.record_write_inputs(record), compute=None,
                               readout_bits=bits, geometry=R.UNJUDGED if geometry is None else geometry)
            out[profile] = view(p)
        except Exception as exc:
            out[profile] = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
    return out


def price_dirs(dirs: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    return {Path(d).name: price_record(gzip.decompress((Path(d) / "deployment_record.json.gz").read_bytes()))
            for d in dirs}


def price_tier(lock: Dict[str, Any], tier: int) -> Dict[str, Any]:
    ""
    out: Dict[str, Any] = {"points": {}, "not_priced": {}}
    for p in P.tier_points(lock, tier):
        d = P.out_dir(lock, p)
        if (d / "deployment_record.json.gz").is_file():
            out["points"][str(p["id"])] = price_record(gzip.decompress((d / "deployment_record.json.gz").read_bytes()))
        else:
            out["not_priced"][str(p["id"])] = "no sealed record (not run, stopped or refused)"
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", type=Path)
    ap.add_argument("--tier", type=int, choices=(1, 2))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dirs", nargs="+", default=None)
    args = ap.parse_args(argv)
    if args.dirs:
        target, dirs = Path(args.dirs[0]), [Path(d) for d in args.dirs[1:]]
        doc = {"label": "W34 pricing of point directories (non-record unless their lock says so)",
               "points": price_dirs(dirs)}
    else:
        if not (args.lock and args.tier and args.out):
            ap.error("--lock, --tier and --out, or --dirs OUT POINT_DIR...")
        lock, sha = P.load(args.lock)
        doc = {"lock": str(args.lock), "lock_sha256": sha, "tier": args.tier, **price_tier(lock, args.tier)}
        target = args.out
    target.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
