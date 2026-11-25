""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_session_r10


def keyed(readings, spec, keys, readings_path):
    ""
    read = readings["realizations"]
    refused = readings.get("refusals", {})
    derived = keys.derive(readings, rtwo={k: v for k, v in spec["rtwo"].items() if k in read},
                          held_out={k: v for k, v in spec["held_out"].items() if k in read})
    entries = []
    for item in spec["keys"]:
        real = item["realization"]
        if real in read:
            entry = keys.entries(readings, derived, {"keys": [item]}, readings_path=readings_path)[0]
        else:
            reasons = refused.get(real, {}).get("reasons") or [f"the readings hold neither this realization nor its refusal"]
            entry = dict(item["entry"], key=item["key"], value=None, sources=[], refused=reasons,
                         display=f"REFUSED: the realization {real}: " + "; ".join(reasons))
        entry["revision_10"] = {"realization": real, "read": real in read,
                                "bitstreams": readings.get("bitstreams", {}).get(real, []),
                                "lock_on_bitstreams": readings.get("lock_on_bitstreams", {}).get(real),
                                "stated_with": readings.get("stated_with", {}).get(real, [])}
        entries.append(entry)
    return derived, entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--readings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--always", action="append", default=[])
    options = parser.parse_args(argv)
    if os.path.exists(options.out):
        raise SystemExit(f"REFUSING: {options.out} exists")
    try:
        fidmodel, _read, keys = read_session_r10.locked(options.unit)
    except read_session_r10.Refusal as exc:
        raise SystemExit(f"REFUSING: {exc}")
    readings_path = os.path.abspath(options.readings)
    spec_path = os.path.join(options.unit, "keys", "spec.json")
    readings, spec = fidmodel.load_json(readings_path), fidmodel.load_json(spec_path)
    derived, entries = keyed(readings, spec, keys, readings_path)
    read = [f"*.test_{e['key']}" for e in entries if e["revision_10"]["read"]]
    tests = ([f"*.{name}" for name in options.always] + read) if read else []
    doc = {"schema": "ufidfpga_keys/1", "revision": "card plan revision 10 (tools_r10/keys_r10.py over the locked keys)",
           "readings": {"path": readings_path, "sha256": fidmodel.sha256(readings_path)},
           "spec": {"path": os.path.abspath(spec_path), "sha256": fidmodel.sha256(spec_path)},
           "derived": derived, "entries": entries, "derivation_tests": tests,
           "refused_keys": [e["key"] for e in entries if not e["revision_10"]["read"]]}
    with open(options.out, "x", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
        handle.write("\n")
    for e in entries:
        print(f"{e['key']}: {e['display']}")
    print(f"wrote {options.out} sha256 {fidmodel.sha256(options.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
