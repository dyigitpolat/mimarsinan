""
import collections
import hashlib
import json
import sys

KEY = ("target", "network", "budget", "cores", "passes", "weight_bits", "softcores")


def norm(field, value):
    ""
    if field == "budget" and value is not None:
        try:
            return str(float(value))
        except (TypeError, ValueError):
            return str(value)
    return value


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rm_path, ca_path, out_path = sys.argv[1:4]
    rm = json.load(open(rm_path))
    pts = rm["points"]
    groups = collections.OrderedDict()
    geoms = collections.defaultdict(set)
    with_cores = 0
    for p in pts:
        if p.get("cores") is None:
            continue
        with_cores += 1
        k = tuple(norm(f, p.get(f)) for f in KEY)
        groups[k] = groups.get(k, 0) + 1
        geoms[k].add(p.get("geometry"))
    configs = [dict(zip(KEY, k), points=n, geometry=sorted(g for g in geoms[k] if g is not None))
               for k, n in groups.items()]
    ca = json.load(open(ca_path))
    rows = ca["regime"]["rows"]
    ca_keys = collections.Counter()
    for r in rows:
        ca_keys[tuple(norm(f, r.get(f)) for f in KEY)] += r["points"]
    mine = collections.Counter({k: n for k, n in groups.items()})
    check = {"config_area_rows": len(rows), "mine": len(configs), "keys_equal": set(ca_keys) == set(mine),
             "points_equal": ca_keys == mine, "only_mine": len(set(mine) - set(ca_keys)),
             "only_config_area": len(set(ca_keys) - set(mine)),
             "config_area_points": sum(ca_keys.values()), "mine_points": sum(mine.values())}
    out = {"format": "uarea-w9-exploration-configs/1",
           "source": {"regime_map": {"path": rm_path, "sha256": sha(rm_path)},
                      "config_area": {"path": ca_path, "sha256": sha(ca_path)}},
           "points_in_map": len(pts), "points_with_cores": with_cores, "configurations": configs,
           "check_against_config_area": check,
           "per_target_points": dict(collections.Counter({c["target"]: 0 for c in configs}) +
                                     collections.Counter())}
    per = collections.Counter()
    for c in configs:
        per[c["target"]] += c["points"]
    out["per_target_points"] = dict(sorted(per.items()))
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("configurations",)}, indent=1))


if __name__ == "__main__":
    main()
