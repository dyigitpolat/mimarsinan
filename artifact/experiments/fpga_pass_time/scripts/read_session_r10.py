""
import argparse
import hashlib
import json
import os
import re
import sys


class Refusal(Exception):
    ""


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def locked(unit):
    ""
    sums = os.path.join(unit, "prediction_lock", "SHA256SUMS")
    rows = {}
    with open(sums, encoding="utf-8") as handle:
        for line in handle:
            value, _, rel = line.rstrip("\n").partition("  ")
            rows[rel] = value
    tools = os.path.join(unit, "tools")
    for name in sorted(os.listdir(tools)):
        if name.endswith(".py"):
            rel = f"tools/{name}"
            if rows.get(rel) != digest(os.path.join(tools, name)):
                raise Refusal(f"{rel} of {unit} does not equal its row of {sums}: the tools are not the lock's")
    sys.path.insert(0, tools)
    import fidmodel
    import keys
    import read_session
    return fidmodel, read_session, keys


def pins_value(path, key):
    ""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields[0] == key and len(fields) > 1:
                return fields[1]
    raise Refusal(f"no pins row {key} in {path}")


def loops(spec, rounds):
    ""
    return {(int(nc), arm, rnd) for nc in spec["core_counts"] for arm in spec["arms"] for rnd in range(1, int(rounds) + 1)}


def last_line(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    return lines[-1] if lines else ""


def job_whole(folder, log, pattern, expected, rule):
    ""
    reasons = []
    if not os.path.isfile(log):
        reasons.append(f"no log {log}")
    else:
        last = last_line(log)
        if last != rule["result_line"]:
            reasons.append(f"its log {os.path.basename(log)} ends {last!r}, not {rule['result_line']!r}")
    steps = os.path.join(folder, "steps.tsv")
    rows = []
    if os.path.isfile(steps):
        with open(steps, encoding="utf-8", errors="replace") as handle:
            rows = [line.rstrip("\n").split("\t") for line in handle][1:]
    rows = [row for row in rows if row and row[0]]
    failed = [row[0] for row in rows if len(row) > 1 and row[1] in rule["failed_statuses"]]
    if not rows:
        reasons.append("its steps table has no rows")
    elif failed:
        reasons.append(f"{len(failed)} of {len(rows)} steps {' or '.join(rule['failed_statuses'])}")
    found = set()
    for name in sorted(os.listdir(folder)):
        m = pattern.match(name)
        if m and os.path.isdir(os.path.join(folder, name)):
            found.add((int(m.group("nc")), m.group("arm"), int(m.group("round"))))
    named = lambda keys: ", ".join(rule["campaign_name"].format(nc=nc, arm=arm, round=rnd) for nc, arm, rnd in sorted(keys))
    missing, extra = expected - found, found - expected
    if missing or extra:
        words = f"{len(found)} campaign folders of the {len(expected)} its loops give"
        if missing:
            words += f"; missing {named(missing)}"
        if extra:
            words += f"; not of its loops {named(extra)}"
        reasons.append(words)
    return reasons


def bitstreams(folder, rule, known):
    ""
    path = os.path.join(folder, "identity.txt")
    if not os.path.isfile(path):
        return [], [f"no identity.txt in {folder}"]
    rows = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < len(("pin", "row", "value")) or fields[0] != "pin" or not fields[1].startswith(rule["bitstream_row_prefix"]):
                continue
            _, row, *rest = fields
            entry = rows.setdefault(row, {"row": row})
            if rest[0] == "UUID":
                entry["uuid"] = rest[-1]
            elif "sha256" in rest:
                entry["path"], entry["sha256"] = rest[0], rest[rest.index("sha256") + 1]
    out, reasons = [], []
    for row, entry in sorted(rows.items()):
        k = known.get(entry.get("sha256"))
        if k is None:
            reasons.append(f"bitstream {entry.get('sha256')} of pins row {row} is not one the data states")
            continue
        if entry.get("uuid") != k["uuid"]:
            reasons.append(f"pins row {row} has UUID {entry.get('uuid')}, not the data's {k['uuid']} for {entry['sha256']}")
            continue
        out.append(dict(entry, wrapper=k["wrapper"], slave=k["slave"], build_record=k.get("build_record"), what=k["what"]))
    if not rows:
        reasons.append(f"identity.txt in {folder} checks no pins row {rule['bitstream_row_prefix']}*")
    return out, reasons


def named_bitstreams(text, known, short):
    ""
    out = []
    for sha, k in sorted(known.items()):
        forms = [sha, sha[:short], k["uuid"], k["uuid"][:short], *k.get("paths", [])]
        if any(form and form in text for form in forms):
            out.append(sha)
    return out


def lock_on(statement, spec, key, own, known, says, short):
    ""
    spec_keys = {item["key"]: named_bitstreams(json.dumps(item), known, short) for item in spec["keys"] if item["realization"] == key}
    mine = [b["sha256"] for b in own]
    departs = {k: [s for s in v if s not in mine] for k, v in spec_keys.items()}
    return {"statement_names": named_bitstreams(statement, known, short), "spec_keys": spec_keys, "own": mine,
            "departs": {k: v for k, v in departs.items() if v}, "says": says}


def read_all(unit, pins, data, earlier, jobs):
    fidmodel, read_session, _keys = locked(unit)
    lock_path = os.path.join(unit, "prediction_lock", "predictions.json")
    statement_path = os.path.join(unit, "prediction_lock", "lock_statement.json")
    spec_path = os.path.join(unit, "keys", "spec.json")
    lock, spec = fidmodel.load_json(lock_path), fidmodel.load_json(spec_path)
    with open(statement_path, encoding="utf-8") as handle:
        statement = handle.read()
    unknown = sorted(set(jobs) - set(data["realizations"]))
    if unknown:
        raise Refusal(f"jobs given for realizations the data does not name: {unknown}")
    pattern, rule, known = re.compile(lock["campaign_pattern"]), data["job_rule"], data["bitstreams"]
    short = int(data["short_digest_length"])
    out = read_session.read(lock, {})
    out.update(revision="card plan revision 10 (tools_r10/read_session_r10.py over the locked read)", refusals={}, jobs={},
               bitstreams={}, lock_on_bitstreams={}, stated_with={})
    for key, spec_r in data["realizations"].items():
        folder, log = jobs.get(key, (None, None))
        reasons, own = [], []
        if folder is None:
            reasons.append(f"absent: no {spec_r['job']} job was given for it")
        elif not os.path.isdir(folder):
            reasons.append(f"absent: no folder {folder}")
        else:
            rounds = pins_value(pins, spec_r["rounds_row"])
            reasons += job_whole(folder, log, pattern, loops(spec_r, rounds), rule)
            own, why = bitstreams(folder, rule, known)
            reasons += why
        if not reasons:
            try:
                out["realizations"].update(read_session.read(lock, {key: [folder]})["realizations"])
            except fidmodel.ModelRefusal as exc:
                reasons.append(f"the locked read refused it: {exc}")
        out["jobs"][key] = {"job": spec_r["job"], "folder": folder and os.path.abspath(folder), "log": log and os.path.abspath(log),
                            "read": not reasons}
        if reasons:
            out["refusals"][key] = {"job": spec_r["job"], "folder": folder and os.path.abspath(folder), "reasons": reasons}
        out["bitstreams"][key] = own
        out["lock_on_bitstreams"][key] = lock_on(statement, spec, key, own, known, data.get("lock_reading", {}).get(key), short)
    for path in earlier:
        doc = fidmodel.load_json(path)
        out["stated_with"].setdefault(doc["realization"], []).append(dict(doc, source={"path": os.path.abspath(path), "sha256": digest(path)}))
    for key in data["realizations"]:
        out["stated_with"].setdefault(key, [])
    out["lock"] = {"path": os.path.abspath(lock_path), "sha256": digest(lock_path)}
    out["lock_statement"] = {"path": os.path.abspath(statement_path), "sha256": digest(statement_path)}
    out["spec"] = {"path": os.path.abspath(spec_path), "sha256": digest(spec_path)}
    return out


def lines(readings, data):
    ""
    short = int(data["short_digest_length"])
    out = []
    for key in data["realizations"]:
        if key in readings["realizations"]:
            real = readings["realizations"][key]
            out.append(f"{key}: read: {len(real['campaigns'])} campaigns, census mismatches {real['census_mismatches']}, "
                       f"passed {sum(c['passed'] for c in real['campaigns'])}")
        else:
            out.append(f"{key}: REFUSED: {'; '.join(readings['refusals'][key]['reasons'])}")
        for b in readings["bitstreams"][key]:
            out.append(f"{key}: bitstream {b['row']} sha256 {b['sha256'][:short]} UUID {b['uuid'][:short]} wrapper "
                       f"{b['wrapper'][:short]} ({b['slave']} slave)")
        lk = readings["lock_on_bitstreams"][key]
        out.append(f"{key}: the lock: {lk['says']}")
        out.append(f"{key}: the lock's statement names {[s[:short] for s in lk['statement_names']] or 'no bitstream'}; "
                   f"the key spec's text names {({k: [s[:short] for s in v] for k, v in lk['spec_keys'].items()})}; "
                   f"not this reading's: {({k: [s[:short] for s in v] for k, v in lk['departs'].items()}) or 'none'}")
        for doc in readings["stated_with"][key]:
            counts = ", ".join(f"{k} {v}" for k, v in doc.items() if isinstance(v, int))
            out.append(f"{key}: stated with it: {os.path.basename(doc['source']['path'])} {doc['source']['sha256'][:short]}: {counts}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--pins", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--earlier", action="append", default=[])
    parser.add_argument("--job", action="append", default=[], help="KEY=DIR,LOG")
    parser.add_argument("--out", required=True)
    options = parser.parse_args(argv)
    if os.path.exists(options.out):
        raise SystemExit(f"REFUSING: {options.out} exists")
    jobs = {}
    for spec in options.job:
        key, _, rest = spec.partition("=")
        folder, _, log = rest.rpartition(",")
        if key in jobs or not folder or not log:
            raise SystemExit(f"REFUSING: --job {spec!r} is not one KEY=DIR,LOG of its own key")
        jobs[key] = (folder, log)
    with open(options.data, encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        readings = read_all(options.unit, options.pins, data, options.earlier, jobs)
    except Refusal as exc:
        raise SystemExit(f"REFUSING: {exc}")
    readings["data"] = {"path": os.path.abspath(options.data), "sha256": digest(options.data)}
    with open(options.out, "x", encoding="utf-8") as handle:
        json.dump(readings, handle, indent=1, sort_keys=True)
        handle.write("\n")
    for line in lines(readings, data):
        print(line)
    print(f"wrote {options.out} sha256 {digest(options.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
