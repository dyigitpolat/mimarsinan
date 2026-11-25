""
import argparse
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime

FORMS = ("P-A", "P-B")
PROGRAMMING_RUN = "programming_run"


class ModelRefusal(Exception):
    ""


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def field(doc, dotted):
    value = doc
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ModelRefusal(f"the description has no field {dotted!r}")
        value = value[part]
    return value


def description(model, key):
    real = model["realizations"][key]
    path = os.path.join(model["co"], real["description"]["path"])
    pinned = real["description"].get("sha256")
    if pinned and sha256(path) != pinned:
        raise ModelRefusal(f"{path} does not hash to the model's pin {pinned}")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return json.loads(text), text


def clock_hz(model, key, text):
    ""
    clock = model["realizations"][key].get("clock_hz")
    if clock is None:
        raise ModelRefusal(f"realization {key} prices a time constant but declares no clock")
    quote = clock["quote"]
    if quote not in text:
        raise ModelRefusal(f"the clock quote {quote!r} is not in the description of {key}")
    found = re.search(r"(?P<number>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>[A-Za-z]+)\s*$", quote)
    hertz = model.get("units_hertz", {})
    if not found or found.group("unit") not in hertz:
        raise ModelRefusal(f"the clock quote {quote!r} states no frequency in a unit the model declares")
    stated = float(found.group("number")) * float(hertz[found.group("unit")])
    if not math.isclose(stated, float(clock["value"])):
        raise ModelRefusal(f"the clock value {clock['value']} differs from its quote {quote!r} ({stated})")
    return float(clock["value"])


def coefficient_cycles(model, key, coeff, desc, text):
    if "profile_constant" in coeff:
        constant = field(desc, "constants." + coeff["profile_constant"])
        units = model.get("units_seconds", {})
        if constant.get("unit") not in units:
            raise ModelRefusal(f"{coeff['profile_constant']} is in unit {constant.get('unit')!r}, which the model does not declare")
        cycles = float(constant["nominal"]) * float(units[constant["unit"]])
        if coeff.get("times_clock"):
            cycles *= clock_hz(model, key, text)
    elif "value" in coeff:
        cycles = float(coeff["value"])
    else:
        raise ModelRefusal(f"a coefficient of {key} names neither a profile constant nor a value")
    if coeff.get("times_description_field"):
        cycles *= float(field(desc, coeff["times_description_field"]))
    return cycles


def coefficients(model, key):
    ""
    desc, text = description(model, key)
    out = {}
    for name, term in model["realizations"][key]["terms"].items():
        out[name] = {"per": term["per"], "cycles": coefficient_cycles(model, key, term["coefficient"], desc, text)}
    out["barrier"] = {"per": "barrier_cycle", "cycles": float(model["barrier"]["cycles_per_barrier_cycle"]["value"])}
    return out


def quantities(model, core, *, cycles, stimulus_words, kind):
    ""
    if kind == PROGRAMMING_RUN:
        streamed, prefix = True, 0
    elif kind in model["arms"]:
        streamed, prefix = bool(model["arms"][kind]["program_streamed"]), int(core["clear_prefix_words"])
    else:
        raise ModelRefusal(f"run kind {kind!r} is neither an arm of the model nor the programming run")
    g = {name: float(entry["value"]) for name, entry in model["grammar"].items()}
    words = float(stimulus_words)
    fetch = cycles * (g["tag_words_per_cycle"] + g["wait_words_per_cycle"])
    return {
        "load": int(streamed),
        "program_word": int(core["program_words"]) if streamed else 0,
        "barrier_cycle": int(core["barrier_cycles"]) * int(cycles),
        "event": {"P-A": words / g["words_per_event"],
                  "P-B": (words - prefix - fetch - g["end_words"]) / g["words_per_event"]},
    }


def predict(coeffs, q):
    out = {}
    for form in FORMS:
        total = coeffs["barrier"]["cycles"] * q["barrier_cycle"]
        for name, term in coeffs.items():
            if name == "barrier":
                continue
            amount = q[term["per"]]
            total += term["cycles"] * (amount[form] if isinstance(amount, dict) else amount)
        out[form] = total
    return out


NAMED_COLUMNS = ("sample", "core", "capture_events", "device_cycles", "stimulus_words")


def tsv_rows(path):
    ""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [c for c in NAMED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ModelRefusal(f"{path} lacks the columns {missing}")
        return [{c: int(row[c]) for c in NAMED_COLUMNS} for row in reader]


def census(bundle, tsv_path):
    ""
    order = [int(c) for c in bundle["pass_order"]]
    cores = {}
    for plan in bundle["cores"]:
        cores[str(int(plan["core"]))] = {"program_words": int(plan["program"]["words"]),
                                         "barrier_cycles": int(plan["barrier_cycles"]),
                                         "clear_prefix_words": len(plan["clear_prefix"])}
    frozen = {int(s["index"]): int(s["stimulus"]["words"]) for s in bundle["samples"]}
    rows = tsv_rows(tsv_path)
    samples = sorted({r["sample"] for r in rows})
    seen = {(r["sample"], r["core"]): r for r in rows}
    if len(seen) != len(rows) or len(rows) != len(samples) * len(order):
        raise ModelRefusal(f"{tsv_path}: {len(rows)} rows are not one per sample and pass ({len(samples)} x {len(order)})")
    passes = []
    for index, core in enumerate(order):
        for sample in samples:
            row = seen.get((sample, core))
            if row is None:
                raise ModelRefusal(f"{tsv_path}: no row for sample {sample}, core {core}")
            if index == 0 and row["stimulus_words"] != frozen.get(sample):
                raise ModelRefusal(f"sample {sample}: the replay's first pass has {row['stimulus_words']} words, the "
                                   f"bundle's frozen stimulus {frozen.get(sample)}")
            passes.append({"sample": sample, "core": core, "pass_index": index,
                           "stimulus_words": row["stimulus_words"], "capture_events": row["capture_events"]})
    return {"pass_order": order, "cycles_per_sample": int(bundle["cycles_per_sample"]), "cores": cores,
            "samples": samples, "passes": passes}


def build_lock(model_path, realizations, *, campaign_pattern):
    ""
    model = load_json(model_path)
    re.compile(campaign_pattern)
    lock = {"schema": "ufidfpga_prediction_lock/1",
            "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": {"path": os.path.abspath(model_path), "sha256": sha256(model_path)},
            "tools": {name: sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), name))
                      for name in sorted(os.listdir(os.path.dirname(os.path.abspath(__file__)))) if name.endswith(".py")},
            "campaign_pattern": campaign_pattern, "arms": sorted(model["arms"]), "forms": list(FORMS),
            "realizations": {}}
    for key, (bundle_path, tsv_path) in realizations.items():
        if key not in model["realizations"]:
            raise ModelRefusal(f"the model has no realization {key!r}")
        bundle = load_json(bundle_path)
        if bundle["self_hash"] != model["realizations"][key]["bundle_self_hash"]:
            raise ModelRefusal(f"{bundle_path} is {bundle['self_hash']}, not the model's {model['realizations'][key]['bundle_self_hash']}")
        coeffs = coefficients(model, key)
        cen = census(bundle, tsv_path)
        cycles = cen["cycles_per_sample"]
        for p in cen["passes"]:
            core = cen["cores"][str(p["core"])]
            p["pred"] = {arm: predict(coeffs, quantities(model, core, cycles=cycles, stimulus_words=p["stimulus_words"], kind=arm))
                         for arm in sorted(model["arms"])}
        programming = {c: predict(coeffs, quantities(model, core, cycles=0, stimulus_words=int(model["grammar"]["end_words"]["value"]),
                                                     kind=PROGRAMMING_RUN))
                       for c, core in cen["cores"].items()}
        lock["realizations"][key] = {
            "bundle": {"path": os.path.abspath(bundle_path), "sha256": sha256(bundle_path), "self_hash": bundle["self_hash"]},
            "census_tsv": {"path": os.path.abspath(tsv_path), "sha256": sha256(tsv_path)},
            "coefficients": coeffs, "pass_order": cen["pass_order"], "cycles_per_sample": cycles,
            "cores": cen["cores"], "samples": cen["samples"], "passes": cen["passes"], "programming_run": programming}
    return lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("lock")
    lk.add_argument("--model", required=True)
    lk.add_argument("--realization", action="append", required=True, help="KEY=BUNDLE,CENSUS_TSV")
    lk.add_argument("--pattern", required=True)
    lk.add_argument("--out", required=True)
    options = parser.parse_args(argv)
    reals = {}
    for spec in options.realization:
        key, paths = spec.split("=", 1)
        bundle_path, tsv_path = paths.split(",", 1)
        reals[key] = (bundle_path, tsv_path)
    if os.path.exists(options.out):
        raise SystemExit(f"REFUSING: {options.out} exists; a lock is never rewritten")
    lock = build_lock(options.model, reals, campaign_pattern=options.pattern)
    with open(options.out, "x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=1, sort_keys=True)
        handle.write("\n")
    for key, real in lock["realizations"].items():
        c = real["coefficients"]
        print(f"{key}: {len(real['passes'])} passes, coefficients " +
              ", ".join(f"{n} {t['cycles']:.6g} per {t['per']}" for n, t in sorted(c.items())))
    print(f"wrote {options.out} sha256 {sha256(options.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
