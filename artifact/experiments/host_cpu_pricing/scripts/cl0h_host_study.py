""
from __future__ import annotations

import contextlib
import copy
import functools
import importlib.util
import io
import itertools
import json
import math
import multiprocessing
import os
import sys
import time
import types
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
RA = next(p for p in itertools.accumulate([HERE] * 12, lambda p, _: os.path.dirname(p))
          if os.path.basename(p) == "research_artifacts")
HARNESS_PATH = os.path.join(RA, "10_wave1", "runs", "cost_identity_w1.py")
HOST_REFUSAL_MARKERS = ("host work exists", "the host share is unknown")

MEASURED_WALL_MARKERS = ("host_ops_s", "measured host wall")
SOURCE_KINDS = ("description", "artifact")


def _harness():
    spec = importlib.util.spec_from_file_location("cost_identity_w1", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_block_of(spec: str) -> dict:
    ""
    kind, _, rest = spec.partition(":")
    if kind == "description":
        from mimarsinan.deployment_record.platform_physics.hosts.description import (
            DEFAULT_RULE, host_description_block)
        name, _, mode = rest.partition("@")
        return host_description_block(name, mode or DEFAULT_RULE)
    if kind == "artifact":
        from mimarsinan.deployment_record.platform_physics.host_calibration import load_host_block
        return load_host_block(rest)
    raise ValueError(f"host source kind {kind!r} is not one of {list(SOURCE_KINDS)} (spec {spec!r})")


def host_overrides(spec: str) -> dict:
    from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks
    return merge_override_blocks({"host": host_block_of(spec)})


def lacking_rows(block) -> list:
    ""
    from mimarsinan.deployment_record.platform_physics.constants import keys_in_group
    from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
    return [key for key in keys_in_group(HOST) if key not in block]


def _declared_band(row) -> list:
    return [float(row.get(end, row["nominal"])) for end in ("low", "nominal", "high")]


def _lacking_refusal(spec: str, key: str, identified: bool):
    ""
    if not identified:
        return None
    reason = "not declared"
    kind, _, rest = spec.partition(":")
    if kind == "description":
        from mimarsinan.deployment_record.platform_physics.hosts.description import host_description, select_mode
        name, _, mode = rest.partition("@")
        description = host_description(name)
        facts = description.modes[select_mode(description, mode or "default")]
        reason = (facts.get("refused") or {}).get(key, f"the description {name!r} declares no {key}")
    return (f"{key}: {reason}; the identified host replaces the profile's host rows whole "
            "(hosts.description.require_one_host), so no profile or measured value fills it and no term reads it")


def block_census(spec: str, profiles=()) -> dict:
    ""
    from mimarsinan.deployment_record.platform_physics.constants import spec_for
    from mimarsinan.deployment_record.platform_physics.hosts.description import HOST_IDENTITY_FIELD
    from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
    from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
    from mimarsinan.deployment_record.units import to_canonical
    block = host_block_of(spec)
    rows = []
    for key, row in block.items():
        source = "; ".join(str(row[f]) for f in ("citation", "derivation") if row.get(f)) or str(row.get("note", ""))
        rows.append({"key": key, "band": _declared_band(row), "unit": row.get("unit"),
                     "evidence_kind": row.get("evidence_kind"), "source": source,
                     "host": row.get(HOST_IDENTITY_FIELD), "note": row.get("note")})
    identified = all(r["host"] for r in rows)
    lacking = [{"key": key, "refused": _lacking_refusal(spec, key, identified),
                "kept_from_profile": not identified} for key in lacking_rows(block)]
    measured = sorted(r["key"] for r in rows if r["evidence_kind"] == "measured")
    per_profile = {}
    overrides = host_overrides(spec)
    for profile in profiles:
        physics = resolve_platform_physics(profile, overrides)
        host_rows = sorted(k for k in physics.constants if spec_for(k).group == HOST)
        equals = all(math.isclose(a, b, rel_tol=1e-12) for key, row in block.items()
                     for a, b in zip([physics.band(key).low, physics.band(key).nominal, physics.band(key).high],
                                     [to_canonical(v, row.get("unit")) for v in _declared_band(row)]))
        per_profile[profile] = {"host_rows": host_rows, "equals_block": equals,
                                "only_block_rows": set(host_rows) == set(block) if identified else None}
    checks = {"no row measured": not measured, "every row names its host": identified,
              "every profile resolves to the block's rows": all(p["equals_block"] for p in per_profile.values()),
              "no lacking row resolves": all(not any(r["key"] in p["host_rows"] for r in lacking)
                                             for p in per_profile.values()) if identified else None}
    return {"spec": spec, "rows": rows, "lacking": lacking, "measured_rows": measured, "checks": checks,
            "profiles": per_profile}


def reference_profiles(points_path: str) -> tuple:
    ""
    seen: dict = {}
    for line in open(points_path):
        if line.strip():
            seen.setdefault(json.loads(line)["profile"], None)
    return tuple(seen)


def _band(band):
    return None if band is None else [band.low, band.nominal, band.high]


def _report_row(report) -> dict:
    ""
    from mimarsinan.deployment_record.cost.absolute import host_energy_split
    from mimarsinan.deployment_record.cost.host_terms import HOST_DECOMPOSITION
    terms = {t.name: t for t in report.energy + report.latency}
    refused = {n.split(":", 1)[0][len("refused "):]: n.split(":", 1)[1].strip()
               for n in report.notes if n.startswith("refused ")}
    split = host_energy_split(report.energy)
    reason = refused.get("energy_per_inference_mj")
    return {
        "energy": _band(terms["energy_per_inference_mj"].band) if "energy_per_inference_mj" in terms else None,
        "energy_refusal": reason,
        "energy_host_refusal": reason is not None and any(m in reason for m in HOST_REFUSAL_MARKERS),
        "energy_source": terms["energy_per_inference_mj"].source if "energy_per_inference_mj" in terms else None,
        "without_host": None if split is None else _band(split.without_host),
        "host": None if split is None else _band(split.host),
        "host_share": None if split is None else list(split.host_share()),
        "e2e": _band(terms["e2e_latency_s"].band) if "e2e_latency_s" in terms else None,
        "e2e_refusal": refused.get("e2e_latency_s"),
        "latency_host": _band(terms["latency_host_s"].band) if "latency_host_s" in terms else None,
        "host_terms": {name: _band(terms[name].band) for name in HOST_DECOMPOSITION if name in terms},
        "host_terms_refused": {name: refused[name] for name in HOST_DECOMPOSITION if name in refused},
        "host_source": terms["energy_host_mj"].source if "energy_host_mj" in terms else None,
        "reads_measured_wall": any(m in str(t.source) for t in terms.values() for m in MEASURED_WALL_MARKERS),
    }


def _quantity(view, key):
    q = view.quantities
    return q.get(key).value if q.has(key) else None


def config_rows(path: str, spec: str, profiles: tuple) -> list:
    h = _harness()
    facts = h._provider_facts()
    overrides = host_overrides(spec)
    cfg = json.load(open(path))
    rel = os.path.relpath(path, h._tree_root())
    with contextlib.redirect_stdout(io.StringIO()):
        config = h.resolve(cfg, facts)
    shared: dict = {}
    overlay = {"model_config": {}, "platform_constraints": {}}
    candidate = None
    rows = []
    for profile, variant in itertools.product(profiles, ("chip", "host")):
        t0 = time.time()
        row = {"kind": "candidate", "config": rel, "profile": profile, "variant": variant,
               "network": [cfg.get("data_provider_name"), cfg.get("deployment_parameters", {}).get("model_type")]}
        point = {**config, "platform_physics_profile": profile}
        if variant == "host":
            point["platform_physics_overrides"] = copy.deepcopy(overrides)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                problem = h._problem(point, shared)
                if str(config.get("hw_config_mode", "fixed")) == "search" and candidate is None:
                    import numpy as np
                    decoded = problem.decode(np.random.default_rng(0).uniform(problem.xl, problem.xu))
                    candidate = {"cores": decoded["platform_constraints"]["cores"]}
                if candidate is not None:
                    overlay = {"model_config": {}, "platform_constraints": dict(candidate)}
                view = problem.candidate_layout(copy.deepcopy(overlay)).view
                report = view.cost_report()
            if report is None:
                row["status"] = "no physics"
            else:
                row.update(_report_row(report))
                row["host_macs"] = _quantity(view, "host_macs")
                row["compute_op_count"] = _quantity(view, "compute_op_count")
                row["status"] = "ok"
        except Exception as exc:
            row["status"] = "refused"
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240] if str(exc) else ''}"
        row["wall_s"] = round(time.time() - t0, 2)
        rows.append(row)
    return rows


def cmd_points(out: str, workers: int, spec: str, profiles: tuple, configs: list) -> None:
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        results = pool.map(functools.partial(config_rows, spec=spec, profiles=profiles), configs, chunksize=1)
    with open(out, "w") as fh:
        for rows in results:
            for row in rows:
                fh.write(json.dumps(row) + "\n")


def cmd_records(out: str, spec: str, profiles: tuple) -> None:
    from mimarsinan.deployment_record.objectives.views import DeploymentRecordView
    from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
    from mimarsinan.deployment_record.schema import load_deployment_record
    h = _harness()
    overrides = host_overrides(spec)
    pins = h._pins()["records"]
    rows = []
    for path in h.FROZEN_RECORDS:
        assert h._sha(path) == pins[path], f"frozen record changed: {path}"
        record = load_deployment_record(path)
        for profile, variant in itertools.product(profiles, ("chip", "host")):
            physics = resolve_platform_physics(profile, overrides if variant == "host" else {})
            row = {"kind": "record", "record": path, "profile": profile, "variant": variant}
            try:
                view = DeploymentRecordView(record=h._with_physics(record, physics.to_dict()))
                row.update(_report_row(view.cost_report()))
                row["host_ops_s"] = _quantity(view, "host_ops_s")
                row["host_macs"] = _quantity(view, "host_macs")
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = "refused"
                row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"
            rows.append(row)
    with open(out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def cmd_block(out: str, spec: str, profiles: tuple) -> None:
    ""
    census = block_census(spec, profiles)
    with open(out, "w") as fh:
        json.dump(census, fh, indent=2)
        fh.write("\n")
    for row in census["rows"]:
        print(f"row {row['key']}: {row['band']} {row['unit']}, {row['evidence_kind']}, host {row['host']!r}")
    for row in census["lacking"]:
        print(f"lacking {row['key']}: " + (row["refused"] or "kept from the profile (the block names no host)"))
    for name, ok in census["checks"].items():
        print(f"{'pass' if ok else 'FAIL' if ok is False else 'n/a'}: {name}")


def _bands(row):
    from mimarsinan.deployment_record.cost.absolute import HostEnergySplit
    from mimarsinan.deployment_record.schema.provenance import Band
    if not row.get("energy"):
        return None
    host = row.get("host")
    return HostEnergySplit(with_host=Band(*row["energy"], "w"), without_host=Band(*row["without_host"], "c"),
                           host=None if host is None else Band(*host, "h"))


CAPS_PER_TARGET = 8


def cap_grid(ends, n: int = CAPS_PER_TARGET) -> list:
    ""
    least, greatest = min(ends), max(ends)
    lo, hi = math.log10(least), math.log10(greatest)
    return [least] + [10 ** (lo + (hi - lo) * i / (n - 1)) for i in range(1, n - 1)] + [greatest]


def ordered_ends(a, b) -> types.SimpleNamespace:
    ""
    return types.SimpleNamespace(low=min(a, b), high=max(a, b))


def exact_cap_ends(split) -> tuple:
    ""
    w, h = split.with_host, split.host
    with_ends = ordered_ends(Fraction(w.low), Fraction(w.high))
    if h is None:
        return with_ends, with_ends
    without = ordered_ends(Fraction(w.low) - Fraction(h.low), Fraction(w.high) - Fraction(h.high))
    return with_ends, without


def cmd_verdicts(out: str, points_path: str, records_path: str, profiles: tuple) -> None:
    from mimarsinan.deployment_record.cost.absolute import dual_verdict
    from mimarsinan.deployment_record.cost.absolute.host_split import band_order, cap_verdict, corner_order
    points = [json.loads(line) for line in open(points_path)]
    records = [json.loads(line) for line in open(records_path)]
    summary: dict = {"candidate": {}, "record": {}}
    for label, rows, key in (("candidate", points, "config"), ("record", records, "record")):
        chip = {(r[key], r["profile"]): r for r in rows if r["variant"] == "chip"}
        host = {(r[key], r["profile"]): r for r in rows if r["variant"] == "host"}
        s = summary[label]
        s["points"] = len(host)
        s["errors"] = sorted({r.get("error") for r in rows if r["status"] == "refused"} - {None})
        s["errored_points"] = sum(r["status"] == "refused" for r in host.values())
        s["energy_priced_chip"] = sum(bool(r.get("energy")) for r in chip.values())
        s["energy_priced_host"] = sum(bool(r.get("energy")) for r in host.values())
        s["host_refusals_chip"] = sum(bool(r.get("energy_host_refusal")) for r in chip.values())
        s["host_refusals_host"] = sum(bool(r.get("energy_host_refusal")) for r in host.values())
        s["host_refusals_host_list"] = [[k[0], k[1], host[k]["energy_refusal"]] for k in sorted(host)
                                        if host[k].get("energy_host_refusal")]
        s["priced_only_with_host_block"] = sum(bool(host[k].get("energy")) and not chip.get(k, {}).get("energy")
                                               for k in host)
        s["other_refusals_host"] = sorted({r["energy_refusal"][:160] for r in host.values()
                                           if r.get("energy_refusal") and not r.get("energy_host_refusal")})
        shares = {}
        for k, r in host.items():
            if r.get("host_share") is not None:
                shares.setdefault(k[1], []).append(r["host_share"])
        s["host_share_by_profile"] = {
            p: {"n": len(v), "with_host_work": sum(x[1] > 0 for x in v),
                "min": [min(x[i] for x in v) for i in range(3)],
                "median": [sorted(x[i] for x in v)[len(v) // 2] for i in range(3)],
                "max": [max(x[i] for x in v) for i in range(3)]} for p, v in sorted(shares.items())}
        splits = {k: _bands(r) for k, r in host.items() if _bands(r) is not None}
        s["splits"] = len(splits)

        flips_corner, flips_interval, n_pairs = [], [], 0
        programs = sorted({k[0] for k in splits})
        for prog in programs:
            for pa, pb in itertools.combinations(profiles, 2):
                if (prog, pa) not in splits or (prog, pb) not in splits:
                    continue
                n_pairs += 1
                pair = {"a": splits[(prog, pa)], "b": splits[(prog, pb)]}
                for rule, sink in ((corner_order, flips_corner), (band_order, flips_interval)):
                    v = dual_verdict(lambda e, rule=rule: rule(e["a"], e["b"]), pair)
                    if v.host_bound:
                        sink.append(v.line(f"{prog}: {pa} vs {pb} ({rule.__name__})"))
        s["cross_target"] = {"pairs": n_pairs, "corner_order_host_bound": len(flips_corner),
                             "band_order_host_bound": len(flips_interval),
                             "corner_order_lines": flips_corner, "band_order_lines_first_20": flips_interval[:20]}

        flips_b, n_b = [], 0
        if label == "candidate":
            network = {r["config"]: tuple(r["network"]) for r in points}
            for profile in profiles:
                progs = sorted(p for p in programs if (p, profile) in splits)
                for x, y in itertools.combinations(progs, 2):
                    if network[x] != network[y]:
                        continue
                    n_b += 1
                    pair = {"x": splits[(x, profile)], "y": splits[(y, profile)]}
                    v = dual_verdict(lambda e: corner_order(e["x"], e["y"]), pair)
                    if v.host_bound:
                        flips_b.append(v.line(f"{profile}: {x} vs {y} (corner_order)"))
        s["same_network_same_target"] = {"pairs": n_b, "host_bound": len(flips_b), "lines": flips_b}


        flips_c, n_c, transitions = [], 0, {}
        for profile in profiles:
            keys = [k for k in splits if k[1] == profile]
            exact = {k: exact_cap_ends(splits[k]) for k in keys}
            ends = [e for k in keys for b in exact[k] for e in (b.low, b.high) if e > 0]
            if not ends:
                continue
            caps = cap_grid(ends)
            for k in keys:
                cell = {"p": types.SimpleNamespace(with_host=exact[k][0], without_host=exact[k][1])}
                for cap in caps:
                    n_c += 1
                    v = dual_verdict(lambda e, cap=cap: cap_verdict(e["p"], cap), cell)
                    if v.host_bound:
                        name = f"{v.without_host} -> {v.with_host}"
                        transitions[name] = transitions.get(name, 0) + 1
                        flips_c.append(v.line(f"{k[0]} @ {profile}, cap {float(cap):.3e} mJ"))
        s["energy_cap"] = {"cells": n_c, "host_bound": len(flips_c), "transitions": transitions,
                           "lines_first_40": flips_c[:40]}
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    for label, s in summary.items():
        print(f"[{label}] points {s['points']} (errored {s['errored_points']}), energy priced {s['energy_priced_chip']}"
              f" -> {s['energy_priced_host']} with the host block; host refusals {s['host_refusals_chip']} -> "
              f"{s['host_refusals_host']}; cross-target pairs {s['cross_target']['pairs']}: corner-order host-bound "
              f"{s['cross_target']['corner_order_host_bound']}, interval host-bound "
              f"{s['cross_target']['band_order_host_bound']}; same-network pairs "
              f"{s['same_network_same_target']['pairs']}: host-bound {s['same_network_same_target']['host_bound']}; "
              f"cap cells {s['energy_cap']['cells']}: host-bound {s['energy_cap']['host_bound']}")


def main() -> int:
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "block":
        cmd_block(args[0], args[1], reference_profiles(args[2]))
    elif cmd == "points":
        cmd_points(args[0], int(args[1]), args[2], reference_profiles(args[3]), args[4:])
    elif cmd == "records":
        cmd_records(args[0], args[1], reference_profiles(args[2]))
    elif cmd == "verdicts":
        cmd_verdicts(args[0], args[1], args[2], reference_profiles(args[3]))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
