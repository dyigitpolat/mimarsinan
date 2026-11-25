""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import re
import sys
import time
from pathlib import Path

import w38common as c
import configs
import lockcheck
import no_update

RESULT_FORMAT = "w38-row-result/1"


def _session(row_dir: Path) -> dict:
    rec = json.loads((row_dir / "session.json").read_text())
    if rec.get("status") != "ran":
        raise SystemExit(f"REFUSED: {row_dir.name} session is {rec.get('status')}")
    return rec


def _same_code(session: dict, start: dict, what: str) -> None:
    if start != session["driver_code_sha256"]:
        changed = sorted(k for k in set(start) | set(session["driver_code_sha256"])
                         if start.get(k) != session["driver_code_sha256"].get(k))
        raise SystemExit(f"REFUSED: the {what} runs other driver code than the row's session: {changed[:8]}")


def merge_code(session: dict, start: dict, arms: dict | None = None) -> dict | None:
    ""
    theirs = session["driver_code_sha256"]
    if start == theirs:
        return None
    changed = sorted(k for k in set(start) | set(theirs) if start.get(k) != theirs.get(k))
    listing = c.listing_sha256(theirs)
    entries = ((arms or configs.arms()).get("revision_rules") or {}).get("merge_resume") or []
    entry = next((e for e in entries if isinstance(e, dict) and e.get("listing_sha256") == listing), None)
    if entry is None:
        raise SystemExit(f"REFUSED: MERGE_RESUME: the merge runs other driver code than the row's session "
                         f"({changed[:8]}), and the lock in force declares no resume of the session's code "
                         f"(revision_rules.merge_resume names no listing {listing[:16]}); nothing was written")
    beyond = sorted(set(changed) - set(entry.get("files") or []))
    if beyond:
        raise SystemExit(f"REFUSED: MERGE_RESUME: the merge's driver code differs from the row's session's in "
                         f"{beyond}, beyond the files the lock's resume of revision {entry.get('revision')} names "
                         f"({entry.get('files')}); nothing was written")
    return {"session_listing_sha256": listing, "merge_listing_sha256": c.listing_sha256(start),
            "files_differing": changed, "declared": entry}


def inputs(entries: list[dict], tf):
    ""
    import torch

    return torch.stack([tf(c.load_image(e)) for e in entries]).contiguous()


def input_path_check(row_dir: Path, session: dict, split: str, tf) -> dict:
    ""
    import torch

    import members_builder
    import provider
    import mimarsinan.mapping.platform.target_expansion as target_expansion
    from mimarsinan.config_schema.runtime import run_sections, top_declaration
    from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
    from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

    document = json.loads((Path(session["working_directory"]) / "_RUN_CONFIG" / "config.json").read_text())
    if c.sha256_file(row_dir / "config.json") != session["config_sha256"]:
        raise SystemExit("REFUSED: config.json differs from the session's digest")
    expanded = target_expansion.expand_target_document(document, write_derived=False)
    _, _, layout = run_sections(expanded)
    factory = BasicDataProviderFactory(expanded["data_provider_name"],
                                       top_declaration(expanded, "datasets_path", "./datasets"),
                                       seed=layout["seed"], preprocessing=layout.get("preprocessing"),
                                       batch_size=layout.get("batch_size"))
    data_provider = factory.create()
    loader = DataLoaderFactory(factory, num_workers=0).create_test_loader(data_provider.get_test_batch_size(),
                                                                          data_provider)
    x0, y0 = next(iter(loader))
    entries = c.split_entries(split, len(x0))
    mine = inputs(entries, tf)
    labels = [int(e["class_index"]) for e in entries[: len(x0)]]
    return {"images": int(len(x0)), "tensors_bit_equal": bool(x0.dtype == mine.dtype and torch.equal(x0, mine)),
            "labels_equal": [int(v) for v in y0.tolist()] == labels,
            "max_abs_difference": float((x0.double() - mine.double()).abs().max()) if x0.shape == mine.shape else None,
            "passed": bool(x0.dtype == mine.dtype and torch.equal(x0, mine)
                           and [int(v) for v in y0.tolist()] == labels)}


def k9_arithmetic():
    ""
    try:
        from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC
    except ImportError as exc:
        raise SystemExit(f"REFUSED: RECORD_ABSENT: the head carries no record of the deployed arithmetic "
                         f"(mapping/ir/deployment_dtype.py, U-DAR): {exc}")
    return VALUE_ARITHMETIC


def load_mapping(session: dict):
    with open(Path(session["working_directory"]) / "hybrid_mapping.pickle", "rb") as f:
        return pickle.load(f)


def parts_dir(session: dict) -> Path:
    return Path(session["working_directory"]).parent / "w38_parts"


def shard(row_dir: Path, lo: int, hi: int, glo: int, ghi: int) -> int:
    import numpy as np
    import torch

    torch.set_num_threads(1)
    lockcheck.verify()
    c.install_data_guard()
    start = c.driver_code_sha256()
    session = _session(row_dir)
    _same_code(session, start, "shard")
    c.toolchain_provenance(session["toolchain"]["path"])
    from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
    from mimarsinan.chip_simulation.value_run.int64_golden import run_golden

    member = session["row"].split(".")[0]
    split = session["view"]["split"]
    batch = int(configs.arms()["harvest"]["executor_batch"])
    entries = c.split_entries(split, max(hi, ghi))
    tf = c.composed_transform(c.members()[member])
    rec = {"lo": lo, "hi": hi, "glo": glo, "ghi": ghi, "split": split, "times": {}}
    if lo == 0:
        rec["input_path_check"] = input_path_check(row_dir, session, split, tf)
        if not rec["input_path_check"]["passed"]:
            raise SystemExit(f"REFUSED: the harvest's inputs differ from the session's test loader: "
                             f"{rec['input_path_check']}")
    mapping = load_mapping(session)
    arithmetic = k9_arithmetic()
    flow = ValueHybridCoreFlow(mapping, device=arithmetic.device, dtype=arithmetic.core_dtype)
    rec["deployed_arithmetic"] = arithmetic.line()
    parts = parts_dir(session)
    parts.mkdir(parents=True, exist_ok=True)
    t = time.time()
    outs = []
    with torch.no_grad():
        for i in range(lo, hi, batch):
            outs.append(flow(inputs(entries[i: min(i + batch, hi)], tf).to(arithmetic.input_dtype)).numpy())
    out = np.concatenate(outs) if outs else np.zeros((0, 1000))
    rec["times"]["packed_fp64_s"] = time.time() - t
    rec["files"] = {f"packed_fp64_{lo:05d}_{hi:05d}.npy": c.save_npy(parts / f"packed_fp64_{lo:05d}_{hi:05d}.npy", out)}
    if ghi > glo:
        x = inputs(entries[glo:ghi], tf)
        t = time.time()
        codes, steps = run_golden(mapping, x.double().numpy())
        rec["times"]["golden_s"] = time.time() - t
        rec["files"][f"golden_codes_{glo:05d}_{ghi:05d}.npy"] = c.save_npy(
            parts / f"golden_codes_{glo:05d}_{ghi:05d}.npy", np.asarray(codes))
        rec["files"][f"golden_steps_{glo:05d}_{ghi:05d}.npy"] = c.save_npy(
            parts / f"golden_steps_{glo:05d}_{ghi:05d}.npy", np.asarray(steps))
    rec["peak_rss_kb"] = c.peak_rss_kb()
    c.refuse_if_changed(start, f"shard {lo}:{hi}")
    rec["driver_code_listing_sha256"] = c.listing_sha256(start)
    c.atomic_write_json(row_dir / "shards" / f"shard_{lo:05d}_{hi:05d}.json", rec)
    print(json.dumps({k: rec[k] for k in ("lo", "hi", "times", "peak_rss_kb")}), flush=True)
    return 0


def _join(parts: Path, prefix: str, n: int):
    import numpy as np

    files = sorted(parts.glob(f"{prefix}_*.npy"))
    spans = [tuple(int(v) for v in f.stem.split("_")[-2:]) for f in files]
    if not files or spans[0][0] != 0 or spans[-1][1] != n or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        return None
    return np.concatenate([np.load(f) for f in files])


def mapping_census(m) -> dict:
    ""
    neural = [s for s in m.stages if s.kind == "neural"]
    passes: dict[int, int] = {}
    for s in neural:
        if getattr(s, "schedule_pass_index", None) is not None:
            si = int(s.schedule_segment_index or 0)
            passes[si] = max(passes.get(si, 0), int(s.schedule_pass_index) + 1)
    physical = [sum(len(getattr(hc, "fused_component_axons", None) or [0]) for hc in s.hard_core_mapping.cores)
                for s in neural]
    return {"physical_cores": int(sum(physical)), "physical_cores_max_stage": int(max(physical, default=0)),
            "listed_cores": int(sum(len(s.hard_core_mapping.cores) for s in neural)), "neural_stages": len(neural),
            "compute_stages": sum(s.kind == "compute" for s in m.stages), "segments_scheduled": len(passes),
            "passes_per_segment_max": max(passes.values()) if passes else 1,
            "passes_summed_over_segments": int(sum(passes.values())) if passes else len(neural)}


def _record_fields(work: Path) -> dict:
    path = work / "deployment_record.json"
    if not path.exists():
        return {"present": False}
    d = json.loads(path.read_text())
    sched = d.get("schedule", {})
    return {"present": True, "sha256": c.sha256_file(path), "identity": d.get("identity"),
            "schedule": {k: v for k, v in sched.items() if isinstance(v, (int, float, str, bool, type(None)))},
            "crossbar": (d.get("utilization") or {}).get("crossbar"), "accuracy": d.get("accuracy"),
            "residency": {k: (d.get(k) if not isinstance(d.get(k), list) else {"length": len(d.get(k))})
                          for k in ("residency_plan", "resident_core_indices") if k in d},
            "declared_departures": d.get("declared_departures")}


def _pipeline_certificates(lines: list[str]) -> dict:
    out = {}
    for ln in lines:
        m = re.search(r"\[ValueParityGate\] (PASS|FAIL).*max\|delta\|=([0-9.e+-]+), atol=([0-9.e+-]+), n=(\d+)", ln)
        if m:
            out["K4_value_parity_gate"] = {"verdict": m[1], "max_abs_delta": float(m[2]), "atol": float(m[3]),
                                           "samples": int(m[4])}
        m = re.search(r"\[ValueTwinCertificate\] (PASS|FAIL).*samples=(\d+) windows=(\d+).*max\|delta\|=([0-9.e+-]+) "
                      r"atol=([0-9.e+-]+)", ln)
        if m:
            out["K5_value_twin_certificate"] = {"verdict": m[1], "samples": int(m[2]), "windows": int(m[3]),
                                                "max_abs_delta": float(m[4]), "atol": float(m[5])}
        m = re.search(r"Hard-core Spiking Simulation Test: ([0-9.]+)", ln)
        if m:
            out["census_metric"] = float(m[1])
    return out


def _resolved(work: Path) -> dict:
    p = work / "Model Configuration.platform_constraints_resolved.json"
    return json.loads(p.read_text()) if p.exists() else {}


def declared_width(config: dict, bundle_values: dict) -> dict:
    ""
    d = configs.declared_platform(config, "activation_bits", bundle_values)
    return {"value": None if d["value"] is None else int(d["value"]), "source": d["source"]}


def declared_cores(config: dict, bundle_cores: list) -> list:
    ""
    return configs.declared_platform(config, "cores", {"cores": bundle_cores})["value"] or bundle_cores


R4_CORE_KEYS = ("max_axons", "max_neurons", "count", "has_bias")


def core_fields_check(record_cores: list, want_cores: list) -> dict:
    ""
    differ, absent, undeclared = [], [], []
    for i in range(max(len(want_cores), len(record_cores))):
        want = want_cores[i] if i < len(want_cores) else {}
        have = record_cores[i] if i < len(record_cores) else {}
        for field in sorted(want):
            if field not in have:
                absent.append({"core": i, "field": field, "declared": want[field]})
            elif have[field] != want[field]:
                differ.append({"core": i, "field": field, "declared": want[field], "record": have[field]})
        for field in R4_CORE_KEYS:
            if field in have and field not in want:
                undeclared.append({"core": i, "field": field, "record": have[field]})
    same_length = len(record_cores) == len(want_cores)
    return {"passed": same_length and not differ and not absent and not undeclared,
            "same_number_of_core_types": same_length, "differ": differ, "absent": absent,
            "record_fields_not_declared": undeclared}


def _k11(identity: dict | None, config: dict, scheduled: bool, census: dict, declared: int,
         bundle_values: dict, width: dict) -> dict:
    ""
    if not identity:
        return {"passed": False, "reason": "no record identity"}
    plat = identity.get("platform", {})

    def want(key: str):
        return configs.declared_platform(config, key, bundle_values)

    cores = core_fields_check(list(plat.get("cores", [])),
                              list(declared_cores(config, bundle_values.get("cores") or [])))
    resolved, wanted = width["resolved"], width["declared"]["value"]
    checks = {
        "cores": cores["passed"],
        "allow_scheduling": plat.get("allow_scheduling") == scheduled,
        "max_schedule_passes": plat.get("max_schedule_passes") == want("max_schedule_passes")["value"],
        "readout_law": plat.get("readout_law") == want("readout_law")["value"],
        "activation_bits": resolved is not None and plat.get("activation_bits") == resolved,
        "activation_bits_declared": resolved is not None and resolved == wanted,
        "core_semantics": plat.get("core_semantics") == "mvm",
        "residency_plan_absent": plat.get("residency_plan") in (None, {}),
        "workload": identity.get("workload") in (None, config["data_provider_name"]),
    }
    unrecorded = []
    for key in ("position_multiplexing", "weight_scale_rule"):
        d = want(key)
        if key in plat:
            checks[key] = plat.get(key) == d["value"]
        elif d["source"] in ("target_overrides", "platform_constraints"):
            unrecorded.append(key)
    if scheduled:
        non_vacuous = census["passes_per_segment_max"] > 1
    else:
        non_vacuous = census["passes_per_segment_max"] == 1 and census["physical_cores"] <= declared
    return {"checks": checks, "declared_but_not_in_record_identity": unrecorded,
            "declared_core_fields_differing": cores["differ"],
            "declared_core_fields_not_in_record_identity": cores["absent"],
            "record_core_fields_not_declared": cores["record_fields_not_declared"],
            "activation_bits": {"record": plat.get("activation_bits"), "resolved": resolved,
                                "declared": width["declared"]},
            "non_vacuous": bool(non_vacuous), "passed": all(checks.values()) and bool(non_vacuous),
            "rule": "chip and chipxK: some segment runs in more than one pass; res: one pass within the pool; "
                    "every declared field of every declared core printed by the record identity and equal, and r4's "
                    "four core keys (max_axons, max_neurons, count, has_bias) compared both ways; the record's input "
                    "width equal to the resolved one, and the resolved one to the declaration"}


def clopper_pearson(k: int, n: int, level: float = 0.95) -> list[float]:
    from scipy.stats import beta

    a = 1.0 - level
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def tie_counts(out, y) -> dict:
    ""
    import numpy as np

    top = out.max(1, keepdims=True)
    tied = (out == top).sum(1) > 1
    pred = out.argmax(1)
    lowest = pred == y
    return {"correct": int((lowest & ~tied).sum()), "correct_ties_lowest_index": int(lowest.sum()),
            "argmax_ties": int(tied.sum()), "correct_mask": lowest & ~tied}


def paired_comparison(deployed_right, float_right) -> dict:
    ""
    import numpy as np
    from scipy.stats import binom

    deployed_right, float_right = np.asarray(deployed_right, bool), np.asarray(float_right, bool)
    n = int(len(deployed_right))
    b = int((float_right & ~deployed_right).sum())
    cc = int((~float_right & deployed_right).sum())
    m = b + cc
    p = 1.0 if m == 0 else float(min(1.0, 2.0 * binom.cdf(min(b, cc), m, 0.5)))
    d = (cc - b) / n
    if m == 0:
        interval = [0.0, 0.0]
    else:
        lo, hi = clopper_pearson(cc, m)
        interval = [m / n * (2 * lo - 1), m / n * (2 * hi - 1)]
    words = configs.arms()["reporting"]["wording"]
    within = interval[0] >= -0.01 and interval[1] <= 0.01
    direction = "below" if interval[1] < 0 else ("above" if interval[0] > 0 else "otherwise")
    return {"images": n, "both_right": int((float_right & deployed_right).sum()), "float_right_deployed_wrong_b": b,
            "float_wrong_deployed_right_c": cc, "both_wrong": int((~float_right & ~deployed_right).sum()),
            "mcnemar_exact_p": p, "difference_d": d, "difference_ci95_conditional": interval,
            "within_one_point": bool(within), "direction": direction,
            "wording": [words["within_one_point"]] * bool(within) + [words[direction]] * (direction != "otherwise")
            or [words["otherwise"]],
            "rule": "ties counted as wrong on both sides; interval conditional on b + c"}


def host_share(work: Path, session: dict, mapping, placement: str = "subsume") -> dict:
    ""
    import torch

    classes = collections.Counter()
    for s in mapping.stages:
        if s.kind == "compute" and s.compute_op is not None:
            module = (s.compute_op.params or {}).get("module")
            classes[f"{s.compute_op.op_type} ({type(module).__name__ if module is not None else 'no module'})"] += 1
    out = {"compute_stages": int(sum(classes.values())), "host_operation_classes": dict(sorted(classes.items()))}
    steps = [st for st in session.get("steps_run", []) if (work / f"{st}.model.pt").exists()]
    if not steps:
        out["host_mac_share"] = None
        out["reason"] = "no cached step model"
        return out
    from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fractions

    member = c.members()[session["row"].split(".")[0]]
    crop = int(member["transforms"]["crop_size"][0])
    obj = torch.load(work / f"{steps[-1]}.model.pt", map_location="cpu", weights_only=False)
    flow = (obj[0] if isinstance(obj, tuple) else obj).eval()
    macs, ops = estimate_onchip_fractions(flow, (3, crop, crop), 1000, metrics=("macs", "ops"),
                                          encoding_placement=placement)
    out.update({"model": steps[-1], "host_macs": int(macs.host), "onchip_macs": int(macs.onchip),
                "total_macs": int(macs.total), "host_mac_share": (macs.host / macs.total) if macs.total else None,
                "host_ops": int(ops.host), "total_ops": int(ops.total),
                "source": "mapping.verification.onchip_fraction.estimate_onchip_fractions"})
    return out


def merge(row_dir: Path) -> int:
    import numpy as np

    lockcheck.verify()
    start = c.driver_code_sha256()
    session = _session(row_dir)
    resume = merge_code(session, start)
    if session.get("mode") != "smoke" and (row_dir / "result.json").exists():
        raise SystemExit(f"REFUSED: {row_dir / 'result.json'} exists: a row is merged once")
    import members_builder
    import provider
    member, bundle, arm, budget = session["row"].split(".")
    a = configs.arms()
    config = json.loads((row_dir / "config.json").read_text())
    if c.sha256_file(row_dir / "config.json") != session["config_sha256"]:
        raise SystemExit("REFUSED: config.json differs from the session's digest")
    work = Path(session["working_directory"])
    persisted = json.loads((work / "_RUN_CONFIG" / "config.json").read_text())
    n = int(session["images"])
    parts = parts_dir(session)
    packed = _join(parts, "packed_fp64", n)
    if packed is None:
        raise SystemExit(f"REFUSED: packed shards do not cover 0..{n}")
    shard_recs = [json.loads(p.read_text()) for p in sorted((row_dir / "shards").glob("shard_*.json"))]
    session_listing = c.listing_sha256(session["driver_code_sha256"])
    if any(s.get("driver_code_listing_sha256") != session_listing for s in shard_recs):
        raise SystemExit("REFUSED: a shard ran other driver code than the row's session")
    for s in shard_recs:
        for name, digest in s["files"].items():
            if c.sha256_file(parts / name) != digest:
                raise SystemExit(f"REFUSED: {parts / name} differs from its shard record")
    split = session["view"]["split"]
    y = np.asarray([e["class_index"] for e in c.split_entries(split, n)], dtype=np.int64)
    pred = packed.argmax(1)
    top5 = np.argsort(-packed, axis=1, kind="stable")[:, :5]
    ties = tie_counts(packed, y)
    k = ties["correct"]
    inv = min(n, int(a["harvest"]["invariance_images"]))
    fdir = c.float_reference_dir(member, split)
    fpred = np.load(fdir / "pred.npy")[:n]
    flog = np.load(fdir / "logits.npy")[:n]
    float_ties = tie_counts(flog, y)
    flat = configs.flat_run_config(persisted)
    mapping = load_mapping(session)
    census = mapping_census(mapping)
    hshare = host_share(work, session, mapping, str(flat.get("encoding_layer_placement", "subsume")))
    del mapping
    record = _record_fields(work)
    certs = _pipeline_certificates(session.get("certificate_lines", []))
    resolved = _resolved(work)
    width = {"resolved": flat.get("activation_bits"),
             "declared": declared_width(config, session["bundle"]["values"])}
    width["declared"]["equals_resolved"] = width["resolved"] is not None and \
        width["resolved"] == width["declared"]["value"]
    res = {
        "format": RESULT_FORMAT, "row": session["row"], "member": member, "bundle": bundle, "arm": arm,
        "budget": budget, "smoke": session["smoke"], "mode": session.get("mode"), "split": split,
        "status": "complete", "law_declarations": session.get("law_declarations"),
        "departures": session.get("departures"),
        "inputs": {"toolchain": session["toolchain"], "lock": session["lock"], "driver_code_sha256": start,
                   "config_sha256": session["config_sha256"], "persisted_config_equals_declared": persisted == config,
                   "view_sha256": session["view_sha256"], "weights": session["K0"], "provider": session.get("provider"),
                   "bundle": session["bundle"], "faithfulness_labels": {k: resolved.get(k) for k in (
                       "target_claim_class", "target_faithfulness_class")}},
        "K0": session["K0"], "K1": session["K1"],
        "deployment": {**census, "declared_count": session["declared_count"], "scheduled": budget != "res",
                       "weight_bits": session["arm_weight_bits"],
                       "activation_bits": width["resolved"],
                       "activation_bits_declared": width["declared"],
                       "activation_bits_rule": "resolved: the run's flat config of the persisted document "
                                               "(configs.flat_run_config); declared: target_overrides, else the "
                                               "plain key, else the bundle",
                       "declared_departures": record.get("declared_departures"),
                       "cores_declared": declared_cores(config, session["bundle"]["values"]["cores"]),
                       "pipeline_wall_s": session["wall_s"], "pipeline_peak_rss_kb": session["peak_rss_kb"],
                       "step_walls_s": session["step_walls_s"], "steps_run": session["steps_run"]},
        "record": record,
        "K4_K5_pipeline": {kk: v for kk, v in certs.items() if kk.startswith("K")},
        "K9": {"images": n, "split": split, "correct": k, "top1": k / n, "clopper_pearson_95": clopper_pearson(k, n),
               "correct_ties_lowest_index": ties["correct_ties_lowest_index"], "argmax_ties": ties["argmax_ties"],
               "tie_rule": configs.arms()["reporting"]["ties"],
               "deployed_arithmetic": next((s_.get("deployed_arithmetic") for s_ in shard_recs), None),
               "top5_correct": int((top5 == y[:, None]).any(1).sum()),
               "outputs_sha256": c.sha256_array(packed),
               "files": {"pred.npy": c.save_npy(row_dir / "pred.npy", pred),
                         "top5.npy": c.save_npy(row_dir / "top5.npy", top5.astype(np.int16)),
                         "packed_prefix_fp64.npy": c.save_npy(row_dir / "packed_prefix_fp64.npy", packed[:inv])},
               "prefix_images": inv, "prefix_outputs_sha256": c.sha256_array(packed[:inv])},
        "float_agreement": {"images": n, "agree": int((pred == fpred).sum()), "float_correct": float_ties["correct"],
                            "float_argmax_ties": float_ties["argmax_ties"],
                            "reference": str((fdir / "pred.npy").relative_to(c.OUT))},
        "paired": paired_comparison(ties["correct_mask"], float_ties["correct_mask"]),
        "host": hshare,
    }
    g = int(a["harvest"]["golden_images"])
    gcodes = _join(parts, "golden_codes", min(g, n))
    if gcodes is not None:
        gsteps = [np.load(f) for f in sorted(parts.glob("golden_steps_*.npy"))]
        if any(not np.array_equal(s, gsteps[0]) for s in gsteps):
            raise SystemExit("REFUSED: the golden output steps differ between shards")
        gv = np.asarray(gcodes * gsteps[0], dtype=np.float64).reshape(packed[: len(gcodes)].shape)
        d = np.abs(gv - packed[: len(gcodes)])
        res["K6"] = {"images": int(len(gcodes)), "outputs": int(d.size), "mismatched_outputs": int((d > 0).sum()),
                     "max_abs_delta": float(d.max()), "passed": bool(d.max() == 0.0),
                     "nonzero_output_fraction": float((gcodes != 0).mean())}
    else:
        res["K6"] = {"passed": None, "reason": "golden shards absent"}
    if "census_metric" in certs:
        cn = int(session["view"]["test_images"])
        srt = np.sort(packed[:cn], axis=1)
        scale = np.maximum(np.abs(packed[:cn]).max(1), 1e-300)
        near = int(((srt[:, -1] - srt[:, -2]) <= float(a["harvest"]["k14_near_tie_rel"]) * scale).sum())
        res["K14"] = {"census_images": cn, "pipeline_census_count": int(round(certs["census_metric"] * cn)),
                      "fp64_count_same_images": int((pred[:cn] == y[:cn]).sum()), "near_tie_images": near,
                      "near_tie_rel": float(a["harvest"]["k14_near_tie_rel"])}
        res["K14"]["difference"] = res["K14"]["pipeline_census_count"] - res["K14"]["fp64_count_same_images"]
        res["K14"]["passed"] = abs(res["K14"]["difference"]) <= near
        res["K14"]["criterion"] = "|difference| at most the near-tie images of the census prefix"
    else:
        res["K14"] = {"passed": None, "reason": "no census line in the session log"}
    res["K11"] = _k11((record.get("identity") or None), config, budget != "res", census,
                      int(session["declared_count"]), session["bundle"]["values"], width)
    if resume is not None:
        res["inputs"]["merge_resume"] = resume
    nu = no_update.check_work(work, Path(session["K0"]["weight_source"]), session["K0"]["weight_source_sha256"],
                              flat=flat)
    if nu.get("applicable") and int(nu["bits"]) != int(session["arm_weight_bits"]):
        nu["passed"] = False
        nu["bits_mismatch"] = [nu["bits"], session["arm_weight_bits"]]
    c.atomic_write_json(row_dir / "no_update.json", nu)
    res["no_update"] = {"passed": nu.get("passed"), "applicable": nu.get("applicable"),
                        "file_sha256": c.sha256_file(row_dir / "no_update.json")}
    res["input_path_check"] = next((s_["input_path_check"] for s_ in shard_recs if "input_path_check" in s_), None)
    res["harvest"] = {"shards": len(shard_recs),
                      "packed_fp64_s_per_image": sum(s["times"]["packed_fp64_s"] for s in shard_recs) / n,
                      "golden_s_per_image": (sum(s["times"].get("golden_s", 0.0) for s in shard_recs) / max(1, min(g, n))),
                      "shard_peak_rss_kb": max(s["peak_rss_kb"] for s in shard_recs)}
    c.refuse_if_changed(start, "the merge")
    c.atomic_write_json(row_dir / "result.json", res)
    print(json.dumps({kk: res.get(kk) for kk in ("row", "K9", "K6", "K11", "K14", "K4_K5_pipeline", "no_update",
                                                 "float_agreement")}, indent=1, default=str)[:5000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--row-dir", required=True)
    s.add_argument("--lo", type=int, required=True)
    s.add_argument("--hi", type=int, required=True)
    s.add_argument("--glo", type=int, default=0)
    s.add_argument("--ghi", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--row-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "shard":
        return shard(Path(a.row_dir), a.lo, a.hi, a.glo, a.ghi)
    return merge(Path(a.row_dir))


if __name__ == "__main__":
    sys.exit(main())
