""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common
import configs
import declarations
import e01_provider
import e2e_members
import footprints
import inventory
import no_update

ROWS = common.ROWS
RESULT_FORMAT = "e2e-row-result/1"
INTEGER_TARGETS = ("T-INT", "T-VAL8-OP", "T-VAL8-INT")
GOLDEN_BATCH = 16
WI1, WI2 = common.RECORDS / "wi1", common.RECORDS / "wi2"


def _row(key: str) -> configs.Row:
    member, target, route, budget, geometry = key.split(".")
    return configs.make_row(member, target, route, budget, geometry)


def _session(key: str) -> dict:
    rec = json.loads((ROWS / key / "session.json").read_text())
    if rec.get("status") != "ran":
        raise SystemExit(f"REFUSED: {key} session is {rec.get('status')}")
    return rec


def e01_inputs(member, provider_name: str = e01_provider.PROVIDER_NAME) -> tuple[torch.Tensor, np.ndarray]:
    provider = e01_provider.PROVIDERS[provider_name](str(common.DATASETS_PATH),
                                                     preprocessing=inventory.preprocessing(member))
    ds = provider._get_test_dataset()
    xs, ys = zip(*(ds[i] for i in range(len(ds))))
    return torch.stack(xs).contiguous(), np.asarray(ys, dtype=np.int64)


def _storage_on_cpu(data: bytes):
    ""
    import io

    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


class _ProgramUnpickler(pickle.Unpickler):
    ""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return _storage_on_cpu
        return super().find_class(module, name)


def load_mapping(session: dict):
    with open(Path(session["working_directory"]) / "hybrid_mapping.pickle", "rb") as f:
        return _ProgramUnpickler(f).load()


def _physical(hc) -> int:
    return len(getattr(hc, "fused_component_axons", None) or [0])


def cores_by_width(m) -> dict:
    ""
    from mimarsinan.mapping.packing.type_binding import bound_bits

    out: dict[str, int] = {}
    for s in m.stages:
        if s.kind != "neural":
            continue
        for hc in s.hard_core_mapping.cores:
            b = bound_bits(hc)
            if b is not None:
                out[str(b)] = out.get(str(b), 0) + _physical(hc)
    return dict(sorted(out.items()))


def mapping_census(m) -> dict:
    ""
    neural = [s for s in m.stages if s.kind == "neural"]
    passes: dict[int, int] = {}
    for s in neural:
        if s.schedule_pass_index is not None:
            si = int(s.schedule_segment_index or 0)
            passes[si] = max(passes.get(si, 0), int(s.schedule_pass_index) + 1)
    physical = [sum(len(getattr(hc, "fused_component_axons", None) or [0]) for hc in s.hard_core_mapping.cores)
                for s in neural]
    return {"physical_cores": int(sum(physical)), "physical_cores_max_stage": int(max(physical)),
            "listed_cores": int(sum(len(s.hard_core_mapping.cores) for s in neural)), "neural_stages": len(neural),
            "compute_stages": sum(s.kind == "compute" for s in m.stages), "segments_scheduled": len(passes),
            "passes_per_segment": {str(k): v for k, v in sorted(passes.items())},
            "max_passes": max(passes.values()) if passes else 1}


def _same_code_as_session(session: dict, start: dict, what: str) -> None:
    if start != session["driver_code_sha256"]:
        changed = sorted(k for k in set(start) | set(session["driver_code_sha256"])
                         if start.get(k) != session["driver_code_sha256"].get(k))
        raise SystemExit(f"REFUSED: the {what} runs other driver code than the row's session: {changed[:8]}")


def _config(key: str) -> dict:
    return json.loads((ROWS / key / "config.json").read_text())


def pipeline_shim(config: dict, scratch: Path):
    ""
    from mimarsinan.pipelining.session import PipelineSession

    doc = json.loads(json.dumps(config))
    doc["_working_directory"] = str(scratch)
    return PipelineSession.from_config(doc, sealed_paths=common.SEALED_PATHS).pipeline


def k9_arithmetic(row: configs.Row):
    ""
    try:
        from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC, VALUE_ARITHMETIC
    except ImportError as exc:
        raise SystemExit(f"REFUSED: RECORD_ABSENT: the head carries no record of the deployed arithmetic "
                         f"(mapping/ir/deployment_dtype.py, U-DAR): {exc}")
    return SPIKING_ARITHMETIC if row.target == "T-LIF" else VALUE_ARITHMETIC


def executor(row: configs.Row, config: dict, mapping, scratch: Path):
    ""
    arithmetic = k9_arithmetic(row)
    if row.target == "T-LIF":
        from mimarsinan.pipelining.core.simulation_factory import build_spiking_hybrid_flow

        flow = build_spiking_hybrid_flow(pipeline_shim(config, scratch), mapping).to(arithmetic.device).eval()
        return flow, arithmetic.input_dtype, "spiking window counts (build_spiking_hybrid_flow)", arithmetic
    from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

    return (ValueHybridCoreFlow(mapping, device=arithmetic.device, dtype=arithmetic.core_dtype), arithmetic.input_dtype,
            "packed value fp64", arithmetic)


def k14_record(census_count: int, k9_correct: int, arithmetic_line: "str | None") -> dict:
    ""
    return {"pipeline_census_count": census_count, "census_arithmetic": arithmetic_line, "fp64_count": k9_correct,
            "difference": census_count - k9_correct}


def shard(key: str, lo: int, hi: int, glo: int, ghi: int) -> int:
    torch.set_num_threads(1)
    from mimarsinan.chip_simulation.value_run.int64_golden import run_golden

    start = common.driver_code_sha256()
    row, session = _row(key), _session(key)
    _same_code_as_session(session, start, "shard")
    member = inventory.member(row.member)
    config = _config(key)
    x, _ = e01_inputs(member, config["data_provider_name"])
    hi = min(hi, len(x))
    mapping = load_mapping(session)
    parts = ROWS / key / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    flow, dtype, kind, arithmetic = executor(row, config, mapping, parts / f"shim_{lo:05d}")
    rec = {"lo": lo, "hi": hi, "times": {}, "executor": kind, "deployed_arithmetic": arithmetic.line()}
    t = time.time()
    with torch.no_grad():
        out = torch.cat([flow(x[i: min(i + 8, hi)].to(dtype)).double() for i in range(lo, hi, 8)]).numpy()
    rec["times"]["packed_fp64_s"] = time.time() - t
    np.save(parts / f"packed_fp64_{lo:05d}_{hi:05d}.npy", out)
    g_lo, g_hi = glo, min(ghi, len(x))
    if g_hi > g_lo and row.target in INTEGER_TARGETS + (("T-INT",) if not row.v3 else ()):
        t = time.time()
        try:
            parts_c, parts_s = [], []
            for j in range(g_lo, g_hi, GOLDEN_BATCH):
                c, st_ = run_golden(mapping, x[j: min(j + GOLDEN_BATCH, g_hi)].double().numpy())
                parts_c.append(np.asarray(c))
                parts_s.append(np.asarray(st_))
            if any(not np.array_equal(p_, parts_s[0]) for p_ in parts_s):
                raise SystemExit("REFUSED: the golden output steps differ between sub-batches")
            codes, steps = np.concatenate(parts_c), parts_s[0]
        except Exception as exc:
            rec["golden_refused"] = f"{type(exc).__name__}: {exc}"[:2000]
        else:
            rec["golden_batch"] = GOLDEN_BATCH
            rec["times"]["golden_s"] = time.time() - t
            np.save(parts / f"golden_codes_{g_lo:05d}_{g_hi:05d}.npy", np.asarray(codes))
            np.save(parts / f"golden_steps_{g_lo:05d}_{g_hi:05d}.npy", np.asarray(steps))
    rec["peak_rss_kb"] = common.peak_rss_kb()
    common.refuse_if_changed(start, f"shard {lo}:{hi}")
    rec["driver_code_listing_sha256"] = common.listing_sha256(start)
    common.atomic_write_json(parts / f"shard_{lo:05d}_{hi:05d}.json", rec)
    print(json.dumps(rec), flush=True)
    return 0


def _join(parts: Path, prefix: str, n: int) -> np.ndarray | None:
    files = sorted(parts.glob(f"{prefix}_*.npy"))
    spans = [tuple(int(v) for v in f.stem.split("_")[-2:]) for f in files]
    if not files or spans[0][0] != 0 or spans[-1][1] != n or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        return None
    return np.concatenate([np.load(f) for f in files])


RESIDENCY_FIELDS = ("residency_plan", "resident_core_indices", "pass_period_census")
PER_PASS_FIELDS = ("resident_core_indices",)
RECORD_WALK_DEPTH = 4


def _per_pass_summary(values: list, passes: int, read_through: str) -> dict:
    ""
    return {"read_through": read_through, "passes": int(passes), "passes_carrying": len(values),
            "marks_total": int(sum(len(v) for v in values))}


def _schema_marks(d: dict) -> tuple[dict, None] | tuple[None, str]:
    ""
    try:
        from mimarsinan.deployment_record.schema.schedule import ScheduleRecord

        segments = ScheduleRecord.from_dict(d["schedule"]).segments()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:400]
    out = {}
    for field in PER_PASS_FIELDS:
        values = [tuple(getattr(s, field)) for s in segments if getattr(s, field, None) is not None]
        if values:
            out[field] = _per_pass_summary(values, len(segments), "schema")
    return out, None


def _walk_residency(d: dict) -> dict:
    ""
    found, per_pass = {}, {}

    def walk(node, depth):
        if depth > RECORD_WALK_DEPTH:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if k in RESIDENCY_FIELDS and k not in PER_PASS_FIELDS and k not in found:
                    found[k] = v if not isinstance(v, list) or len(v) < 64 else {"length": len(v)}
                walk(v, depth + 1)
        elif isinstance(node, (list, tuple)):
            items = [item for item in node if isinstance(item, dict)]
            for field in PER_PASS_FIELDS:
                values = [item[field] for item in items if isinstance(item.get(field), (list, tuple))]
                if values and field not in per_pass:
                    per_pass[field] = _per_pass_summary(values, len(items), "walk")
            for item in node:
                walk(item, depth + 1)

    walk(d, 0)
    for field in PER_PASS_FIELDS:
        if field not in per_pass:
            once = _first_at_depth(d, field)
            if once is not None:
                per_pass[field] = once if not isinstance(once, list) or len(once) < 64 else {"length": len(once)}
    return {**found, **per_pass}


def _first_at_depth(d: dict, field: str):
    ""
    def walk(node, depth):
        if depth > RECORD_WALK_DEPTH or not isinstance(node, dict):
            return None
        for k, v in node.items():
            if k == field:
                return v
            got = walk(v, depth + 1)
            if got is not None:
                return got
        return None

    return walk(d, 0)


def _record_fields(work: Path) -> dict:
    ""
    path = work / "deployment_record.json"
    if not path.exists():
        return {"present": False}
    d = json.loads(path.read_text())
    sched = d.get("schedule", {})
    found = _walk_residency(d)
    schema, refusal = _schema_marks(d)
    if schema is not None:
        for field in PER_PASS_FIELDS:
            found.pop(field, None)
        found.update(schema)
    else:
        for field in PER_PASS_FIELDS:
            if isinstance(found.get(field), dict) and found[field].get("read_through") == "walk":
                found[field]["schema_refusal"] = refusal
    return {"present": True, "sha256": common.sha256_file(path), "identity": d.get("identity"),
            "schedule": {k: v for k, v in sched.items() if isinstance(v, (int, float, str, bool, type(None)))},
            "crossbar": d.get("utilization", {}).get("crossbar"), "accuracy": d.get("accuracy"),
            "residency": found}


def _marks_present(value) -> bool:
    ""
    if isinstance(value, dict) and "passes_carrying" in value:
        return int(value["passes_carrying"]) > 0
    return value is not None


HOST_CONDITION_ROW = ("host_condition", "host")
HOST_CONDITION_STATUSES = ("established", "outside_scope", "not_checked")
HOST_CONDITION_LINE = re.compile(r"\[HostCondition\] Theorem 1 host condition: (\w+)")


def _deployment_record(work: Path) -> dict | None:
    ""
    path = Path(work) / "deployment_record.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _status_of(verdict: str) -> str:
    head = str(verdict).split(":", 1)[0].strip()
    return head if head in HOST_CONDITION_STATUSES else "unknown"


def host_condition(record: dict | None, lines: list[str] | None = None) -> dict:
    ""
    printed = [m[1] for m in (HOST_CONDITION_LINE.search(ln) for ln in (lines or [])) if m]
    if record is None:
        out = {"recorded": False, "reason": "no deployment record in the working directory"}
    else:
        rows = [c for c in (record.get("accuracy") or {}).get("certificates") or []
                if (c.get("name"), c.get("backend")) == HOST_CONDITION_ROW]
        if not rows:
            out = {"recorded": False, "reason": "the deployment record holds no host_condition certificate row "
                                                "(backend host): a head before W49"}
        else:
            row = rows[-1]
            try:
                detail = json.loads(row.get("detail") or "{}")
            except ValueError:
                detail = {"verdict": f"unknown: the row's detail is not JSON: {str(row.get('detail'))[:200]}"}
            kernels = detail.get("kernels") or []
            label = lambda k: f"{k.get('name')} ({k.get('kind')})"
            out = {"recorded": True, "source": "deployment record", "status": _status_of(detail.get("verdict", "")),
                   "passed": bool(row.get("passed")), "verdict": detail.get("verdict"),
                   "condition": detail.get("condition"), "samples": detail.get("samples"),
                   "program_host_ops": detail.get("program_host_ops"), "kernels": len(kernels),
                   "batch_dependent": [label(k) for k in kernels if k.get("batch_dependent")],
                   "nondeterministic": [label(k) for k in kernels if k.get("nondeterministic")],
                   "max_abs_delta": row.get("max_abs_delta"), "rows_in_record": len(rows)}
    if printed:
        out["printed_status"] = printed[-1]
        if out.get("recorded"):
            out["printed_equals_record"] = printed[-1] == out["status"]
    return out


def _pipeline_certificates(lines: list[str], record: dict | None = None) -> dict:
    out = {"lines": {}}
    for ln in lines:
        for tag in ("ValueParityGate", "ValueTwinCertificate", "SpikeCountCertificate", "SpikeCountParityGate",
                    "SpikeTransientReport", "OperandCertificate", "StaticProgramCheck", "HostCondition"):
            if f"[{tag}]" in ln:
                out["lines"].setdefault(tag, []).append(ln.strip()[:600])
        m = re.search(r"\[ValueParityGate\] (PASS|FAIL).*max\|delta\|=([0-9.e+-]+), atol=([0-9.e+-]+), n=(\d+)", ln)
        if m:
            out["K4_value_parity_gate"] = {"verdict": m[1], "max_abs_delta": float(m[2]), "atol": float(m[3]),
                                           "samples": int(m[4])}
        m = re.search(r"\[ValueTwinCertificate\] (PASS|FAIL).*samples=(\d+) windows=(\d+).*max\|delta\|=([0-9.e+-]+) "
                      r"atol=([0-9.e+-]+)", ln)
        if m:
            out["K5_value_twin_certificate"] = {"verdict": m[1], "samples": int(m[2]), "windows": int(m[3]),
                                                "max_abs_delta": float(m[4]), "atol": float(m[5])}
        m = re.search(r"\[ValueTwinCertificate\] (PASS|FAIL) operand program .*?C \(SCM/HCM windows\): (\d+)/(\d+)", ln)
        if m:
            out["K5_value_twin_certificate"] = {"verdict": m[1], "form": "operand program", "windows": int(m[3]),
                                                "mismatched_windows": int(m[2]), "atol": 0.0, "samples": None,
                                                "max_abs_delta": 0.0 if int(m[2]) == 0 else None}
        m = re.search(r"Hard-core Spiking Simulation Test: ([0-9.]+)", ln)
        if m:
            out["census_metric"] = float(m[1])
        m = re.search(r"\[convert_torch_model\] value parity (PASS|FAIL).*relative error ([0-9.e+-]+) <= bound "
                      r"([0-9.e+-]+)", ln)
        if m:
            out.setdefault("K3_conversion", []).append({"verdict": m[1], "relative_error": float(m[2]),
                                                       "bound": float(m[3]), "line": ln.strip()[:400]})
    out["host_condition"] = host_condition(record, lines)
    return out


def _declared_cores(config: dict) -> list:
    plat = config["platform_constraints"]
    over = plat.get("target_overrides", {}).get("cores")
    return over["value"] if over else plat["cores"]


def _subset_equal(declared, recorded) -> bool:
    ""
    if isinstance(declared, dict):
        return isinstance(recorded, dict) and all(_subset_equal(v, recorded.get(k)) for k, v in declared.items())
    return declared == recorded


def _declared_platform_check_v3(identity: dict | None, config: dict, row: configs.Row) -> dict:
    ""
    if not identity:
        return {"passed": False, "reason": "no record identity"}
    plat, declared = identity.get("platform", {}), config["platform_constraints"]
    cores = _declared_cores(config)
    rec_cores = plat.get("cores", [])
    checks = {
        "cores": len(cores) == len(rec_cores) and all(_subset_equal(c, r) for c, r in zip(cores, rec_cores)),
        "allow_scheduling": bool(plat.get("allow_scheduling")) == row.scheduled,
        "max_schedule_passes": plat.get("max_schedule_passes") == declared["max_schedule_passes"],
        "readout_law": plat.get("readout_law") == declared.get("readout_law"),
        "residency_plan": (plat.get("residency_plan") or None) == (declared.get("residency_plan") or None),
        "workload": identity.get("workload") == config["data_provider_name"],
    }
    for k in ("weight_scale_rule", "bias_resolution", "quantizer_state", "layer_core_types", "readout_ranges"):
        if k in declared:
            checks[k] = _subset_equal(declared[k], plat.get(k))
    declared_keys = _declared_keys_check(declared, plat, named=set(checks))
    return {"passed": all(checks.values()) and all(declared_keys.values()), "checks": checks,
            "declared_keys": declared_keys}


def _declared_in(declared, recorded) -> bool:
    ""
    if isinstance(declared, list):
        return isinstance(recorded, list) and len(declared) == len(recorded) and all(
            _declared_in(d, r) for d, r in zip(declared, recorded))
    if isinstance(declared, dict):
        return isinstance(recorded, dict) and all(_declared_in(v, recorded.get(k)) for k, v in declared.items())
    return declared == recorded


def _declared_keys_check(declared: dict, plat: dict, named) -> dict:
    ""
    out: dict = {}
    rec_cores = [c for c in (plat.get("cores") or []) if isinstance(c, dict)]
    for k, v in sorted(declared.items()):
        if k in named or k == "cores":
            continue
        if k in plat:
            out[k] = _declared_in(v, plat[k])
        elif rec_cores and all(k in c for c in rec_cores):
            out[k] = all(_declared_in(v, c[k]) for c in rec_cores)
        else:
            out[k] = False
    for k, ov in sorted((declared.get("target_overrides") or {}).items()):
        if isinstance(ov, dict) and "value" in ov:
            out[f"{k} (resolved override)"] = k in plat and _declared_in(ov["value"], plat[k])
    return out


def _declared_platform_check(identity: dict | None, config: dict, row: configs.Row) -> dict:
    ""
    if row.v3:
        return _declared_platform_check_v3(identity, config, row)
    if not identity:
        return {"passed": False, "reason": "no record identity"}
    plat = identity.get("platform", {})
    declared_cores = config["platform_constraints"]["target_overrides"]["cores"]["value"]
    checks = {
        "cores": [{k: c.get(k) for k in ("max_axons", "max_neurons", "count", "has_bias")} for c in plat.get("cores", [])]
        == declared_cores,
        "allow_scheduling": plat.get("allow_scheduling") == row.scheduled,
        "max_schedule_passes": plat.get("max_schedule_passes") == config["platform_constraints"]["max_schedule_passes"],
        "readout_law": plat.get("readout_law") == config["platform_constraints"]["readout_law"],
        "residency_plan_absent": plat.get("residency_plan") in (None, {}),
        "workload": identity.get("workload") == config["data_provider_name"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def _reference_predictions(row: configs.Row, n: int):
    ""
    member = inventory.member(row.member)
    if member.family == "vit":
        ref = WI2 / "ViT" / "vit_reference.json"
        role = "operand" if row.route == "b-op" else "twin"
        preds = WI2 / "ViT" / f"gate_pred_{role}.npy"
        if not ref.exists() or not preds.exists():
            return None
        p = np.load(preds)[:n]
        return p, None, ref, configs.checkpoint_of(row).e01_correct
    ref = WI2 / row.member / "harness_reference.json"
    if not ref.exists():
        return None
    href = json.loads(ref.read_text())
    return (np.load(WI2 / row.member / "harness_pred_fp64.npy")[:n], np.load(WI2 / row.member / "harness_pred_fp32.npy")[:n],
            ref, href["K2"]["recorded"])


TUNED_NETWORK_FILE = "Normalization Fusion.fused_model.pt"


def tuned_network(work: Path) -> dict:
    ""
    import hashlib

    path = Path(work) / TUNED_NETWORK_FILE
    if not path.is_file():
        return {"present": False, "file": TUNED_NETWORK_FILE}
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model = obj[0] if isinstance(obj, tuple) else obj
    state = model.state_dict()
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode() + str(t.dtype).encode() + str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.view(torch.int16).numpy().tobytes())
    return {"present": True, "file": TUNED_NETWORK_FILE, "file_sha256": common.sha256_file(path),
            "tensors": len(state), "state_sha256": h.hexdigest()}


def _k10(row: configs.Row, packed: np.ndarray, y: np.ndarray, tuned: dict | None = None) -> dict | None:
    ""
    if row.budget in ("B0", "disc") or not row.v3:
        return None
    try:
        base = configs.make_row(row.member, row.target, row.route, "B0")
    except SystemExit:
        return None
    base_dir, source = common.needed_row_dir(base.key, ROWS)
    path = base_dir / "packed_fp64.npy"
    res = base_dir / "result.json"
    if not path.exists() or not res.exists():
        return {"against": base.key, "status": "B0 row not harvested yet", "against_rows_root": source["rows_root"]}
    other = np.load(path)
    if other.shape != packed.shape:
        return {"against": base.key, "status": f"B0 outputs {other.shape} against {packed.shape}"}
    pb, po = other.argmax(1), packed.argmax(1)
    out = {"against": base.key, "outputs": int(packed.size), "outputs_differ": int((packed != other).sum()),
           "max_abs_delta": float(np.abs(packed - other).max()), "pred_differ": int((pb != po).sum()),
           "accuracy_difference_images": int((po == y).sum() - (pb == y).sum()),
           "b0_result_sha256": common.sha256_file(res), "passed": bool(np.array_equal(packed, other)),
           "against_rows_root": source["rows_root"], "against_read_from": source["read_from"]}
    if row.route in declarations.TUNED_ROUTES:
        theirs = (json.loads(res.read_text()).get("tuned_network") or {}).get("state_sha256")
        mine = (tuned or {}).get("state_sha256")
        same = bool(mine) and mine == theirs
        out.update(same_tuned_network=same, b0_tuned_network_sha256=theirs, tuned_network_sha256=mine,
                   reading="budget invariance" if same else
                   "two training draws (the B0 row deployed another tuned network): not a budget-invariance statement")
    return out


K10_ADDENDUM_FORMAT = "e2e-row-k10-addendum/1"
K10_ADDENDUM_FILE = "k10_addendum.json"


def _e01_labels(member: str) -> tuple[Path | None, np.ndarray | None]:
    ""
    for wi in (WI2.name, WI1.name):
        path = common.RECORDS / wi / member / "e01_labels.npy"
        if path.is_file():
            return path, np.load(path)
    return None, None


def k10_addendum(key: str, out: Path | None = None) -> dict:
    ""
    start = common.driver_code_sha256()
    row = _row(key)
    row_dir = ROWS / key
    res_path = row_dir / "result.json"
    try:
        res = json.loads(res_path.read_text())
    except (OSError, ValueError):
        res = None
    if not isinstance(res, dict) or res.get("status") != "complete" or res.get("format") != RESULT_FORMAT:
        status = res.get("status") if isinstance(res, dict) else "absent"
        raise SystemExit(f"REFUSED: K10_ADDENDUM_ROW: {key} has no complete result in {ROWS} (status {status}): an "
                         "addendum reads a row of record, it never runs one")
    if row.budget in ("B0", "disc") or not row.v3:
        raise SystemExit(f"REFUSED: K10_ADDENDUM_NOT_APPLICABLE: {key} reads no K10 (a B0 row, a discovery row or a v2 "
                         "row)")
    target = Path(out) if out is not None else row_dir / K10_ADDENDUM_FILE
    if target.name == res_path.name or target.resolve() == res_path.resolve():
        raise SystemExit(f"REFUSED: K10_ADDENDUM_TARGET: {target} is a result.json: an addendum never writes a result")
    if target.exists():
        raise SystemExit(f"REFUSED: K10_ADDENDUM_EXISTS: {target} exists: an addendum is never overwritten")
    packed_path = row_dir / "packed_fp64.npy"
    want = ((res.get("K9") or {}).get("files") or {}).get("packed_fp64.npy")
    if not packed_path.is_file() or not want or common.sha256_file(packed_path) != want:
        raise SystemExit(f"REFUSED: K10_ADDENDUM_PACKED: {packed_path} is absent or not at the digest the result "
                         f"records ({str(want)[:16]})")
    packed = np.load(packed_path)
    labels_path, y = _e01_labels(row.member)
    n = len(packed)
    if labels_path is None or y is None or len(y) < n:
        raise SystemExit(f"REFUSED: K10_ADDENDUM_LABELS: no E01 labels of {row.member} for {n} images under "
                         f"{common.RECORDS}")
    y = y[:n]
    correct = int((packed.argmax(1) == y).sum())
    if n != res["K9"].get("images") or correct != res["K9"].get("correct"):
        raise SystemExit(f"REFUSED: K10_ADDENDUM_LABELS: {labels_path} gives {correct} correct of {n} images, the "
                         f"result records {res['K9'].get('correct')} of {res['K9'].get('images')}")
    k10 = _k10(row, packed, y, tuned=res.get("tuned_network"))
    own = (res.get("inputs") or {}).get("session_driver_code_sha256") or {}
    declared = {"footprint_root": common.FOOTPRINT_ROOT, "carried_rows_root": common.CARRIED_ROWS_ROOT}
    rec = {"format": K10_ADDENDUM_FORMAT, "row": key, "rows_root": str(ROWS),
           "result_sha256": common.sha256_file(res_path), "packed_fp64_sha256": want,
           "labels": {"file": str(labels_path), "sha256": common.sha256_file(labels_path), "images": n,
                      "correct": correct},
           "declared": {k: None if v is None else str(v) for k, v in declared.items()},
           "K10": k10, "result_K10": res.get("K10"),
           "equal_to_result": json.dumps(k10, sort_keys=True) == json.dumps(res.get("K10"), sort_keys=True),
           "driver_code_sha256": start, "driver_listing_sha256": common.listing_sha256(start),
           "driver_files_differing_from_session": sorted(k for k in set(own) | set(start)
                                                         if own.get(k) != start.get(k)),
           "rule": "the row's K10 read again by the harvest's own K10 path on the driver in force; the row was not run "
                   "again and its result.json was not written",
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    common.refuse_if_changed(start, "the K10 addendum")
    if target.exists():
        raise SystemExit(f"REFUSED: K10_ADDENDUM_EXISTS: {target} exists: an addendum is never overwritten")
    common.atomic_write_json(target, rec)
    return rec


def _k12(row: configs.Row, record: dict, census: dict, config: dict, work: Path) -> dict | None:
    ""
    import k12_census

    declared = config["platform_constraints"].get("residency_plan")
    if not declared:
        return None
    found = record.get("residency", {})
    out = {"declared": declared, "record_residency_plan": found.get("residency_plan"),
           "plan_equals_declared": _subset_equal(declared, found.get("residency_plan") or {}),
           "resident_core_indices_present": _marks_present(found.get("resident_core_indices")),
           "resident_core_marks": found.get("resident_core_indices"),
           "passes_above_one": census["max_passes"] > 1}
    out["period_census"] = k12_census.census(work, declared)
    out["passed"] = bool(out["plan_equals_declared"] and out["resident_core_indices_present"] and out["passes_above_one"]
                         and out["period_census"].get("passed"))
    return out


DESIGN_PASSES_FIELD = {"exploration": "Q", "main_level": "passes"}


def _design(row: configs.Row, census: dict) -> dict | None:
    ""
    if row.budget not in declarations.DESIGN_BUDGETS:
        return None
    source = declarations.DESIGN_SOURCES[row.budget]
    d = declarations.design_reading(row.member, row.budget)
    field = DESIGN_PASSES_FIELD[source]
    return {source: d, "toolchain": {"geometry": row.geometry, "declared_cores": d["C"],
                                     "physical_cores": census["physical_cores"],
                                     "passes": census["max_passes"],
                                     "position_multiplexing": row.budget in declarations.MULTIPLEXED,
                                     "residency_plan": declarations.residency_plan(row.budget, row.member)},
            f"{field}_{source}_against_toolchain_passes": [d[field], census["max_passes"]]}


CERTIFICATE_BACKEND = re.compile(r"spike-count certificate \[(\w+)/")
TWIN_BACKEND = "hcm"


def _lif_certificates(certs: dict, nevresim_declared: bool = True) -> dict:
    ""
    lines = certs.get("lines", {})
    count_lines = lines.get("SpikeCountCertificate", [])
    nev = [ln for ln in count_lines if "nevresim" in ln]
    k7, backends = [], {}
    for ln in count_lines:
        if "nevresim" in ln:
            continue
        m = CERTIFICATE_BACKEND.search(ln)
        if m is None or m[1] == TWIN_BACKEND or "streaming-twin" in ln:
            k7.append(ln)
        else:
            backends.setdefault(m[1], []).append(ln)
    verdict = lambda ls: ("not run" if not ls else ("FAIL" if any("FAIL" in ln or "violated" in ln for ln in ls)
                                                       else "PASS"))
    k8 = {"verdict": verdict(nev), "lines": nev[:6]}
    if not nevresim_declared:
        k8["declared"] = False
        if nev:
            k8["note"] = "nevresim printed a certificate line although the row declares it off"
        else:
            k8["reason"] = declarations.NEVRESIM_OFF_K8
    return {"K7_window_counts": {"verdict": verdict(k7), "lines": k7[:6]}, "K8_nevresim": k8,
            "backend_spike_parity": {b: {"verdict": verdict(ls), "lines": ls[:6]} for b, ls in sorted(backends.items())}}


def merge(key: str, golden: int, images: int | None = None) -> int:
    start = common.driver_code_sha256()
    row, session = _row(key), _session(key)
    _same_code_as_session(session, start, "merge")
    member = inventory.member(row.member)
    out = ROWS / key
    config = json.loads((out / "config.json").read_text())
    if common.sha256_file(out / "config.json") != session["config_sha256"]:
        raise SystemExit("REFUSED: config.json differs from the session's digest")
    work = Path(session["working_directory"])
    persisted = json.loads((work / "_RUN_CONFIG" / "config.json").read_text())
    x, y = e01_inputs(member, config["data_provider_name"])
    if images is not None and int(images) < len(y):
        if row.budget != "disc":
            raise SystemExit(f"REFUSED: only a discovery row reads fewer than its provider's {len(y)} images")
        x, y = x[: int(images)], y[: int(images)]
    n = len(y)
    golden = min(int(golden), n)
    packed = _join(out / "parts", "packed_fp64", n)
    if packed is None:
        raise SystemExit(f"REFUSED: {key} packed shards do not cover 0..{n}")
    shard_recs = [json.loads(p.read_text()) for p in sorted((out / "parts").glob("shard_*.json"))]
    gcodes = _join(out / "parts", "golden_codes", golden)
    gsteps = None
    if gcodes is not None:
        step_parts = [np.load(f) for f in sorted((out / "parts").glob("golden_steps_*.npy"))]
        if any(not np.array_equal(sp, step_parts[0]) for sp in step_parts):
            raise SystemExit("REFUSED: the golden output steps differ between shards")
        gsteps = step_parts[0]
    pred = packed.argmax(1)
    top = packed.max(1, keepdims=True)
    mapping = load_mapping(session)
    census = mapping_census(mapping)
    if declarations_typed(row):
        census["physical_cores_by_width"] = cores_by_width(mapping)
    record = _record_fields(work)
    certs = _pipeline_certificates(session.get("certificate_lines", []), _deployment_record(work))
    params, plat = config["deployment_parameters"], config["platform_constraints"]
    resolved = json.loads((work / footprints.RESOLVED).read_text()) if (work / footprints.RESOLVED).exists() else None
    res = {
        "format": RESULT_FORMAT, "row": key, "member": row.member, "target": row.target, "route": row.route,
        "budget": row.budget, "geometry": row.geometry, "status": "complete",
        "smoke": config["data_provider_name"] != e01_provider.PROVIDER_NAME,
        "inputs": {"toolchain": session["toolchain"], "driver_code_sha256": start,
                   "session_driver_code_sha256": session["driver_code_sha256"],
                   "config_sha256": session["config_sha256"],
                   "persisted_config_equals_declared": persisted == config,
                   "checkpoint": configs.checkpoint_of(row).path, "checkpoint_sha256": session["K0"]["sha256"],
                   "weight_source_sha256": session["weight_source"]["sha256"], "provider": session.get("provider"),
                   "e01_inputs_fp32_sha256": common.sha256_bytes(x.numpy().tobytes()),
                   "quantizer_state": session.get("quantizer_state"),
                   "sealed_paths": session.get("sealed_paths")},
        "K0": session["K0"],
        "packing_declaration": None if resolved is None else footprints.packing_declaration(resolved),
        "deployment": {**census, "declared_count": session["declared_count"], "footprint_input": session["footprint_input"],
                       "footprint_source": session.get("footprint_source"),
                       "activation_bits": plat.get("activation_bits"),
                       "value_parity_samples": params.get("value_parity_samples"),
                       "spike_count_parity_samples": params.get("spike_count_parity_samples"),
                       "simulation_batch_size": params.get("simulation_batch_size"),
                       "pipeline_wall_s": session["wall_s"], "pipeline_peak_rss_kb": session["peak_rss_kb"],
                       "step_walls_s": session["step_walls_s"], "steps_run": session["steps_run"]},
        "record": record,
        "K4_K5_pipeline": {k: v for k, v in certs.items() if k.startswith(("K4", "K5"))},
        "certificate_lines_by_tag": certs.get("lines", {}),
        "K9": {"images": n, "correct": int((pred == y).sum()), "argmax_ties": int(((packed == top).sum(1) > 1).sum()),
               "executor": shard_recs[0].get("executor", "packed value fp64") if shard_recs else None,
               "deployed_arithmetic": shard_recs[0].get("deployed_arithmetic") if shard_recs else None,
               "files": {"packed_fp64.npy": common.save_npy(out / "packed_fp64.npy", packed),
                         "pred.npy": common.save_npy(out / "pred.npy", pred)},
               "outputs_sha256": common.sha256_array(packed)},
    }
    if gcodes is not None:
        gv = np.asarray(gcodes * gsteps, dtype=np.float64).reshape(packed[:golden].shape)
        d = np.abs(gv - packed[:golden])
        res["K6"] = {"samples": int(golden), "outputs": int(d.size), "mismatched_outputs": int((d > 0).sum()),
                     "max_abs_delta": float(d.max()), "passed": bool(d.max() == 0.0),
                     "nonzero_output_fraction": float((gcodes != 0).mean()),
                     "files": {"golden_codes.npy": common.save_npy(out / "golden_codes.npy", gcodes)}}
    refused = [s["golden_refused"] for s in shard_recs if s.get("golden_refused")]
    if refused:
        res["K6"] = {"refused": refused[0], "passed": False}
    if row.target == "T-LIF":
        res.update(_lif_certificates(certs, nevresim_declared=params.get("enable_nevresim_simulation") is not False))
    if row.budget == "disc" and row.v3:
        res["K14"] = {"applicable": False, "reason": "discovery row: the census reads a seeded 64-image subsample"}
    elif "census_metric" in certs:
        census_count = int(round(certs["census_metric"] * n))
        res["K14"] = k14_record(census_count, res["K9"]["correct"], res["K9"].get("deployed_arithmetic"))
    k11 = _declared_platform_check(record.get("identity"), config, row)
    k11["max_passes"] = census["max_passes"]
    k11["non_vacuous"] = (census["max_passes"] > 1) if row.scheduled else True
    k11["passed"] = bool(k11["passed"] and k11["non_vacuous"])
    if row.budget == "B0":
        declared = session["declared_count"]
        own = census.get("physical_cores_by_width") if isinstance(declared, dict) else census["physical_cores"]
        k11["resident_equals_declared_count"] = own == declared
        source = session.get("footprint_source") or {}
        if source.get("packing") is not None and resolved is not None:
            k11["packing_equals_footprint_source"] = footprints.packing_declaration(resolved) == source["packing"]
        k11["passed"] = bool(k11["passed"] and k11["resident_equals_declared_count"]
                             and k11.get("packing_equals_footprint_source", True))
    res["K11"] = k11
    tuned = row.route in declarations.TUNED_ROUTES
    if tuned:
        res["tuned_network"] = tuned_network(work)
        res["device"] = session.get("device")
        res["nondeterminism"] = session.get("nondeterminism")
    k10 = _k10(row, packed, y, tuned=res.get("tuned_network"))
    if k10 is not None:
        res["K10"] = k10
    k12 = _k12(row, record, census, config, work)
    if k12 is not None:
        res["K12"] = k12
    design = _design(row, census)
    if design is not None:
        res["design"] = design
    static = certs.get("lines", {}).get("StaticProgramCheck")
    if static:
        res["static_program_check"] = {"lines": static,
                                       "theorem_1_failed": any("Theorem 1 failed" in ln for ln in static),
                                       "proposition_2_failed": any("Proposition 2 failed" in ln for ln in static)}
    res["host_condition"] = certs["host_condition"]
    if certs.get("K3_conversion"):
        res["K3"] = {"lines": certs["K3_conversion"], "passed": all(c["verdict"] == "PASS" for c in certs["K3_conversion"]),
                     "inputs": "the toolchain's probe set (seeded fp64 inputs); the 512-E01 arm of design.md is not run"}
    if configs.ROUTES[row.route] == "qat":
        ref = _reference_predictions(row, n)
        if ref is not None:
            p64, p32, path, recorded = ref
            res["K13"] = {"reference": str(path.relative_to(common.E2E)), "reference_sha256": common.sha256_file(path),
                          "images": n, "agree_harness_fp64": int((pred == p64).sum()),
                          "harness_fp64_correct": int((p64 == y).sum()), "harness_recorded": recorded,
                          "differing_vs_fp64": int((pred != p64).sum())}
            if p32 is not None:
                res["K13"].update(agree_harness_fp32=int((pred == p32).sum()), harness_fp32_correct=int((p32 == y).sum()))
            if member.family == "vit":
                res["K13"]["note"] = "the reference is the gate's own E01 predictions (fp32, GPU), not a CPU fp64 re-read"
        else:
            res["K13"] = {"reference": None, "note": "harness reference not yet harvested (WI-2 for this member)"}
    else:
        ref = WI1 / row.member / "float_pred_fp32.npy"
        if ref.exists():
            fp = np.load(ref)[:n]
            res["float_agreement"] = {"agree_float_fp32": int((pred == fp).sum()), "float_correct": int((fp == y).sum())}
    session_listing = common.listing_sha256(session["driver_code_sha256"])
    foreign = [s["lo"] for s in shard_recs if s.get("driver_code_listing_sha256") != session_listing]
    if foreign:
        raise SystemExit(f"REFUSED: shards starting at {foreign} ran other driver code than the row's session")
    if tuned:
        legs = no_update.training_legs(work, out / "pipeline_stdout.log")
        common.atomic_write_json(out / "training_legs.json", legs)
        res["training_legs"] = {"file": "training_legs.json", "file_sha256": common.sha256_file(out / "training_legs.json"),
                                "endpoint_legs": legs["endpoint_legs"], "fast_ladders": legs["fast_ladders"],
                                "ledgers": legs["ledgers"], "fine_tune_passes": legs["fine_tune_passes"]}
        res["no_update"] = {"applicable": False, "reason": "tuned arm: its training legs are declared on "
                                                           "(training_legs, tlif_tuned/design.md section 5)",
                            "a_preload_equals_checkpoint": no_update.check_preload(
                                work, Path(session["weight_source"]["path"]), session["weight_source"]["sha256"])}
    else:
        nu = no_update.check_work(work, Path(session["weight_source"]["path"]), session["weight_source"]["sha256"],
                                  **no_update.declared_grids(config), stdout_log=out / "pipeline_stdout.log")
        nu_sha = None
        if nu.get("applicable"):
            common.atomic_write_json(out / "no_update.json", nu)
            nu_sha = common.sha256_file(out / "no_update.json")
        res["no_update"] = {k: v for k, v in nu.items() if k != "per_perceptron"} | {"file_sha256": nu_sha}
    res["harvest"] = {"shards": len(shard_recs),
                      "packed_fp64_s_per_image": sum(s["times"]["packed_fp64_s"] for s in shard_recs) / n,
                      "golden_s_per_image": (sum(s["times"].get("golden_s", 0.0) for s in shard_recs) / golden)
                      if gcodes is not None else None,
                      "shard_peak_rss_kb": max(s["peak_rss_kb"] for s in shard_recs)}
    common.refuse_if_changed(start, "the merge")
    common.atomic_write_json(out / "result.json", res)
    print(json.dumps({k: res.get(k) for k in ("row", "K9", "K3", "K6", "K10", "K11", "K12", "K13", "K14", "K4_K5_pipeline",
                                              "K7_window_counts", "K8_nevresim", "backend_spike_parity",
                                              "no_update")},
                     indent=1, default=str)[:6000])
    return 0


def declarations_typed(row: configs.Row) -> bool:
    import declarations

    return row.v3 and declarations.is_typed(row.member)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--row", required=True)
    s.add_argument("--lo", type=int, required=True)
    s.add_argument("--hi", type=int, required=True)
    s.add_argument("--glo", type=int, default=0, help="golden slice of the first --golden images, start")
    s.add_argument("--ghi", type=int, default=0, help="golden slice end (exclusive); equal to --glo for none")
    m = sub.add_parser("merge")
    m.add_argument("--row", required=True)
    m.add_argument("--golden", type=int, default=512)
    m.add_argument("--images", type=int, default=None, help="a discovery row: the first N images only")
    kp = sub.add_parser("k10", help="[U-K10R] a complete row's K10 read again, as an addendum beside its result")
    kp.add_argument("--row", required=True)
    kp.add_argument("--out", default=None, help="the addendum's file (default: k10_addendum.json in the row's folder)")
    a = ap.parse_args()
    if a.cmd == "shard":
        return shard(a.row, a.lo, a.hi, a.glo, a.ghi)
    if a.cmd == "k10":
        rec = k10_addendum(a.row, Path(a.out) if a.out else None)
        print(json.dumps({k: rec[k] for k in ("row", "rows_root", "result_sha256", "declared", "K10", "equal_to_result",
                                              "driver_files_differing_from_session")}, indent=1, sort_keys=True))
        return 0
    return merge(a.row, a.golden, a.images)


if __name__ == "__main__":
    sys.exit(main())
