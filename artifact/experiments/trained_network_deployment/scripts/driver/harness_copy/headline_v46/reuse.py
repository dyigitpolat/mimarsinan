""

from __future__ import annotations

import json
from pathlib import Path

import screen_common as sc

REPEAT_DIR = "_smoke_repeat"
LANE_KEYS = ("class", "head", "tail", "kinds", "nodes", "quantized", "logits", "quantizer", "rounding", "signed",
             "theta", "shift")


def compared_fields(result: dict) -> dict:
    ""
    h = result["phases"]["qat"]["history"][0]
    return {
        "float_val_correct_reeval": result.get("float_val_correct_reeval"),
        "calibration": {k: result.get("calibration", {}).get(k) for k in ("quantizers", "signed", "relu_lanes_signed")},
        "lane_census": {k: v for k, v in sorted(result.get("lane_census", {}).items()) if k != "w_bits"},
        "qat_first_epoch": {k: h.get(k) for k in ("loss", "train_acc", "val_correct", "lr", "updates")},
        "lanes": [{k: lane.get(k) for k in LANE_KEYS} for lane in result.get("lanes", [])],
    }


def _differing(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def check(rules: dict | None = None) -> dict:
    ""
    rules = rules or sc.load_rules()
    lock = sc.locked_digests()
    cands = rules["reuse"]["candidates"]
    out = {"rule": rules["reuse"]["rule"], "pairs": {}, "identity": None, "problems": []}
    first_job = rules["smoke"]["repeat"][0]
    mine_first = None
    for job, arm in cands.items():
        mine = _read(sc.RUNS / "_smoke" / job / "result.json")
        theirs_path = sc.SCREEN_P_RUNS / "_smoke" / arm / "result.json"
        theirs = _read(theirs_path)
        row = {"arm": arm, "smoke": f"runs/_smoke/{job}/result.json",
               "screen_v42p_smoke": f"screen_v42p/runs/_smoke/{arm}/result.json", "equal": False}
        if mine is None or mine.get("status") != "ok":
            row["why"] = "this harness's smoke record is missing or not ok"
        elif mine.get("digests") != lock:
            row["why"] = "this harness's smoke record was not written under headline_lock.json"
        elif theirs is None or sc.sha256_file(theirs_path) != lock["screen_p_inputs_sha256"]["smoke_result"][arm]:
            row["why"] = "the screen_v42p smoke record differs from its pinned sha256"
        else:
            a, b = compared_fields(mine), compared_fields(theirs)
            row["equal"] = a == b
            row["differing_fields"] = _differing(a, b)
            row["mine"] = {k: a[k] for k in ("float_val_correct_reeval", "qat_first_epoch")}
            row["screen_v42p"] = {k: b[k] for k in ("float_val_correct_reeval", "qat_first_epoch")}
        if job == first_job:
            mine_first = mine
        out["pairs"][job] = row
    rep = _read(sc.RUNS / REPEAT_DIR / first_job / "result.json")
    ident = {"job": first_job, "runs": [f"runs/_smoke/{first_job}", f"runs/{REPEAT_DIR}/{first_job}"],
             "identical": False}
    if mine_first is None or rep is None or rep.get("status") != "ok" or mine_first.get("status") != "ok":
        ident["why"] = "a smoke record of the identity pair is missing or not ok"
    elif rep.get("digests") != lock or mine_first.get("digests") != lock:
        ident["why"] = "a smoke record of the identity pair was not written under headline_lock.json"
    else:
        a, b = compared_fields(mine_first), compared_fields(rep)
        ident["identical"] = a == b
        ident["differing_fields"] = _differing(a, b)
        ident["gpus"] = [mine_first.get("cuda_visible_devices"), rep.get("cuda_visible_devices")]
    out["identity"] = ident
    reuse_ok = bool(ident["identical"]) and all(r["equal"] for r in out["pairs"].values())
    out["decision"] = "reuse" if reuse_ok else "rerun"
    out["reused_jobs"] = sorted(cands) if reuse_ok else []
    out["reused_values"] = reused_values(rules) if reuse_ok else {}
    return out


def reused_values(rules: dict) -> dict:
    ""
    lock = sc.locked_digests()
    p_lock = json.loads(sc.SCREEN_P_LOCK_JSON.read_text())
    out = {}
    for job, arm in rules["reuse"]["candidates"].items():
        path = sc.SCREEN_P_RUNS / arm / "result.json"
        if sc.sha256_file(path) != lock["screen_p_inputs_sha256"]["result"][arm]:
            raise SystemExit(f"REFUSED: {path} differs from its pinned sha256")
        res = json.loads(path.read_text())
        if res.get("status") != "ok" or res.get("digests") != p_lock:
            raise SystemExit(f"REFUSED: screen_v42p arm {arm} did not finish ok under its lock")
        out[job] = {"arm": arm, "qat_val_correct": res["qat_val_correct"], "val_total": res["val_total"],
                    "float_val_correct": res.get("float_val_correct"), "source": f"screen_v42p/runs/{arm}/result.json"}
    return out


def decision_path() -> Path:
    return sc.RUNS / "_launch" / "reuse_decision.json"


def frozen_or_current(rules: dict | None = None) -> dict:
    ""
    p = decision_path()
    if p.exists():
        return json.loads(p.read_text())
    return check(rules)


def mismatch(frozen: dict, now: dict) -> list[str]:
    ""
    out = []
    if frozen.get("decision") != now["decision"]:
        out.append(f"decision: frozen {frozen.get('decision')!r}, current check {now['decision']!r}")
    if sorted(frozen.get("reused_jobs") or []) != sorted(now["reused_jobs"]):
        out.append(f"reused jobs: frozen {frozen.get('reused_jobs')}, current check {now['reused_jobs']}")
    fv, nv = frozen.get("reused_values") or {}, now.get("reused_values") or {}
    if fv != nv:
        jobs = sorted(j for j in set(fv) | set(nv) if fv.get(j) != nv.get(j))
        out.append("reused values differ from screen_v42p's pinned records for "
                   + ", ".join(f"{j} (frozen {fv.get(j)}, recomputed {nv.get(j)})" for j in jobs))
    return out


def _frozen_checked(rules: dict | None, write: bool) -> dict:
    rules = rules or sc.load_rules()
    now = check(rules)
    p = decision_path()
    if not p.exists():
        if write:
            sc.atomic_write_json(p, now)
        return now
    frozen = json.loads(p.read_text())
    problems = mismatch(frozen, now)
    if problems:
        raise SystemExit(f"REFUSED: the frozen reuse decision ({p}) differs from the current check:\n  "
                         + "\n  ".join(problems))
    return frozen


def verified_decision(rules: dict | None = None) -> dict:
    ""
    return _frozen_checked(rules, write=False)


def freeze(rules: dict | None = None) -> dict:
    ""
    return _frozen_checked(rules, write=True)
