""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import signal
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

import w38common as c
import configs
import lockcheck
import preflight
import run_row

FORMAT = "w38-structure/1"


STOP_GRACE_S = 120.0


def _ladder_script(root: Path):
    ""
    path = root / "scripts" / "scaling_ladder.py"
    if not path.is_file():
        raise configs.RowRefusal(f"REFUSED: the head has no {path}")
    spec = importlib.util.spec_from_file_location("w38_head_scaling_ladder", path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def build(a, gates: dict | None = None) -> dict:
    import torch
    from torchvision import models

    import members_builder
    import provider

    from mimarsinan.deployment_record.introspection.scaling_ladder import (
        LadderNetwork, certificate_record, packed_tiles, price_program, shape_census)
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fractions
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    from mimarsinan.torch_mapping.converter import convert_torch_model

    token = getattr(a, "token", None) or configs.arms()["structure"]["budgets"][0]
    rec = {"format": FORMAT, "group": f"{a.member}.{a.bundle}.{a.arm}", "mode": a.mode, "token": token,
           "plane": "shape-only plane (the toolchain's scaling ladder)", "departures": configs.departures(a.arm)}
    rec["preflight"] = gates
    rec["toolchain"] = gates["toolchain"] if gates is not None else c.toolchain_provenance(a.worktree)
    ladder = _ladder_script(Path(rec["toolchain"]["path"]))
    bundle = configs.resolve_bundle(a.bundle)
    rec["bundle"] = {k: bundle[k] for k in ("name", "role", "values", "lock_check")}
    if bundle["role"] != "of record" and a.mode != "smoke":
        raise configs.RowRefusal(f"REFUSED: bundle {a.bundle} is {bundle['role']}")
    member = c.members()[a.member]
    crop = int(member["transforms"]["crop_size"][0])
    shape = (3, crop, crop)
    row = configs.make_row(a.member, a.bundle, a.arm, token)
    caps = configs.law_capabilities()
    law = {p: v for p, v in configs.arms()["law_declarations"]["values"].items() if caps[p]}
    scratch = Path(a.runs) / f"{rec['group']}.structure.{token}"
    doc = configs.make_config(row, bundle, weight_source=scratch / "unused.pt", view_dir=scratch / "view",
                              work_root=scratch, law=law)
    flat = configs.flat_run_config(doc)
    rec["conversion_keys"] = {k: flat.get(k) for k in ("encoding_layer_placement", "decompose_leaf_modules")}
    platform = build_platform_constraints_resolved(dict(flat))
    rec["platform"] = {k: platform.get(k) for k in ("cores", "allow_coalescing", "allow_neuron_splitting",
                                                     "allow_scheduling", "max_schedule_passes", "weight_bits")}
    rec["platform"]["physics_profile"] = (platform.get("platform_physics_resolved") or {}).get("name")
    C = sum(int(t.get("count", 1)) for t in platform["cores"])
    t0 = time.time()
    torch.manual_seed(0)
    model = models.get_model(a.member, weights=c.weights_of(member)).eval()
    rec["canonical_state_digest"] = c.state_digest(model.state_dict())
    if rec["canonical_state_digest"] != member["weights_file"]["canonical_state_digest"]:
        raise SystemExit("REFUSED: the canonical state differs from members.json")
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        flow = convert_torch_model(
            model, shape, 1000, packaging=MVM_PACKAGING, strict=False,
            encoding_layer_placement=str(flat.get("encoding_layer_placement", "subsume")),
            decompose_leaf_modules=bool(flat.get("decompose_leaf_modules", False))).eval()
        params, macs, ops = estimate_onchip_fractions(
            flow, shape, 1000, metrics=("params", "macs", "ops"),
            encoding_placement=str(flat.get("encoding_layer_placement", "subsume")))
    rec["walls_s"] = {"convert": time.time() - t0}
    rec["host_census"] = {"host_macs": int(macs.host), "onchip_macs": int(macs.onchip), "total_macs": int(macs.total),
                          "host_mac_share": (macs.host / macs.total) if macs.total else None,
                          "host_params": int(params.host), "onchip_params": int(params.onchip),
                          "host_ops": int(ops.host), "total_ops": int(ops.total),
                          "source": "mapping.verification.onchip_fraction.estimate_onchip_fractions"}
    network = LadderNetwork(
        name=a.member, mapper_repr=flow.get_mapper_repr(), resolution=shape,
        total_params=float(sum(p.numel() for p in flow.parameters())),
        host_census={"host_macs": macs.host, "onchip_macs": macs.onchip, "compute_op_count": ops.host,
                     "host_params": params.host, "onchip_params": params.onchip},
        conversion=certificate_record(getattr(flow, "conversion_certificate", None)))
    rec["conversion_certificate"] = network.conversion
    t1 = time.time()
    census = shape_census(network, platform, wire_census=True)
    rec["walls_s"]["layout"] = time.time() - t1
    t2 = time.time()
    F = packed_tiles(census, platform)
    rec["walls_s"]["pack"] = time.time() - t2
    capacity = sum(int(t.get("count", 1)) * int(t["max_axons"]) * int(t["max_neurons"]) for t in platform["cores"])
    rec["shape"] = {"softcores": len(census.softcores), "segments": census.segments,
                    "host_segments": census.host_segments, "logical_cells": census.logical_cells,
                    "as_mapped_cells": census.as_mapped_cells, "packed_tiles_F": int(F), "C": C,
                    "x_packed": F / C, "x_fluid": census.logical_cells / capacity, "capacity_cells": capacity}
    t3 = time.time()
    priced = price_program(network, census, platform, ladder.program_facts, tiles=F)
    rec["walls_s"]["schedule_and_price"] = time.time() - t3
    if priced.refusal is not None:
        rec["schedule"] = {"refusal": dict(priced.refusal)}
    else:
        rec["schedule"] = {"passes": priced.passes, "reprogram_passes": priced.reprogram_passes,
                           "written_passes": priced.written_passes, "latency_steps": priced.latency_steps,
                           "reprogrammed_bytes_per_inference": priced.period_bytes,
                           "programmed_bytes_once": priced.once_bytes, "programmed_cores_once": priced.once_cores,
                           "max_writes": list(priced.max_writes) if priced.max_writes else None,
                           "priced_terms": {k: {n: getattr(b, "nominal", None) for n, b in v.items()}
                                            for k, v in (priced.terms or {}).items()},
                           "pricing_refusals": priced.refusals}
    rec["peak_rss_kb"] = c.peak_rss_kb()
    return rec


def record_name(group: str, token: str, tokens: list[str]) -> str:
    ""
    return f"{group}.structure.json" if token == tokens[0] else f"{group}.structure.{token}.json"


class _Stopped(BaseException):
    ""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def _on_signal(signum, frame):
    raise _Stopped(signum)


_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def _kill_group(pid: int) -> None:
    ""
    for kill, target in ((os.killpg, pid), (os.kill, pid)):
        try:
            kill(target, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def run_child(child, timeout: float) -> int | str:
    ""
    main_thread = threading.current_thread() is threading.main_thread()
    sys.stdout.flush()
    sys.stderr.flush()
    if main_thread:
        signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS)
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.setpgid(0, 0)
            if main_thread:
                for sig in _SIGNALS:
                    signal.signal(sig, signal.SIG_DFL)
                signal.pthread_sigmask(signal.SIG_UNBLOCK, _SIGNALS)
            code = int(child())
        except BaseException:
            traceback.print_exc()
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(code)
    try:
        os.setpgid(pid, pid)
    except (ProcessLookupError, PermissionError):
        pass
    if main_thread:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _SIGNALS)
    deadline = time.time() + timeout
    try:
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                return -os.WTERMSIG(status) if os.WIFSIGNALED(status) else os.WEXITSTATUS(status)
            if time.time() >= deadline:
                _kill_group(pid)
                return "timeout"
            time.sleep(0.2)
    except BaseException:
        _kill_group(pid)
        raise


def record_ending(a, group: str, tokens: list[str], code, checked: dict, limit_s: float, wall_s: float) -> list[str]:
    ""
    records = Path(a.records)
    have = {name for name, _ in preflight.structure_records(records, group)}
    absent = [t for t in tokens if record_name(group, t, tokens) not in have]
    if not absent:
        return []
    c.refuse_if_changed(checked["start"], "the structural pack's ending record")
    stop = preflight.stop_name(code)
    for token in absent:
        error = (f"stopped: {stop}" if stop else
                 f"the structure process ended with exit {code} without this record: no stop by name, a held state "
                 "(rules.tiers.defect_hold)")
        c.atomic_write_json(records / record_name(group, token, tokens), {
            "format": FORMAT, "group": group, "mode": a.mode, "token": token, "tokens": tokens,
            "status": "stopped" if stop else "ended_without_record", "exit": code, "stop": stop,
            "in_flight": token == absent[0], "error": error, "refusal": None,
            "toolchain": (checked.get("gates") or {}).get("toolchain"), "launch_order": checked.get("order"),
            "lock": checked.get("lock"), "driver_code_sha256": checked["start"], "wall_s": wall_s,
            "supervisor": {"form": "launched alone (the structure-first form): the packs ran in one child process",
                           "limit_s": limit_s, "child_exit": code}})
    return absent


def supervise(a, group: str, tokens: list[str], checked: dict, started: float) -> int:
    ""
    limit = float(checked["limit_s"])
    previous = {}
    if threading.current_thread() is threading.main_thread():
        previous = {sig: signal.signal(sig, _on_signal) for sig in _SIGNALS}
    signalled = None
    t0 = time.time()
    try:
        try:
            code = run_child(lambda: packs(a, group, tokens, checked["lock"], checked["gates"], checked["order"],
                                           checked["start"]), max(1.0, started + limit - time.time()))
        except _Stopped as stop:
            signalled = stop.signum
            code = 128 + stop.signum
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        written = record_ending(a, group, tokens, code, checked, limit, time.time() - t0)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if written:
        print(json.dumps({"group": group, "child_exit": code, "stop": preflight.stop_name(code), "written": written},
                         default=str), flush=True)
    if signalled is not None:
        return 128 + signalled
    statuses = [rec.get("status") for _, rec in preflight.structure_records(Path(a.records), group)]
    return 0 if code == 0 and len(statuses) == len(tokens) and all(st == "complete" for st in statuses) else 1


def main() -> int:
    started = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--mode", default="record", choices=("record", "measure", "smoke"))
    ap.add_argument("--group-start", action="store_true", help="deploy.py's child at the group's start: the group's "
                                                                "own launch order check, not the structure-first form")
    a = ap.parse_args()
    group = f"{a.member}.{a.bundle}.{a.arm}"
    limit_s = configs.timeouts(a.member)["structure"]
    existing = [name for name, _ in preflight.structure_records(Path(a.records), group)] \
        if Path(a.records).is_dir() else []
    if existing and a.mode != "smoke":
        raise SystemExit(f"REFUSED: {Path(a.records) / existing[0]} exists: a group's structure is computed once")
    start = c.driver_code_sha256()
    lock = lockcheck.verify()
    c.install_data_guard()
    gates = None
    if a.mode != "smoke":
        gates = preflight.gates(a.worktree, a.member, a.bundle, a.arm, [], a.mode)
    order = None
    if a.mode == "record":
        order = preflight.launch_order_check(Path(a.records), a.member, a.bundle, a.arm, a.mode,
                                             structure_first=not a.group_start)
    tokens = configs.structure_tokens(a.member, a.bundle, a.arm)
    if a.group_start:
        return packs(a, group, tokens, lock, gates, order, start)
    return supervise(a, group, tokens, {"lock": lock, "gates": gates, "order": order, "tokens": tokens,
                                        "start": start, "limit_s": limit_s}, started)


def packs(a, group: str, tokens: list[str], lock, gates, order, start) -> int:
    ""
    statuses = []
    for token in tokens:
        t0 = time.time()
        try:
            import torch

            torch.set_num_threads(1)
            a.token = token
            rec = build(a, gates)
            rec.setdefault("token", token)
            rec["lock"] = lock
            rec["status"] = "complete"
        except BaseException as exc:
            rec = {"format": FORMAT, "group": group, "mode": a.mode, "token": token,
                   "status": "refused_or_failed", "error": f"{type(exc).__name__}: {exc}"[:4000],
                   "traceback": traceback.format_exc()[-6000:],
                   "refusal": run_row.refusal_record(exc)}
        if order is not None:
            rec["launch_order"] = order
        rec["tokens"] = tokens
        rec["wall_s"] = time.time() - t0
        rec["driver_code_sha256"] = start
        c.refuse_if_changed(start, "the structural pack")
        c.atomic_write_json(Path(a.records) / record_name(group, token, tokens), rec)
        statuses.append(rec["status"])
        print(json.dumps({k: rec.get(k) for k in ("group", "token", "status", "error", "shape", "schedule",
                                                  "host_census", "wall_s")}, indent=1, default=str)[:4000])
    return 0 if all(st == "complete" for st in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
