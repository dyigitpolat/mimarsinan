#!/usr/bin/env python3
"""The ODIN card power protocol: one allocation, one sampler, one declared plan.

WHAT ONE SESSION DOES (``session``, which ``ODIN_MODE=power`` in
``odin_board.sbatch`` runs on the allocated board node):

  1. refuses a plan whose predicted card-seconds exceed its own budget, before
     touching the card;
  2. resolves the BDF behind the XRT device index and probes both telemetry
     routes (``odin_power_sampler.py probe``) in child processes, so no device
     handle outlives them;
  3. resets the card to its shell (``xrt-smi reset -d <BDF> --force``, the
     cluster's own recipe), reads back which xclbin UUID is loaded, and starts
     the sidecar sampler on the chosen route;
  4. runs the plan's steps in order, each gated by the pilot verdict
     (``when``) and by the remaining budget:
       shell_only  the shell after reset, sampled for ``seconds``;
       pilot       one design's xclbin loaded, kernel idle, for ``seconds``;
                   the pilot gate (GO, REDUCED, NO_GO) is computed right after;
       recertify   the deployment executor with the resident arm, certified;
       protocol    xclbin loaded, then ``rounds`` rounds of interleaved windows
                   of ``block_s`` each, conditions rotated per round:
                   kernel_idle, loop_as_run, loop_resident, and optionally
                   transfer_<arm> (the arm's certified buffer writes and syncs
                   per sample, paced as in the loop, never ap_start);
       drift       the pilot's design reloaded and idle: the drift reference;
  5. stops the sampler, writes session.json, then the estimate
     (``odin_power_estimate.py``), and session.json again.

N COMPUTE UNITS (route D). A design may declare ``compute_units`` (default 1):
its xclbin links N units of the kernel, unit k on its own bank. A protocol step
on such a design loads the xclbin once, then starts one ``unit`` worker process
per unit, one after another; each holds its unit exclusively and runs the same
certified campaigns and controls on its own buffers. Every loop window drives
all N units at once and every transfer window replays all N units' buffer
traffic, never ``ap_start``; the window's record keeps every unit's row
(``per_cu``) beside their sums, with the mean and the summed kernel duty and
the busy share of the whole window (``kernel_duty_window``). The
recertification runs the executor with ``--compute-units N``. A worker answers
on a duplicate of its stdout and points fd 1 at its log, so no library output
can reach the channel; a line that is not a reply is skipped and recorded.
Every child of the session runs in its own process group, and a child killed
on its timeout is killed with its group, so no unit process outlives its step
holding a compute unit.

IDENTITY. Every design step checks the staged xclbin against the plan's
``xclbin_identity`` before it loads it (the file size the archived link wrote)
and after (the UUID in the file's own metadata, once the owner has declared
it): a mismatch refuses the step, unless the plan declares the design
``rebuilt``, in which case every record of it carries that label and the
archived Vivado numbers are not quoted for it. ``preflight`` runs the file
checks and the bundle pins on the head node, with no card.

BUDGET. A step that would cross the budget line is skipped before it starts,
and every child process (identity, probe, reset, design steps) is given at most
the budget left minus the declared teardown, so the session always stops its
sampler and writes its records inside its own budget.

Every design step runs in its own child process (``design``), so each xclbin
load gets a fresh pyxrt and the card is released before the next one. Every
window is bracketed by markers on the host's CLOCK_MONOTONIC, the clock the
sampler stamps its readings with; markers also carry the protocol ``state``
(shell_only, loading, kernel_idle, loop), which is all a FAKE card's synthetic
power plant reads.

STRUCTURE, NOT SILICON, in fake mode (``plan.fake`` or ``--fake-pyxrt``): the
fake pyxrt answers runs from the frozen replay and power from a declared
synthetic plant; the reset is simulated. ``selftest`` runs the parser, the
estimator and two whole fake sessions (one that measures, one whose card has no
sensors) and checks every declared outcome.

    python3 host/odin_power_protocol.py session --plan power/plan_hacc_u55c_tcad.json \\
        --results results/power
    python3 host/odin_power_protocol.py budget --plan power/plan_hacc_u55c_tcad.json
    python3 host/odin_power_protocol.py preflight --plan power/plan_hacc_u55c_tcad.json
    python3 host/odin_power_protocol.py selftest --results results/power_selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import odin_board_driver as driver  # noqa: E402
import odin_deployment_executor as executor  # noqa: E402
import odin_power_estimate as estimator  # noqa: E402
import odin_power_sampler as sampler  # noqa: E402

PACKAGE = os.path.dirname(_HERE)
THIS = os.path.abspath(__file__)
SAMPLER = os.path.join(_HERE, "odin_power_sampler.py")
EXECUTOR = os.path.join(_HERE, "odin_deployment_executor.py")
FAKE = os.path.join(_HERE, "fake_pyxrt_for_selftest.py")

PLAN_SCHEMA = "odin_power_plan/1"
SESSION_SCHEMA = "odin_power_session/1"
STEP_KINDS = ("shell_only", "pilot", "recertify", "protocol", "drift")
DESIGN_KINDS = ("pilot", "protocol", "drift")
WHEN = ("always", "go", "reduced", "go_or_reduced")
CONDITIONS = ("kernel_idle", "loop_as_run", "loop_resident",
              "transfer_as_run", "transfer_resident")
STATE_OF = {"kernel_idle": "kernel_idle", "loop_as_run": "loop",
            "loop_resident": "loop", "transfer_as_run": "transfer",
            "transfer_resident": "transfer", "shell_only": "shell_only"}
LOOP, TRANSFER = "loop_", "transfer_"

#: Declared overheads the budget adds (ESTIMATE: the xclbin load and reset walls
#: on this cluster are unknown until a session measures them, HWA section 6).
DEFAULT_OVERHEADS = {"session_start_s": 60.0, "reset_s": 60.0,
                     "xclbin_load_s": 30.0, "teardown_s": 60.0,
                     "recertify_host_s": 10.0, "unit_start_s": 5.0}

#: How long one unit worker may take to answer a command beyond the command's
#: own seconds (a window ends at the first campaign boundary after its block).
UNIT_REPLY_SLACK_S = 600.0

ZERO_UUID_CHARS = set("0-")

#: A design step that ran to the end but saw a red certificate or readout in a
#: window exits with this code; an uncaught error exits 1 (Python's own), and a
#: typed refusal 2. Only the first two leave a step.json behind.
WINDOWS_FAILED = 4


class OdinPowerRefusal(RuntimeError):
    """The power protocol refused; the message says what and how to fix it."""


# ---------------------------------------------------------------------------
# The plan and its budget
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(plan_dir: str, path: Optional[str]) -> Optional[str]:
    """A plan path: absolute as written, or relative to the plan's own folder."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(plan_dir, path))


def load_plan(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    validate_plan(plan, source=path)
    plan_dir = os.path.dirname(os.path.abspath(path))
    for design in plan["designs"].values():
        for key in ("xclbin", "bundle", "replay"):
            if design.get(key):
                design[key] = resolve(plan_dir, design[key])
    fake = plan.get("fake")
    if fake and fake.get("pyxrt"):
        fake["pyxrt"] = resolve(plan_dir, fake["pyxrt"])
    return plan


def validate_plan(plan: Dict[str, Any], *, source: str) -> None:
    def refuse(message: str) -> NoReturn:
        raise OdinPowerRefusal(f"{source}: {message}")
    if plan.get("schema") != PLAN_SCHEMA:
        refuse(f"schema {plan.get('schema')!r} is not {PLAN_SCHEMA!r}")
    designs = plan.get("designs")
    if not isinstance(designs, dict) or not designs:
        refuse("no designs declared")
    fake = bool((plan.get("fake") or {}).get("pyxrt"))
    for name, design in designs.items():
        for key in ("xclbin", "bundle", "chip"):
            if not design.get(key):
                refuse(f"design {name!r} declares no {key}")
        identity = design.get("xclbin_identity")
        if identity is None and not fake:
            refuse(f"design {name!r} declares no xclbin_identity: a card plan pins "
                   f"each xclbin to the build its numbers describe (archived.size_bytes "
                   f"from the link log, uuid from owner step 3)")
        if identity is not None:
            archived = identity.get("archived") or {}
            if not (archived.get("size_bytes") or identity.get("uuid")
                    or identity.get("sha256")):
                refuse(f"design {name!r}: xclbin_identity pins nothing (declare "
                       f"archived.size_bytes, uuid or sha256)")
            if identity.get("rebuilt") and not identity.get("uuid"):
                refuse(f"design {name!r}: xclbin_identity.rebuilt is true but no uuid "
                       f"is declared; record the rebuilt xclbin's UUID (xclbinutil "
                       f"--info) so the session can tell that rebuild from any other")
        units = design.get("compute_units", 1)
        if isinstance(units, bool) or not isinstance(units, int) or units < 1:
            refuse(f"design {name!r}: compute_units {units!r} must be a positive integer "
                   f"(the units its xclbin links, declared as data; absent means 1)")
        if identity is not None and identity.get("compute_unit_names") is not None:
            # Read from the xclbin itself (xclbinutil --info, its Instance lines):
            # the link's own names must be the ones the host opens, in unit order.
            declared_names = identity["compute_unit_names"]
            expected = driver.compute_unit_names(units)
            if not isinstance(declared_names, list) or \
                    [str(item) for item in declared_names] != expected:
                refuse(f"design {name!r}: xclbin_identity.compute_unit_names "
                       f"{declared_names!r} (the instances the xclbin declares) are not "
                       f"{expected}, the names the host opens for {units} unit(s); the "
                       f"link and the host disagree")
        campaign = design.get("campaign_s", {})
        for arm in executor.ARMS:
            if not isinstance(campaign.get(arm), (int, float)) or campaign[arm] <= 0:
                refuse(f"design {name!r}: campaign_s.{arm} (the predicted wall of "
                       f"one campaign, for the budget) must be a positive number")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        refuse("no steps declared")
    loaded = False
    for index, step in enumerate(steps):
        kind = step.get("kind")
        if kind not in STEP_KINDS:
            refuse(f"step {index}: kind {kind!r} is not one of {STEP_KINDS}")
        if step.get("when", "always") not in WHEN:
            refuse(f"step {index}: when {step.get('when')!r} is not one of {WHEN}")
        if kind == "shell_only":
            if loaded:
                refuse(f"step {index}: a shell_only window after an xclbin was "
                       f"loaded would not be the shell; put it first")
            if float(step.get("seconds", 0)) <= 0:
                refuse(f"step {index}: shell_only needs seconds > 0")
            continue
        loaded = True
        if step.get("design") not in designs:
            refuse(f"step {index}: design {step.get('design')!r} is not declared")
        if kind in ("pilot", "drift") and float(step.get("seconds", 0)) <= 0:
            refuse(f"step {index}: {kind} needs seconds > 0")
        if kind == "protocol":
            conditions = step.get("conditions", [])
            if not conditions or any(c not in CONDITIONS for c in conditions):
                refuse(f"step {index}: conditions {conditions} must be drawn from "
                       f"{CONDITIONS}")
            if "kernel_idle" not in conditions or len(conditions) < 2:
                refuse(f"step {index}: a protocol step needs kernel_idle and at "
                       f"least one loop condition (the differential is loop "
                       f"minus idle)")
            for condition in conditions:
                if condition.startswith(TRANSFER) and \
                        LOOP + condition[len(TRANSFER):] not in conditions:
                    refuse(f"step {index}: {condition} replays the certified warm-up "
                           f"of {LOOP + condition[len(TRANSFER):]}, which the step "
                           f"does not run")
            if int(step.get("rounds", 0)) < 1 or float(step.get("block_s", 0)) <= 0:
                refuse(f"step {index}: rounds >= 1 and block_s > 0 are required")
        if kind == "recertify" and step.get("arm", executor.ARM_RESIDENT) not in executor.ARMS:
            refuse(f"step {index}: arm {step.get('arm')!r} is not one of {executor.ARMS}")
    if float(plan.get("budget_s", 0)) <= 0:
        refuse("budget_s (the card-seconds this plan may spend) must be declared")


def compute_units_of(design: Dict[str, Any]) -> int:
    """The compute units a design's xclbin links (1 unless the plan says more)."""
    return int(design.get("compute_units", 1))


def aggregate_units(rows: Sequence[Dict[str, Any]], names: Sequence[str]) -> Dict[str, Any]:
    """One window of N units (the executor's own rule: sums, duty, every unit)."""
    return executor.aggregate_units(rows, names)


def overheads_of(plan: Dict[str, Any]) -> Dict[str, float]:
    merged = dict(DEFAULT_OVERHEADS)
    merged.update({k: float(v) for k, v in plan.get("overheads", {}).items()})
    return merged


def step_conditions(step: Dict[str, Any]) -> List[str]:
    return list(step.get("conditions") or ["kernel_idle"])


def step_rounds(step: Dict[str, Any]) -> int:
    return int(step.get("rounds", 1)) if step["kind"] == "protocol" else 1


def step_block_s(step: Dict[str, Any]) -> float:
    return float(step["block_s"] if step["kind"] == "protocol" else step["seconds"])


def predict_step(plan: Dict[str, Any], step: Dict[str, Any]) -> float:
    """Predicted card-seconds of one step (ESTIMATE, from the plan's own walls)."""
    over = overheads_of(plan)
    kind = step["kind"]
    if kind == "shell_only":
        return float(step["seconds"])
    design = plan["designs"][step["design"]]
    campaign = design["campaign_s"]
    if kind in ("pilot", "drift"):
        return over["xclbin_load_s"] + float(step["seconds"])
    units = compute_units_of(design)
    # N > 1: the units start one after another before anything runs.
    starts = over["unit_start_s"] * units if units > 1 else 0.0
    if kind == "recertify":
        arm = step.get("arm", executor.ARM_RESIDENT)
        return (over["xclbin_load_s"] + float(campaign[arm]) + over["recertify_host_s"]
                + starts)
    arms = [c[len(LOOP):] for c in step_conditions(step) if c.startswith(LOOP)]
    warmups = sum(float(campaign[arm]) for arm in arms)
    per_round = 0.0
    for condition in step_conditions(step):
        per_round += float(step["block_s"])
        if condition.startswith(LOOP):
            # A loop stops only at a campaign boundary: at worst one more campaign.
            per_round += float(campaign[condition[len(LOOP):]])
        elif condition.startswith(TRANSFER):
            # A control stops at the first pass boundary: at worst one more
            # sample's wall (every pass of one sample), and it needs no warm-up
            # of its own (it replays its arm's certified warm-up).
            per_round += (float(campaign[condition[len(TRANSFER):]])
                          / max(1, int(design.get("samples", 64))))
    return over["xclbin_load_s"] + starts + warmups + step_rounds(step) * per_round


def budget(plan: Dict[str, Any]) -> Dict[str, Any]:
    """The worst-case path through the plan against its declared budget."""
    over = overheads_of(plan)
    rows = []
    paths = {"always": 0.0, "go": 0.0, "reduced": 0.0}
    for index, step in enumerate(plan["steps"]):
        seconds = predict_step(plan, step)
        if step["kind"] == "shell_only" and plan.get("reset_at_start", True):
            seconds += 0.0  # the reset is booked once in the fixed part below
        when = step.get("when", "always")
        rows.append({"index": index, "kind": step["kind"], "design": step.get("design"),
                     "when": when, "label": step.get("label"), "predicted_s": seconds})
        if when == "always":
            paths["always"] += seconds
        if when in ("go", "go_or_reduced"):
            paths["go"] += seconds
        if when in ("reduced", "go_or_reduced"):
            paths["reduced"] += seconds
    fixed = over["session_start_s"] + over["teardown_s"] + (
        over["reset_s"] if plan.get("reset_at_start", True) else 0.0)
    worst = fixed + paths["always"] + max(paths["go"], paths["reduced"])
    return {"steps": rows, "fixed_s": fixed, "paths_s": paths,
            "go_path_s": fixed + paths["always"] + paths["go"],
            "reduced_path_s": fixed + paths["always"] + paths["reduced"],
            "worst_path_s": worst, "budget_s": float(plan["budget_s"]),
            "fits": worst <= float(plan["budget_s"]), "overheads": over}


# ---------------------------------------------------------------------------
# Which xclbin: the archived build, a declared rebuild, or a refusal
# ---------------------------------------------------------------------------


def normalize_uuid(text: Optional[str]) -> Optional[str]:
    """A UUID compared as XRT prints it, without case, dashes or braces."""
    if text is None:
        return None
    return "".join(ch for ch in str(text).lower() if ch not in "-{} ")


def xclbin_identity(design: Dict[str, Any], *, metadata_uuid: Optional[str] = None,
                    check_file: bool = True) -> Dict[str, Any]:
    """The staged xclbin against the plan's pins: what was checked, what differs.

    Before the load only the file is checked (its size against the archived
    link's "Successfully wrote (N bytes)", its sha256 when declared); after the
    load ``metadata_uuid`` (the UUID in the file's own metadata) is checked
    against the declared ``uuid``. The archived pins describe the archived
    build, so a design declared ``rebuilt`` is held to its own UUID and, when
    declared, its own sha256 only.
    ``check_file=False`` (a fake session, which has no file) checks the UUID only.
    """
    declared = design.get("xclbin_identity") or {}
    archived = declared.get("archived") or {}
    path = design["xclbin"]
    rebuilt = bool(declared.get("rebuilt"))
    out: Dict[str, Any] = {"xclbin": path, "declared": declared, "checks": {},
                           "problems": [], "uuid_checked": False}
    if check_file and not os.path.isfile(path):
        out["problems"].append(f"{path} does not exist")
        out["status"] = "missing"
        return out
    size = os.path.getsize(path) if check_file else None
    if check_file:
        out["checks"]["size_bytes"] = size
    if check_file and not rebuilt and archived.get("size_bytes") is not None \
            and size != int(archived["size_bytes"]):
        out["problems"].append(
            f"{path} is {size} bytes, but the archived link wrote "
            f"{int(archived['size_bytes'])} bytes ({archived.get('written', 'archived build')})")
    # The sha256 pins the file itself, whichever build made it: a rebuilt design's
    # declared sha256 is its rebuild's own (route D fills it from the new build).
    if check_file and declared.get("sha256"):
        digest = sha256_file(path)
        out["checks"]["sha256"] = digest
        if digest != str(declared["sha256"]).replace("sha256:", ""):
            out["problems"].append(f"{path} hashes to {digest}, not the declared sha256")
    if metadata_uuid is not None:
        out["checks"]["metadata_uuid"] = metadata_uuid
        if declared.get("uuid"):
            out["uuid_checked"] = True
            if normalize_uuid(metadata_uuid) != normalize_uuid(declared["uuid"]):
                out["problems"].append(
                    f"{path} carries UUID {metadata_uuid}, but the plan declares "
                    f"{declared['uuid']}")
    if out["problems"]:
        out["status"] = "mismatch"
    elif rebuilt:
        out["status"] = "rebuilt"
        out["note"] = ("declared rebuilt: the archived report_power does not describe "
                       "this placement; re-extract report_power from the new build "
                       "before quoting the interim row or the gate's expected differential")
    else:
        out["status"] = "archived"
        if not declared.get("uuid"):
            out["note"] = ("size only: no UUID is declared (owner step 3 records it "
                           "from xclbinutil --info)")
    return out


def require_xclbin_identity(design: Dict[str, Any], *, name: str,
                            metadata_uuid: Optional[str] = None,
                            check_file: bool = True) -> Dict[str, Any]:
    """``xclbin_identity``, refusing a mismatch by name, with the way out."""
    result = xclbin_identity(design, metadata_uuid=metadata_uuid, check_file=check_file)
    if result["problems"]:
        raise OdinPowerRefusal(
            f"design {name}: {'; '.join(result['problems'])}. This is not the "
            f"xclbin the plan's numbers describe. Restore the archived file, or "
            f"declare xclbin_identity.rebuilt = true with the new build's uuid "
            f"(xclbinutil --info) and re-extract report_power from it")
    return result


def uuid_command(xclbin: str) -> str:
    return f"xclbinutil --info --input {xclbin} | grep -i -E 'uuid|created|date'"


def preflight(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Owner step 3 on the head node, no card: every staged file against its pins."""
    report: Dict[str, Any] = {"designs": {}, "ok": True}
    for name, design in sorted(plan["designs"].items()):
        problems: List[str] = []
        identity = xclbin_identity(design)
        problems += identity["problems"]
        bundle: Dict[str, Any] = {"path": design["bundle"]}
        if not os.path.isfile(design["bundle"]):
            problems.append(f"{design['bundle']} does not exist")
        else:
            try:
                document = executor.load_bundle(design["bundle"])
                bundle["self_hash"] = document["self_hash"]
                pinned = design.get("bundle_self_hash")
                if pinned and document["self_hash"] != pinned:
                    problems.append(f"{design['bundle']} hashes to {document['self_hash']}, "
                                    f"but the plan pins {pinned}")
            except (driver.OdinDriverError, ValueError, KeyError) as exc:
                problems.append(f"{design['bundle']}: {type(exc).__name__}: {exc}")
        declared = design.get("xclbin_identity") or {}
        report["designs"][name] = {
            "ok": not problems, "problems": problems, "xclbin": identity,
            "bundle": bundle, "uuid_declared": declared.get("uuid"),
            "rebuilt": bool(declared.get("rebuilt")),
            "uuid_command": uuid_command(design["xclbin"]),
            "archived": declared.get("archived"),
        }
        report["ok"] = report["ok"] and not problems
    return report


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def write_marker(path: str, **fields: Any) -> Dict[str, Any]:
    record = {"t_mono": time.monotonic(),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    record.update(fields)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
    return record


def rotate(conditions: Sequence[str], offset: int) -> List[str]:
    """Round r runs the conditions starting at r mod k: a cyclic rotation."""
    k = len(conditions)
    shift = offset % k
    return list(conditions[shift:]) + list(conditions[:shift])


# ---------------------------------------------------------------------------
# One design step, in its own process
# ---------------------------------------------------------------------------


def _compact(stats: Dict[str, Any]) -> Dict[str, Any]:
    renamed = dict(stats)
    for key in ("t_start", "t_end"):
        if key in renamed:
            renamed["loop_" + key] = renamed.pop(key)
    if isinstance(renamed.get("per_cu"), list):
        renamed["per_cu"] = [_compact(row) for row in renamed["per_cu"]]
    return renamed


def _design_document(design: Dict[str, Any]) -> Tuple[Dict[str, Any], List[int]]:
    """The design's sealed bundle, checked against its pin, and its samples."""
    document = executor.load_bundle(design["bundle"])
    executor.require_protocol_agrees(document, source=design["bundle"])
    executor.require_pass_order_is_causal(document)
    pinned = design.get("bundle_self_hash")
    if pinned and document["self_hash"] != pinned:
        raise OdinPowerRefusal(
            f"{design['bundle']} hashes to {document['self_hash']}, but the plan "
            f"pins {pinned}: this is not the bundle the archived campaign ran")
    return document, executor.sample_indices(document, int(design.get("samples", 64)))


def _session_options(plan: Dict[str, Any], design: Dict[str, Any],
                     fake_pyxrt: Optional[str], cu: Optional[str] = None) -> SimpleNamespace:
    return SimpleNamespace(
        xclbin=design["xclbin"], device_index=int(plan.get("device_index", 0)),
        capture_events=driver.DEFAULT_CAPTURE_EVENTS, declare_cores=0,
        capture_ram_events=0, chip=design["chip"],
        run_timeout_ms=driver.BLOCK_UNTIL_DONE_MS, fake_pyxrt=fake_pyxrt, cu=cu)


def run_design_step(plan: Dict[str, Any], index: int, *, markers: str,
                    results: str, fake_pyxrt: Optional[str],
                    plan_path: Optional[str] = None) -> int:
    step = plan["steps"][index]
    name = step["design"]
    design = plan["designs"][name]
    kind = step["kind"]
    if kind == "protocol" and compute_units_of(design) > 1:
        if not plan_path:
            raise OdinPowerRefusal("an N-unit protocol step needs the plan's path for "
                                   "its unit workers")
        return run_units_step(plan, index, plan_path=plan_path, markers=markers,
                              results=results, fake_pyxrt=fake_pyxrt)
    os.makedirs(results, exist_ok=True)
    document, samples = _design_document(design)
    options = _session_options(plan, design, fake_pyxrt)
    if not fake_pyxrt:
        # Before the load: the file itself (size, sha256) against the plan's pins.
        require_xclbin_identity(design, name=name)
    xclbin_sha = sha256_file(design["xclbin"])
    write_marker(markers, event="xclbin_load", step=index, kind=kind, design=name,
                 state="loading", xclbin=design["xclbin"], xclbin_sha256=xclbin_sha)
    session = driver.open_session(options)
    result: Dict[str, Any] = {"step": index, "kind": kind, "design": name,
                              "xclbin": design["xclbin"], "xclbin_sha256": xclbin_sha,
                              "bundle": design["bundle"],
                              "bundle_self_hash": document["self_hash"],
                              "samples": len(samples), "windows": [], "warmups": {}}
    try:
        # After the load: the UUID in the file's own metadata against the plan's.
        identity = require_xclbin_identity(
            design, name=name, metadata_uuid=session.identity.get("xclbin_uuid_metadata"),
            check_file=not fake_pyxrt)
        result["xclbin_identity"] = identity
        executor.require_fabric_matches(document, session.capacity, source=design["bundle"])
        replay_path = design.get("replay")
        if replay_path and fake_pyxrt:
            with open(replay_path, "r", encoding="utf-8") as handle:
                replay = json.load(handle)
            executor.verify_replay(replay, document, source=replay_path)
            session.arm_fake(deployment=replay)
        result["identity"] = session.identity
        result["device"] = {k: v for k, v in session.device_stamp().items()
                            if k in ("device_index", "kernel", "cu_access_mode", "capacity")}
        write_marker(markers, event="xclbin_loaded", step=index, kind=kind, design=name,
                     state="kernel_idle", identity=session.identity,
                     xclbin_sha256=xclbin_sha, xclbin_status=identity["status"],
                     xclbin_uuid_checked=identity["uuid_checked"])
        print(f"[power] step {index} {kind} {name}: xclbin loaded in "
              f"{session.identity.get('xclbin_load_s', 0.0):.3f} s, uuid "
              f"{session.identity.get('xclbin_uuid')}, bdf {session.identity.get('bdf')}, "
              f"identity {identity['status']}"
              + (f" ({identity['note']})" if identity.get("note") else ""))
        conditions = step_conditions(step)
        campaigns: Dict[str, Any] = {}
        for condition in conditions:
            if condition.startswith(LOOP):
                arm = condition[len(LOOP):]
                campaign = executor.Campaign(document, session, samples=samples, arm=arm)
                campaign.quiet = True
                # A control of this arm replays the warm-up's transfers.
                campaign.record_transfers = TRANSFER + arm in conditions
                campaigns[arm] = campaign
        controls: Dict[str, Any] = {}
        for arm, campaign in campaigns.items():
            # One whole certified campaign per arm before any measured window:
            # buffers allocated, caches warm, and the arm proven on this card.
            warm = executor.timed_loop(campaign, 0.0)
            result["warmups"][arm] = warm
            write_marker(markers, event="warmup", step=index, design=name,
                         **_compact(warm))
            print(f"[power] warm-up {arm}: {warm['inferences']} inference(s) in "
                  f"{warm['wall_s']:.3f} s, passed={warm['passed']}")
            if campaign.record_transfers:
                controls[arm] = executor.TransferControl(campaign)
                campaign.record_transfers = False
        block = step_block_s(step)
        for round_number in range(step_rounds(step)):
            for condition in rotate(conditions, round_number):
                window_id = f"s{index}-r{round_number}-{condition}"
                write_marker(markers, event="window_start", window_id=window_id,
                             step=index, kind=kind, design=name, condition=condition,
                             round=round_number, state=STATE_OF[condition])
                started = time.monotonic()
                if condition == "kernel_idle":
                    deadline = started + block
                    while time.monotonic() < deadline:
                        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
                    stats: Dict[str, Any] = {"passed": True}
                elif condition.startswith(TRANSFER):
                    stats = _compact(executor.transfer_loop(
                        controls[condition[len(TRANSFER):]], block))
                else:
                    stats = _compact(executor.timed_loop(
                        campaigns[condition[len(LOOP):]], block))
                end = write_marker(markers, event="window_end", window_id=window_id,
                                   state="kernel_idle", **stats)
                row = {"window_id": window_id, "condition": condition,
                       "round": round_number, "seconds": end["t_mono"] - started, **stats}
                result["windows"].append(row)
                if "control" in stats:
                    summary = (f"control: {stats['passes']} pass(es) of transfers, "
                               f"{stats['ap_starts']} ap_start, "
                               f"{stats['bytes_to_device']} B to and "
                               f"{stats['bytes_from_device']} B from the card")
                elif "inferences" in stats:
                    summary = (f"{stats['inferences']} inference(s), duty "
                               f"{stats['kernel_duty']:.3f}, passed={stats['passed']}")
                else:
                    summary = "idle"
                print(f"[power] {window_id}: {row['seconds']:.2f} s, {summary}")
    finally:
        session.close()
    driver.write_result(results, "step.json", result)
    failed = [w for w in result["windows"] if not w.get("passed", True)]
    failed += [w for w in result["warmups"].values() if not w.get("passed", True)]
    return WINDOWS_FAILED if failed else 0


def _pump(stream: Any, lines: "queue.Queue[Optional[str]]") -> None:
    """Every line a worker writes on its channel, then None at its end."""
    try:
        for line in stream:
            lines.put(line)
    except (OSError, ValueError):
        pass
    finally:
        lines.put(None)


class UnitWorkers:
    """N ``unit`` worker processes, one per compute unit, spoken to in JSON lines.

    Each worker's channel is read by its own thread into a queue, so a reply is
    waited for with a deadline and a line that is not a reply (none should
    reach the channel, see ``open_unit_channel``) is skipped and recorded in
    ``stray`` instead of ending the step.
    """

    def __init__(self, *, plan_path: str, index: int, units: int, results: str,
                 fake_pyxrt: Optional[str]) -> None:
        self.names = driver.compute_unit_names(units)
        self.procs: List["subprocess.Popen[str]"] = []
        self.lines: List["queue.Queue[Optional[str]]"] = []
        self.logs: List[Any] = []
        self.stray: List[Dict[str, str]] = []
        self.plan_path, self.index, self.results = plan_path, index, results
        self.fake_args = ["--fake-pyxrt", fake_pyxrt] if fake_pyxrt else []

    def argv_for(self, cu_index: int) -> List[str]:
        """The command line of unit ``cu_index``'s worker."""
        return [sys.executable, THIS, "unit", "--plan", self.plan_path, "--step",
                str(self.index), "--cu-index", str(cu_index)] + self.fake_args

    def start(self, timeout: float) -> List[Dict[str, Any]]:
        """Start the units one after another: the next once this one holds its unit."""
        ready = []
        for cu_index, name in enumerate(self.names):
            log = open(os.path.join(self.results, f"unit_{name}.log"), "a", encoding="utf-8")
            self.logs.append(log)
            proc = subprocess.Popen(
                self.argv_for(cu_index), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=log, cwd=PACKAGE, universal_newlines=True, bufsize=1)
            self.procs.append(proc)
            lines: "queue.Queue[Optional[str]]" = queue.Queue()
            self.lines.append(lines)
            threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()
            ready.append(self._reply(cu_index, timeout))
        return ready

    def _reply(self, cu_index: int, timeout: float) -> Dict[str, Any]:
        name, proc = self.names[cu_index], self.procs[cu_index]
        deadline = time.monotonic() + max(1.0, float(timeout))
        while True:
            left = deadline - time.monotonic()
            try:
                line = self.lines[cu_index].get(timeout=left) if left > 0 else \
                    self.lines[cu_index].get_nowait()
            except queue.Empty:
                raise OdinPowerRefusal(
                    f"unit {name} gave no answer within {timeout:.0f} s (exit "
                    f"{proc.poll()}); see unit_{name}.log") from None
            if line is None:
                raise OdinPowerRefusal(
                    f"unit {name} closed its channel without an answer (exit "
                    f"{proc.poll()}); see unit_{name}.log")
            text = line.strip()
            if not text:
                continue
            try:
                reply = json.loads(text)
            except ValueError:
                reply = None
            if not isinstance(reply, dict):
                self.stray.append({"compute_unit": name, "line": text[:500]})
                print(f"[power] unit {name}: skipped a line that is not a reply: "
                      f"{text[:200]!r}")
                continue
            if reply.get("error"):
                raise OdinPowerRefusal(f"unit {name}: {reply['error']}")
            return reply

    def broadcast(self, command: Dict[str, Any], timeout: float) -> List[Dict[str, Any]]:
        """The same command to every unit at once, then every answer in unit order."""
        text = json.dumps(command) + "\n"
        for proc in self.procs:
            assert proc.stdin is not None
            proc.stdin.write(text)
            proc.stdin.flush()
        return [self._reply(k, timeout) for k in range(len(self.procs))]

    def stop(self) -> None:
        for proc in self.procs:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass
        for proc in self.procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in self.logs:
            log.close()


def run_units_step(plan: Dict[str, Any], index: int, *, plan_path: str, markers: str,
                   results: str, fake_pyxrt: Optional[str]) -> int:
    """A protocol step on an N-unit design: every window drives all N units at once."""
    step = plan["steps"][index]
    name = step["design"]
    design = plan["designs"][name]
    kind = step["kind"]
    units = compute_units_of(design)
    os.makedirs(results, exist_ok=True)
    document, samples = _design_document(design)
    options = _session_options(plan, design, fake_pyxrt)
    if not fake_pyxrt:
        require_xclbin_identity(design, name=name)
    xclbin_sha = sha256_file(design["xclbin"])
    write_marker(markers, event="xclbin_load", step=index, kind=kind, design=name,
                 state="loading", xclbin=design["xclbin"], xclbin_sha256=xclbin_sha)
    # ONE load, holding no unit, before any unit worker opens its own.
    session = driver.open_session(options, open_kernel=False)
    try:
        identity = require_xclbin_identity(
            design, name=name, metadata_uuid=session.identity.get("xclbin_uuid_metadata"),
            check_file=not fake_pyxrt)
        executor.require_fabric_matches(document, session.capacity, source=design["bundle"])
        load_identity = session.identity
        device = {k: v for k, v in session.device_stamp().items()
                  if k in ("device_index", "kernel", "cu_access_mode", "capacity")}
    finally:
        session.close()
    result: Dict[str, Any] = {"step": index, "kind": kind, "design": name,
                              "xclbin": design["xclbin"], "xclbin_sha256": xclbin_sha,
                              "bundle": design["bundle"],
                              "bundle_self_hash": document["self_hash"],
                              "samples": len(samples), "compute_units": units,
                              "windows": [], "warmups": {}, "xclbin_identity": identity,
                              "identity": load_identity, "device": device}
    over = overheads_of(plan)
    workers = UnitWorkers(plan_path=plan_path, index=index, units=units, results=results,
                          fake_pyxrt=fake_pyxrt)
    try:
        ready = workers.start(timeout=over["unit_start_s"] * 12 + 60.0)
        result["units"] = [{"compute_unit": r["compute_unit"], "identity": r["identity"]}
                           for r in ready]
        write_marker(markers, event="xclbin_loaded", step=index, kind=kind, design=name,
                     state="kernel_idle", identity=load_identity, xclbin_sha256=xclbin_sha,
                     xclbin_status=identity["status"],
                     xclbin_uuid_checked=identity["uuid_checked"], compute_units=units,
                     units=[r["compute_unit"] for r in ready])
        print(f"[power] step {index} {kind} {name}: xclbin loaded in "
              f"{load_identity.get('xclbin_load_s', 0.0):.3f} s, uuid "
              f"{load_identity.get('xclbin_uuid')}, bdf {load_identity.get('bdf')}, "
              f"identity {identity['status']}, {units} units held by their workers")
        conditions = step_conditions(step)
        arms = [c[len(LOOP):] for c in conditions if c.startswith(LOOP)]
        campaign_wall = max([float(design["campaign_s"][arm]) for arm in arms] or [0.0])
        replies = workers.broadcast({"cmd": "warmup"},
                                    timeout=len(arms) * campaign_wall * 4 + UNIT_REPLY_SLACK_S)
        for arm in arms:
            warm = aggregate_units([r["warmups"][arm] for r in replies], workers.names)
            result["warmups"][arm] = warm
            write_marker(markers, event="warmup", step=index, design=name, **_compact(warm))
            print(f"[power] warm-up {arm}: {warm['inferences']} inference(s) over {units} "
                  f"units in {warm['wall_s']:.3f} s, passed={warm['passed']}")
        block = step_block_s(step)
        for round_number in range(step_rounds(step)):
            for condition in rotate(conditions, round_number):
                window_id = f"s{index}-r{round_number}-{condition}"
                opened = write_marker(markers, event="window_start", window_id=window_id,
                                      step=index, kind=kind, design=name,
                                      condition=condition, round=round_number,
                                      state=STATE_OF[condition], compute_units=units)
                started = time.monotonic()
                if condition == "kernel_idle":
                    deadline = started + block
                    while time.monotonic() < deadline:
                        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
                    stats: Dict[str, Any] = {"passed": True}
                else:
                    replies = workers.broadcast(
                        {"cmd": "window", "condition": condition, "seconds": block},
                        timeout=block + campaign_wall * 4 + UNIT_REPLY_SLACK_S)
                    stats = _compact(aggregate_units([r["stats"] for r in replies],
                                                     workers.names))
                    if condition.startswith(LOOP):
                        # The busy share of the whole power window (from its start
                        # marker to now): units that finished early idle in it.
                        window_s = time.monotonic() - float(opened["t_mono"])
                        busy = (float(stats.get("run_s", 0.0))
                                + float(stats.get("programming_run_s", 0.0)))
                        stats["kernel_duty_window"] = (busy / (units * window_s)
                                                       if window_s > 0 else None)
                end = write_marker(markers, event="window_end", window_id=window_id,
                                   state="kernel_idle", **stats)
                row = {"window_id": window_id, "condition": condition,
                       "round": round_number, "seconds": end["t_mono"] - started, **stats}
                result["windows"].append(row)
                if "control" in stats:
                    summary = (f"control over {units} units: {stats['passes']} pass(es) of "
                               f"transfers, {stats['ap_starts']} ap_start, "
                               f"{stats['bytes_to_device']} B to and "
                               f"{stats['bytes_from_device']} B from the card")
                elif "inferences" in stats:
                    summary = (f"{stats['inferences']} inference(s) over {units} units, mean "
                               f"duty {stats['kernel_duty']:.3f}, passed={stats['passed']}")
                else:
                    summary = "idle"
                print(f"[power] {window_id}: {row['seconds']:.2f} s, {summary}")
        closing = workers.broadcast({"cmd": "close"}, timeout=UNIT_REPLY_SLACK_S)
        result["unit_devices"] = [r.get("device") for r in closing]
    finally:
        workers.stop()
        if workers.stray:
            result["unit_stray_lines"] = list(workers.stray)
    driver.write_result(results, "step.json", result)
    failed = [w for w in result["windows"] if not w.get("passed", True)]
    failed += [w for w in result["warmups"].values() if not w.get("passed", True)]
    return WINDOWS_FAILED if failed else 0


def open_unit_channel() -> Any:
    """A unit worker's reply channel: a duplicate of its stdout, opened before fd 1
    is pointed at stderr (the unit's log), so neither a print nor a write to fd 1
    from C code (XRT, the binding) can reach the parent's pipe."""
    channel = os.fdopen(os.dup(1), "w", buffering=1)
    sys.stdout.flush()
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return channel


def run_unit_worker(plan_path: str, index: int, cu_index: int,
                    fake_pyxrt: Optional[str]) -> int:
    """One unit of an N-unit protocol step: its session, its campaigns, its windows.

    Answers on its own stdout, one JSON line per command; everything it prints
    goes to stderr (the unit's log), so nothing else can reach the channel. A
    closed stdin (the step's process ended, or was killed) ends the worker.
    """
    channel = open_unit_channel()

    def reply(payload: Dict[str, Any]) -> None:
        channel.write(json.dumps(payload, default=str) + "\n")
        channel.flush()

    try:
        plan = load_plan(plan_path)
        step = plan["steps"][index]
        design = plan["designs"][step["design"]]
        fake = fake_pyxrt or (plan.get("fake") or {}).get("pyxrt")
        document, samples = _design_document(design)
        name = driver.compute_unit_name(cu_index)
        session = driver.open_session(_session_options(plan, design, fake, cu=name))
    except Exception as exc:
        reply({"error": f"{type(exc).__name__}: {exc}"})
        return 2
    try:
        replay_path = design.get("replay")
        if replay_path and fake:
            with open(replay_path, "r", encoding="utf-8") as handle:
                replay = json.load(handle)
            executor.verify_replay(replay, document, source=replay_path)
            session.arm_fake(deployment=replay)
        conditions = step_conditions(step)
        campaigns: Dict[str, Any] = {}
        for condition in conditions:
            if condition.startswith(LOOP):
                arm = condition[len(LOOP):]
                campaign = executor.Campaign(document, session, samples=samples, arm=arm)
                campaign.quiet = True
                campaign.record_transfers = TRANSFER + arm in conditions
                campaigns[arm] = campaign
        controls: Dict[str, Any] = {}
        reply({"ready": True, "compute_unit": name, "identity": session.identity})
        for line in sys.stdin:
            if not line.strip():
                continue
            command = json.loads(line)
            if command["cmd"] == "warmup":
                warmups = {}
                for arm, campaign in campaigns.items():
                    warmups[arm] = executor.timed_loop(campaign, 0.0)
                    if campaign.record_transfers:
                        controls[arm] = executor.TransferControl(campaign)
                        campaign.record_transfers = False
                reply({"warmups": warmups})
            elif command["cmd"] == "window":
                condition, seconds = command["condition"], float(command["seconds"])
                if condition.startswith(TRANSFER):
                    stats = executor.transfer_loop(controls[condition[len(TRANSFER):]], seconds)
                else:
                    stats = executor.timed_loop(campaigns[condition[len(LOOP):]], seconds)
                reply({"stats": stats})
            elif command["cmd"] == "close":
                reply({"device": session.device_stamp()})
                break
            else:
                reply({"error": f"unknown command {command['cmd']!r}"})
    except Exception as exc:
        reply({"error": f"{type(exc).__name__}: {exc}"})
        return 2
    finally:
        session.close()
    return 0


def run_identity(*, device_index: int, fake_pyxrt: Optional[str],
                 fake_plant: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The card behind a device index and the xclbin it holds right now."""
    xrt = driver.load_pyxrt(fake_pyxrt)
    if fake_pyxrt and fake_plant:
        # The fake card answers from the protocol state, so a simulated reset
        # reads back the zero UUID a real one should.
        xrt.arm(plant=fake_plant)
    device = xrt.device(int(device_index))
    out: Dict[str, Any] = {"device_index": int(device_index), "errors": []}
    for field, read in (("bdf", lambda: device.get_info(xrt.xrt_info_device.bdf)),
                        ("name", lambda: device.get_info(xrt.xrt_info_device.name)),
                        ("loaded_xclbin_uuid", lambda: device.get_xclbin_uuid().to_string())):
        try:
            out[field] = str(read())
        except Exception as exc:
            out[field] = None
            out["errors"].append(f"{field}: {type(exc).__name__}: {exc}")
    return out


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


def _kill_group(proc: "subprocess.Popen[Any]") -> None:
    """SIGKILL a child's whole process group (it leads its own), then reap it."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait()


def run_child(argv: List[str], *, log: str, timeout: float) -> Tuple[int, str]:
    """A child process with its own log, in its own process group.

    A design step and an N-unit recertification start one process per compute
    unit; those inherit the child's group, so a child killed on its timeout is
    killed with every unit process it started, and none of them keeps holding its
    compute unit while the next step loads its xclbin.
    """
    with open(log, "a", encoding="utf-8") as handle:
        proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,
                                cwd=PACKAGE, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return -9, (f"timed out after {timeout:.0f} s and was killed with its "
                        f"process group")
    return code, ""


class Session:
    """One allocation: the plan's steps, gated, budgeted and sampled."""

    def __init__(self, plan_path: str, results: str, *,
                 fake_pyxrt: Optional[str] = None) -> None:
        self.plan_path = os.path.abspath(plan_path)
        self.plan = load_plan(self.plan_path)
        self.plan_sha256 = sha256_file(self.plan_path)
        self.results = os.path.abspath(results)
        os.makedirs(self.results, exist_ok=True)
        fake = self.plan.get("fake") or {}
        self.fake_pyxrt = fake_pyxrt or fake.get("pyxrt")
        self.fake = bool(self.fake_pyxrt)
        self.markers = os.path.join(self.results, "markers.jsonl")
        self.samples = os.path.join(self.results, "samples.jsonl")
        self.stop_file = os.path.join(self.results, "sampler.stop")
        self.started = time.monotonic()
        self.device_index = int(self.plan.get("device_index", 0))
        self.bdf: Optional[str] = None
        self.route: Optional[str] = None
        self.sampler_proc: Optional[subprocess.Popen] = None
        self.sampler_log: Any = None
        self.verdict: Optional[str] = None
        self.pilot_result: Optional[Dict[str, Any]] = None
        self.shell_verified: Optional[bool] = None
        self.record: Dict[str, Any] = {
            "schema": SESSION_SCHEMA, "plan": self.plan_path,
            "plan_sha256": self.plan_sha256, "plan_name": self.plan.get("name"),
            "host": sampler.host_stamp(), "fake": self.fake, "steps": [],
        }

    # -- helpers -----------------------------------------------------------

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def say(self, text: str) -> None:
        print(f"[power] t+{self.elapsed():7.1f}s {text}", flush=True)

    def fake_args(self) -> List[str]:
        return ["--fake-pyxrt", self.fake_pyxrt] if self.fake_pyxrt else []

    def plant(self) -> Optional[Dict[str, Any]]:
        plant = (self.plan.get("fake") or {}).get("plant")
        if not (self.fake and plant):
            return None
        return dict(plant, markers=self.markers)

    def teardown_s(self) -> float:
        return overheads_of(self.plan)["teardown_s"]

    def child_timeout(self, wanted: float) -> float:
        """``wanted``, but never past the budget line minus the teardown.

        Whatever a child does, the session keeps the declared teardown to stop
        its sampler and write its records inside its own budget (never less
        than one second, so a child is always given a chance to answer).
        """
        left = float(self.plan["budget_s"]) - self.elapsed() - self.teardown_s()
        return max(1.0, min(float(wanted), left))

    def stop_waits(self) -> Tuple[float, float]:
        """(graceful, after SIGTERM) seconds stop_sampler may wait: 3/4 of the teardown."""
        interval = float(self.plan.get("sampler", {}).get("interval_s", 0.25))
        teardown = self.teardown_s()
        return min(interval * 4 + 60.0, 0.5 * teardown), min(30.0, 0.25 * teardown)

    def child(self, argv: List[str], *, log: str, timeout: float) -> Tuple[int, str]:
        """A child process with its own log, killed with its process group on timeout."""
        return run_child(argv, log=log, timeout=timeout)

    # -- phases ------------------------------------------------------------

    def identify(self, tag: str) -> Dict[str, Any]:
        out_path = os.path.join(self.results, f"identity_{tag}.json")
        argv = [sys.executable, THIS, "identity", "--device-index", str(self.device_index),
                "--out", out_path] + self.fake_args()
        plant = self.plant()
        if plant:
            argv += ["--fake-plant", json.dumps(plant)]
        code, note = self.child(argv, log=os.path.join(self.results, "identity.log"),
                                timeout=self.child_timeout(120))
        if code != 0 or not os.path.isfile(out_path):
            return {"error": f"identity child exited {code} {note}".strip()}
        with open(out_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def probe(self) -> Dict[str, Any]:
        config = self.plan.get("sampler", {})
        routes = config.get("routes", ["xrt-smi", "pyxrt"])
        if self.fake:
            routes = [r for r in routes if r == "pyxrt"] or ["pyxrt"]
        out_dir = os.path.join(self.results, "probe")
        argv = [sys.executable, SAMPLER, "probe", "--device-index", str(self.device_index),
                "--routes", ",".join(routes), "--reads", str(int(config.get("probe_reads", 5))),
                "--out", out_dir] + self.fake_args()
        if self.bdf:
            argv += ["--bdf", self.bdf]
        plant = self.plant()
        if plant:
            argv += ["--fake-plant", json.dumps(plant)]
        self.child(argv, log=os.path.join(self.results, "probe.log"),
                   timeout=self.child_timeout(300))
        path = os.path.join(out_dir, "probe.json")
        if not os.path.isfile(path):
            return {"routes": [], "chosen": None}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def choose_route(self, probe: Dict[str, Any]) -> Tuple[Optional[str], str]:
        config = self.plan.get("sampler", {})
        wanted = config.get("source", "auto")
        usable = {route["describe"]["source"]: route for route in probe.get("routes", [])
                  if route["ok"] == route["reads"] and route["reads"] > 0}
        if wanted != "auto":
            if wanted in usable:
                return wanted, f"the plan pins {wanted}"
            return None, f"the plan pins {wanted}, which returned no power reading"
        ceiling = float(config.get("max_read_latency_s", 1.0))
        smi = usable.get("xrt-smi")
        if smi and smi["latency_s_median"] <= ceiling:
            return "xrt-smi", (f"xrt-smi answers in {smi['latency_s_median']:.3f} s "
                               f"(<= {ceiling} s) and holds no device handle")
        if "pyxrt" in usable:
            return "pyxrt", ("pyxrt is the only fast route" if smi
                             else "xrt-smi returned no power reading")
        if smi:
            return "xrt-smi", "xrt-smi is slow but the only route"
        return None, "no route returned a board power reading"

    def reset(self) -> Dict[str, Any]:
        if not self.plan.get("reset_at_start", True):
            return {"skipped": "reset_at_start is false in the plan"}
        if self.fake:
            return {"simulated": True, "command": "fake: no card to reset"}
        template = self.plan.get("reset_command") or [
            "{smi}", "reset", "-d", "{bdf}", "--force"]
        try:
            smi = sampler.resolve_smi(self.plan.get("sampler", {}).get("smi"))
        except sampler.OdinTelemetryError as exc:
            return {"error": str(exc)}
        if not self.bdf:
            return {"error": "no BDF resolved; refusing to reset an unnamed card"}
        argv = [part.replace("{smi}", smi).replace("{bdf}", self.bdf) for part in template]
        started = time.monotonic()
        timeout = self.child_timeout(600)
        try:
            completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       universal_newlines=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"command": argv, "error": f"reset did not finish within {timeout:.0f} s"}
        return {"command": argv, "exit": completed.returncode,
                "seconds": time.monotonic() - started,
                "output_tail": completed.stdout[-2000:]}

    def start_sampler(self) -> bool:
        config = self.plan.get("sampler", {})
        argv = [sys.executable, SAMPLER, "sample", "--source", str(self.route),
                "--device-index", str(self.device_index),
                "--interval-s", str(float(config.get("interval_s", 0.25))),
                "--out", self.samples, "--stop-file", self.stop_file] + self.fake_args()
        if self.bdf:
            argv += ["--bdf", self.bdf]
        plant = self.plant()
        if plant:
            argv += ["--fake-plant", json.dumps(plant)]
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
        self.sampler_log = open(os.path.join(self.results, "sampler.log"), "a",
                                encoding="utf-8")
        self.sampler_proc = subprocess.Popen(argv, stdout=self.sampler_log,
                                             stderr=subprocess.STDOUT, cwd=PACKAGE)
        deadline = time.monotonic() + float(config.get("first_reading_timeout_s", 60.0))
        while time.monotonic() < deadline:
            if self.sampler_proc.poll() is not None:
                return False
            rows = [r for r in estimator.load_jsonl(self.samples) if not r.get("header")]
            if any(r.get("ok") for r in rows):
                return True
            time.sleep(0.05)
        return False

    def stop_sampler(self) -> Optional[int]:
        if self.sampler_proc is None:
            return None
        with open(self.stop_file, "w", encoding="utf-8") as handle:
            handle.write("stop\n")
        graceful, after_term = self.stop_waits()
        try:
            code = self.sampler_proc.wait(timeout=graceful)
        except subprocess.TimeoutExpired:
            self.sampler_proc.terminate()
            try:
                code = self.sampler_proc.wait(timeout=after_term)
            except subprocess.TimeoutExpired:
                self.sampler_proc.kill()
                code = self.sampler_proc.wait()
        self.sampler_log.close()
        self.sampler_proc = None
        return code

    def applies(self, step: Dict[str, Any]) -> Tuple[bool, str]:
        when = step.get("when", "always")
        if when == "always":
            return True, ""
        if self.verdict is None:
            return False, f"when={when} but no pilot verdict exists"
        allowed = {"go": (estimator.GO,), "reduced": (estimator.REDUCED,),
                   "go_or_reduced": (estimator.GO, estimator.REDUCED)}[when]
        if self.verdict in allowed:
            return True, ""
        return False, f"when={when} and the pilot verdict is {self.verdict}"

    def run_step(self, index: int, step: Dict[str, Any]) -> Dict[str, Any]:
        kind = step["kind"]
        step_dir = os.path.join(self.results, f"step{index:02d}_{kind}"
                                + (f"_{step['design']}" if step.get("design") else ""))
        os.makedirs(step_dir, exist_ok=True)
        predicted = predict_step(self.plan, step)
        timeout = self.child_timeout(2.0 * predicted + 300.0)
        if kind == "shell_only":
            window_id = f"s{index}-shell_only"
            write_marker(self.markers, event="window_start", window_id=window_id, step=index,
                         kind="shell_only", condition="shell_only", round=0,
                         state="shell_only", shell_verified=self.shell_verified)
            time.sleep(float(step["seconds"]))
            write_marker(self.markers, event="window_end", window_id=window_id,
                         state="shell_only", passed=True)
            return {"status": "done", "dir": step_dir}
        if kind == "recertify":
            design = self.plan["designs"][step["design"]]
            if not self.fake:
                # The executor child knows no plan; the file is checked here.
                try:
                    require_xclbin_identity(design, name=step["design"])
                except OdinPowerRefusal as exc:
                    return {"status": "failed", "exit": None, "note": f"refused: {exc}",
                            "passed": False, "dir": step_dir, "timeout_s": timeout}
            write_marker(self.markers, event="recertify", step=index, design=step["design"],
                         state="loading")
            argv = [sys.executable, EXECUTOR, "--xclbin", design["xclbin"],
                    "--bundle", design["bundle"], "--chip", design["chip"],
                    "--arm", step.get("arm", executor.ARM_RESIDENT),
                    "--samples", str(int(step.get("samples", design.get("samples", 64)))),
                    "--device-index", str(self.device_index), "--results", step_dir]
            if compute_units_of(design) > 1:
                argv += ["--compute-units", str(compute_units_of(design))]
            if self.fake_pyxrt:
                argv += ["--fake-pyxrt", self.fake_pyxrt]
                if design.get("replay"):
                    argv += ["--replay", design["replay"]]
            code, note = self.child(argv, log=os.path.join(step_dir, "executor.log"),
                                    timeout=timeout)
            write_marker(self.markers, event="recertified", step=index,
                         design=step["design"], state="kernel_idle", exit=code)
            report_path = os.path.join(step_dir, executor.REPORT_NAME)
            passed = None
            identity: Optional[Dict[str, Any]] = None
            if os.path.isfile(report_path):
                with open(report_path, "r", encoding="utf-8") as handle:
                    report = json.load(handle)
                passed = bool(report.get("passed"))
                loaded = ((report.get("device") or {}).get("identity") or {})
                identity = xclbin_identity(
                    design, metadata_uuid=loaded.get("xclbin_uuid_metadata"),
                    check_file=not self.fake)
                if identity["problems"]:
                    # Certified, but on another xclbin than the plan describes.
                    passed = False
                    note = (note + "; " if note else "") + "; ".join(identity["problems"])
            # The executor writes its report for a PASS and for a red
            # certificate alike; no report means it refused or crashed.
            return {"status": "done" if passed is not None else "failed", "exit": code,
                    "note": note, "passed": passed, "dir": step_dir, "timeout_s": timeout,
                    "xclbin_status": identity["status"] if identity else None}
        argv = [sys.executable, THIS, "design", "--plan", self.plan_path, "--step", str(index),
                "--markers", self.markers, "--results", step_dir] + self.fake_args()
        code, note = self.child(argv, log=os.path.join(step_dir, "design.log"),
                                timeout=timeout)
        finished = code in (0, WINDOWS_FAILED) and os.path.isfile(
            os.path.join(step_dir, "step.json"))
        outcome = {"status": "done" if finished else "failed", "exit": code,
                   "note": note, "passed": code == 0 if finished else False,
                   "dir": step_dir, "timeout_s": timeout}
        if finished:
            with open(os.path.join(step_dir, "step.json"), "r", encoding="utf-8") as handle:
                outcome["xclbin_status"] = (json.load(handle).get("xclbin_identity")
                                            or {}).get("status")
        if kind == "pilot" and finished:
            self.pilot_result = estimator.pilot(self.samples, self.markers, self.plan,
                                                probe_ok=self.route is not None)
            self.pilot_result["route"] = self.route
            self.verdict = self.pilot_result["verdict"]
            driver.write_result(self.results, "pilot.json", self.pilot_result)
            self.say(f"PILOT VERDICT {self.verdict}: {'; '.join(self.pilot_result['reasons'])}")
            outcome["verdict"] = self.verdict
        return outcome

    def run(self) -> int:
        plan_budget = budget(self.plan)
        self.record["budget"] = plan_budget
        self.say(f"plan {self.plan.get('name')} ({(self.plan_sha256 or '?')[:12]}), worst path "
                 f"{plan_budget['worst_path_s']:.0f} of {plan_budget['budget_s']:.0f} "
                 f"card-seconds")
        if not plan_budget["fits"]:
            raise OdinPowerRefusal(
                f"the plan's worst path is {plan_budget['worst_path_s']:.0f} s against "
                f"its budget of {plan_budget['budget_s']:.0f} s; shorten it before "
                f"booking the card")
        write_marker(self.markers, event="session_start", plan_sha256=self.plan_sha256,
                     fake=self.fake)
        identity = self.identify("start")
        self.record["identity_start"] = identity
        self.bdf = sampler.normalize_bdf(identity["bdf"]) if identity.get("bdf") else None
        self.say(f"device {self.device_index} is {self.bdf}, loaded xclbin "
                 f"{identity.get('loaded_xclbin_uuid')}")
        probe = self.probe()
        self.record["probe"] = probe
        self.route, why = self.choose_route(probe)
        self.record["route"] = {"chosen": self.route, "why": why}
        self.say(f"telemetry route: {self.route} ({why})")
        reset = self.reset()
        self.record["reset"] = reset
        if not reset.get("skipped"):
            # From here the card holds the shell only (a fake card reads this).
            write_marker(self.markers, event="reset_issued", state="shell_only",
                         result=reset)
        if not reset.get("simulated") and not reset.get("skipped"):
            # A hot reset re-enumerates the user function; give it the declared
            # settle time before asking which xclbin it holds.
            time.sleep(float(self.plan.get("reset_settle_s", 10.0)))
        after = self.identify("after_reset")
        self.record["identity_after_reset"] = after
        uuid = after.get("loaded_xclbin_uuid")
        if reset.get("skipped"):
            # Nothing was reset, so nothing is verified: the user-region
            # increment is refused (valid only after a zero UUID read back).
            self.shell_verified = None
        elif (reset.get("simulated") or reset.get("exit") == 0) and uuid is not None:
            self.shell_verified = set(uuid) <= ZERO_UUID_CHARS
        else:
            self.shell_verified = False
        self.record["shell_verified"] = self.shell_verified
        write_marker(self.markers, event="reset", state="shell_only", result=reset,
                     loaded_xclbin_uuid=uuid, shell_verified=self.shell_verified)
        self.say(f"reset: {reset}; xclbin after reset {uuid} "
                 f"(shell verified: {self.shell_verified})")
        sampling = False
        if self.route is not None:
            sampling = self.start_sampler()
            if not sampling:
                self.say("the sampler produced no good reading; card power is not "
                         "measurable in this allocation")
        if not sampling:
            self.verdict = estimator.NO_GO
            self.pilot_result = {"schema": estimator.PILOT_SCHEMA, "verdict": estimator.NO_GO,
                                 "reasons": [why if self.route is None else
                                             "the sampler produced no good reading"],
                                 "route": self.route}
            driver.write_result(self.results, "pilot.json", self.pilot_result)
        teardown = overheads_of(self.plan)["teardown_s"]
        failures = 0
        try:
            for index, step in enumerate(self.plan["steps"]):
                row: Dict[str, Any] = {"index": index, "kind": step["kind"],
                                       "design": step.get("design"),
                                       "label": step.get("label"),
                                       "when": step.get("when", "always")}
                ok, reason = self.applies(step)
                needs_power = step["kind"] in ("shell_only", "pilot", "protocol", "drift")
                predicted = predict_step(self.plan, step)
                if ok and needs_power and not sampling:
                    ok, reason = False, "no telemetry: nothing to measure"
                if ok and self.elapsed() + predicted + teardown > float(self.plan["budget_s"]):
                    ok, reason = False, (f"budget: {self.elapsed():.0f} s used, "
                                         f"{predicted:.0f} s predicted")
                if not ok:
                    row.update({"status": "skipped", "reason": reason})
                    self.say(f"step {index} {step['kind']} {step.get('design', '')}: "
                             f"SKIPPED ({reason})")
                    self.record["steps"].append(row)
                    continue
                self.say(f"step {index} {step['kind']} {step.get('design', '')}: start "
                         f"(predicted {predicted:.0f} s)")
                began = self.elapsed()
                row.update(self.run_step(index, step))
                row.update({"started_s": began, "seconds": self.elapsed() - began,
                            "predicted_s": predicted})
                if row.get("status") == "failed" or row.get("passed") is False:
                    failures += 1
                self.say(f"step {index}: {row.get('status')} in {row['seconds']:.1f} s "
                         f"(passed={row.get('passed')})")
                self.record["steps"].append(row)
        finally:
            self.record["sampler_exit"] = self.stop_sampler()
        write_marker(self.markers, event="session_end")
        self.record["pilot"] = self.pilot_result
        self.record["card_seconds"] = self.elapsed()
        self.record["failures"] = failures
        # The records first: whatever the estimate costs, session.json exists.
        driver.write_result(self.results, "session.json", self.record)
        if sampling and any(w.get("kind") == "protocol"
                            for w in estimator.windows_of(estimator.load_jsonl(self.markers))):
            result = estimator.estimate(self.samples, self.markers, self.plan,
                                        pilot_result=self.pilot_result,
                                        shell_verified=self.shell_verified)
            driver.write_result(self.results, "estimate.json", result)
            table = estimator.format_estimate(result)
            with open(os.path.join(self.results, "estimate.txt"), "w", encoding="utf-8") as handle:
                handle.write(table + "\n")
            print(table)
            self.record["estimate"] = "estimate.json"
        self.record["card_seconds"] = self.elapsed()
        self.record["failures"] = failures
        driver.write_result(self.results, "session.json", self.record)
        self.say(f"session done in {self.elapsed():.1f} s, {failures} failed step(s); "
                 f"wrote {os.path.join(self.results, 'session.json')}")
        return 1 if failures else 0


# ---------------------------------------------------------------------------
# The selftest
# ---------------------------------------------------------------------------


def _check(rows: List[Dict[str, Any]], name: str, ok: bool, detail: Any = "") -> None:
    rows.append({"check": name, "passed": bool(ok),
                 "detail": json.loads(json.dumps(detail, default=str))})
    print(f"[selftest] {'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))


def _xrt_smi_document(electrical: Dict[str, Any], thermal: Dict[str, Any],
                      bdf: str) -> Dict[str, Any]:
    """The examine JSON shape of XBHelpMenus.cpp produce_reports."""
    return {"schema_version": {"schema": "JSON", "creation_date": "selftest"},
            "system": {}, "devices": [{"interface_type": "pcie", "device_id": bdf,
                                       "electrical": electrical, **thermal}]}


def selftest_parser(rows: List[Dict[str, Any]]) -> None:
    fake = driver.load_pyxrt(FAKE)
    fake.arm(plant=None)
    device = fake.device(0)
    electrical = json.loads(device.get_info(fake.xrt_info_device.electrical))
    thermal = json.loads(device.get_info(fake.xrt_info_device.thermal))
    bdf = device.get_info(fake.xrt_info_device.bdf)
    full = _xrt_smi_document(electrical, thermal, bdf)
    reading = sampler.parse_report(json.dumps(full), bdf=bdf)
    _check(rows, "parser: xrt-smi examine shape", reading["power_w"] == 21.987654
           and reading["bdf"] == bdf and reading["temps_c"].get("fpga0") == 44.0
           and "cage_temp_0" not in reading["temps_c"]
           and reading["rails"]["12v_pex"]["watts"] is not None, reading["power_field"])
    bare = sampler.parse_report({"device_id": bdf, "electrical": electrical, "thermal": thermal})
    _check(rows, "parser: pyxrt get_info shape", bare["power_w"] == 21.987654)
    numeric = json.loads(json.dumps(full))
    numeric["devices"][0]["electrical"]["power_consumption_watts"] = 21.5
    _check(rows, "parser: number instead of string",
           sampler.parse_report(numeric, bdf=bdf)["power_w"] == 21.5)
    renamed = json.loads(json.dumps(full))
    elec = renamed["devices"][0]["electrical"]
    elec["power_watts"] = elec.pop("power_consumption_watts")
    got = sampler.parse_report(renamed, bdf=bdf)
    _check(rows, "parser: renamed power field", got["power_field"].endswith("power_watts"),
           got["power_field"])
    wrapped = {"report": {"card": full}}
    _check(rows, "parser: extra wrapper level",
           sampler.parse_report(wrapped, bdf=bdf)["power_w"] == 21.987654)
    lean = {"electrical": {"power_consumption_watts": "19.0"}}
    got = sampler.parse_report(lean)
    _check(rows, "parser: no rails, no thermals", got["rails"] == {} and got["temps_c"] == {})
    refusals = (
        ("missing power field", {"electrical": {"power_rails": electrical["power_rails"]}},
         sampler.OdinTelemetryMissing, "no board power field"),
        ("N/A power", {"electrical": dict(electrical, power_consumption_watts="N/A")},
         sampler.OdinTelemetryMissing, "not a number"),
        ("no sensors present", {"electrical": {"msg": "No sensors present"}},
         sampler.OdinTelemetryMissing, "No sensors present"),
        ("zero watts", {"electrical": dict(electrical, power_consumption_watts="0.000000")},
         sampler.OdinTelemetryMissing, "outside"),
        ("implausible watts", {"electrical": dict(electrical, power_consumption_watts="9999")},
         sampler.OdinTelemetryMissing, "outside"),
        ("other card only", _xrt_smi_document(electrical, thermal, "0000:c1:00.1"),
         sampler.OdinTelemetryMismatch, "another card"),
        ("not JSON", "Error: device busy", sampler.OdinTelemetryMissing, "not JSON"),
    )
    for name, document, error, needle in refusals:
        try:
            sampler.parse_report(document, bdf=bdf)
        except error as exc:
            _check(rows, f"parser refuses: {name}", needle in str(exc), type(exc).__name__)
            continue
        except sampler.OdinTelemetryError as exc:
            _check(rows, f"parser refuses: {name}", False, f"wrong type {type(exc).__name__}")
            continue
        _check(rows, f"parser refuses: {name}", False, "no refusal")


def _synthetic(path_samples: str, path_markers: str, *, dynamic: float, static: float,
               sigma: float, seed: int, rounds: int = 5, per_window: int = 200,
               dt: float = 0.25, transfer: Optional[float] = None,
               shell_verified: Optional[bool] = True, trend_w_per_s: float = 0.0,
               rotate_rounds: bool = True) -> None:
    """Records of a session with known levels: shell 20 W, idle + static, loop +
    dynamic, and (``transfer``) a transfer control at idle + static + transfer.

    ``trend_w_per_s`` adds a linear drift of the whole card from the first
    reading on; ``rotate_rounds=False`` keeps one condition order in every round
    (so a trend does not cancel). The defaults give the archived records."""
    rng = random.Random(seed)
    t = 1000.0
    with open(path_samples, "w", encoding="utf-8") as samples, \
            open(path_markers, "w", encoding="utf-8") as markers:
        samples.write(json.dumps({"header": True, "interval_s": dt}) + "\n")
        markers.write(json.dumps({"event": "reset", "t_mono": t - 1.0,
                                  "shell_verified": shell_verified}) + "\n")

        def window(kind: str, condition: str, round_number: int, level: float) -> None:
            nonlocal t
            wid = f"{kind}-{round_number}-{condition}"
            markers.write(json.dumps({"event": "window_start", "window_id": wid, "kind": kind,
                                      "design": "stock", "condition": condition,
                                      "round": round_number, "t_mono": t}) + "\n")
            for _ in range(per_window):
                t += dt
                watts = level + rng.gauss(0.0, sigma)
                if trend_w_per_s:
                    watts += trend_w_per_s * (t - 1000.0)
                samples.write(json.dumps({"t_mono": t, "ok": True, "latency_s": 0.01,
                                          "power_w": watts}) + "\n")
            t += dt
            end = {"event": "window_end", "window_id": wid, "t_mono": t, "passed": True}
            if condition.startswith(LOOP):
                end.update({"inferences": 64, "kernel_duty": 0.9})
            elif condition.startswith(TRANSFER):
                end.update({"inferences": 64.0, "kernel_duty": 0.0, "control": "transfer"})
            markers.write(json.dumps(end) + "\n")
            t += 5 * dt

        added = {"kernel_idle": 0.0, "loop_as_run": dynamic,
                 "transfer_as_run": transfer or 0.0}
        conditions = ["kernel_idle", "loop_as_run"] + (
            ["transfer_as_run"] if transfer is not None else [])
        window("shell_only", "shell_only", 0, 20.0)
        window("pilot", "kernel_idle", 0, 20.0 + static)
        for round_number in range(rounds):
            order = rotate(conditions, round_number) if rotate_rounds else list(conditions)
            for condition in order:
                window("protocol", condition, round_number, 20.0 + static + added[condition])
        window("drift", "kernel_idle", 0, 20.0 + static)


def selftest_estimator(rows: List[Dict[str, Any]], work: str) -> None:
    plan = {"steps": [{"kind": "protocol", "design": "stock", "rounds": 5, "block_s": 50,
                       "when": "go"}],
            "thresholds": {"pilot": {"expected_differential_w": {"stock": 0.3},
                                     "samples_per_condition_min": 100}}}
    cases = (("NUMBER", 0.30, 0.02, 11), ("BOUND", 0.012, 0.05, 12),
             ("NOT_SEPARATED", 0.0, 0.05, 13))
    for expected, dynamic, sigma, seed in cases:
        samples = os.path.join(work, f"synthetic_{expected}_samples.jsonl")
        markers = os.path.join(work, f"synthetic_{expected}_markers.jsonl")
        _synthetic(samples, markers, dynamic=dynamic, static=0.4, sigma=sigma, seed=seed)
        result = estimator.estimate(samples, markers, plan)
        dyn = result["designs"]["stock"]["arms"]["as_run"]["workload_increment_w"]
        covered = dyn["lo"] <= dynamic <= dyn["hi"]
        _check(rows, f"estimator: known differential {dynamic} W -> {expected}",
               dyn["outcome"] == expected and covered,
               f"{dyn['outcome']} [{dyn['lo']:.4f}, {dyn['hi']:.4f}]")
        region = result["designs"]["stock"]["user_region_increment_w"]
        _check(rows, f"estimator: user-region increment 0.4 W recovered ({expected} case)",
               region["lo"] <= 0.4 <= region["hi"], region["outcome"])
    labels = set()
    for expected, _d, _s, _seed in cases:
        labels.add(expected)
    _check(rows, "estimator: outcome labels are exactly the three declared",
           labels == set(estimator.OUTCOMES))
    # The pilot gate on synthetic pilot windows.
    pilot_plan = dict(plan)
    good = os.path.join(work, "synthetic_NUMBER_samples.jsonl")
    good_m = os.path.join(work, "synthetic_NUMBER_markers.jsonl")
    verdict = estimator.pilot(good, good_m, pilot_plan)
    _check(rows, "pilot gate: quiet sensor -> GO", verdict["verdict"] == estimator.GO,
           verdict["reasons"])
    noisy = os.path.join(work, "synthetic_noisy_samples.jsonl")
    noisy_m = os.path.join(work, "synthetic_noisy_markers.jsonl")
    _synthetic(noisy, noisy_m, dynamic=0.3, static=0.4, sigma=1.5, seed=14)
    verdict = estimator.pilot(noisy, noisy_m, pilot_plan)
    _check(rows, "pilot gate: noisy sensor -> REDUCED",
           verdict["verdict"] == estimator.REDUCED, verdict["reasons"])
    stuck = os.path.join(work, "synthetic_stuck_samples.jsonl")
    stuck_m = os.path.join(work, "synthetic_stuck_markers.jsonl")
    _synthetic(stuck, stuck_m, dynamic=0.3, static=0.4, sigma=0.0, seed=15)
    verdict = estimator.pilot(stuck, stuck_m, pilot_plan)
    _check(rows, "pilot gate: frozen reading -> NO_GO",
           verdict["verdict"] == estimator.NO_GO, verdict["reasons"])
    # The warm-up the gate took off its counts, and the sample floor it implies.
    verdict = estimator.pilot(good, good_m, pilot_plan)
    warm = verdict.get("warmup") or {}
    _check(rows, "pilot: the warm-up is recorded with its rule",
           warm.get("readings", 0) >= 5 and "2 x the measured refresh" in warm.get("rule", "")
           and all(row["warmup_readings_per_window"] == warm["readings"]
                   for row in verdict["predictions"]), warm)
    slow = estimator.pilot_verdict(
        {"availability": 1.0, "changes": 200, "refresh_s": 1.0, "sigma_w": 0.001,
         "quantum_w": 0.001},
        {"steps": [{"kind": "protocol", "design": "stock", "rounds": 5, "block_s": 60,
                    "when": "go"}]}, interval_s=1.0)
    _check(rows, "pilot gate: 1 s polling leaves 275 usable readings, under the 300 floor",
           slow["predictions"][0]["polled_per_condition"] == 275.0
           and slow["verdict"] == estimator.REDUCED, slow["predictions"][0])


def selftest_plan(work: str, *, absent: bool) -> str:
    deployment = os.path.join(PACKAGE, "deployment")
    plan = {
        "schema": PLAN_SCHEMA, "name": "selftest" + ("_no_sensors" if absent else ""),
        "budget_s": 600, "device_index": 0, "reset_at_start": True,
        "overheads": {"session_start_s": 5, "reset_s": 1, "xclbin_load_s": 1,
                      "teardown_s": 5, "recertify_host_s": 1},
        "sampler": {"source": "auto", "interval_s": 0.02, "probe_reads": 3,
                    "first_reading_timeout_s": 20},
        "designs": {"stock": {
            "xclbin": "/selftest/no-such.xclbin",
            "xclbin_identity": {"uuid": "uuid-of-/selftest/no-such.xclbin",
                                "rebuilt": False},
            "vivado": {"odin_0_dynamic_w": 0.2, "hmss_0_dynamic_w": 2.0,
                       "class": "selftest: illustrative numbers, not a report",
                       "describes": "the selftest's fake build"},
            "bundle": os.path.join(deployment, "nc1_two_core_passes.json"),
            "replay": os.path.join(deployment, "nc1_two_core_passes_capture.json"),
            "chip": "odin_stock_256x256", "samples": 4,
            "campaign_s": {"as_run": 0.2, "resident": 0.2}}},
        "thresholds": {"pilot": {"expected_differential_w": {"stock": 0.3},
                                 "samples_per_condition_min": 100},
                       "estimate": {"bootstrap": 500, "warmup_samples": 3,
                                    "drift_window_s": 0.6}},
        "steps": [
            {"kind": "shell_only", "seconds": 1.5},
            {"kind": "pilot", "design": "stock", "seconds": 3.0},
            {"kind": "recertify", "design": "stock", "arm": "resident"},
            {"kind": "protocol", "design": "stock", "when": "go",
             "conditions": ["kernel_idle", "loop_as_run", "loop_resident",
                            "transfer_as_run"],
             "rounds": 3, "block_s": 1.2},
            {"kind": "protocol", "design": "stock", "when": "reduced", "label": "reduced",
             "conditions": ["kernel_idle", "loop_as_run"], "rounds": 3, "block_s": 2.0},
            {"kind": "drift", "design": "stock", "seconds": 1.5, "when": "go_or_reduced"},
        ],
        "fake": {"pyxrt": FAKE, "plant": {
            "shell_w": 20.0, "static_w": 0.40, "dynamic_w": 0.30, "transfer_w": 0.05,
            "sigma_w": 0.02,
            "quantum_w": 0.001, "refresh_s": 0.0, "seed": 7,
            "initial_state": "kernel_idle", "absent": absent}},
    }
    path = os.path.join(work, f"plan_{plan['name']}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
    return path


def selftest_session(rows: List[Dict[str, Any]], work: str) -> None:
    results = os.path.join(work, "session_measures")
    plan_path = selftest_plan(work, absent=False)
    code = Session(plan_path, results).run()
    with open(os.path.join(results, "session.json"), "r", encoding="utf-8") as handle:
        record = json.load(handle)
    _check(rows, "session: completed with no failed step", code == 0, record["failures"])
    _check(rows, "reset: the fake card reads back a zero UUID (shell verified)",
           record.get("shell_verified") is True,
           (record.get("identity_after_reset") or {}).get("loaded_xclbin_uuid"))
    _check(rows, "session: pilot verdict GO", record["pilot"]["verdict"] == estimator.GO,
           record["pilot"].get("reasons"))
    kinds = [(s["kind"], s["status"]) for s in record["steps"]]
    _check(rows, "session: the reduced branch is skipped under GO",
           ("protocol", "skipped") in kinds and kinds.count(("protocol", "done")) == 1, kinds)
    recert = next(s for s in record["steps"] if s["kind"] == "recertify")
    with open(os.path.join(recert["dir"], executor.REPORT_NAME), "r", encoding="utf-8") as handle:
        report = json.load(handle)
    programmed = [row.get("programming_run") for row in report["per_core"]]
    _check(rows, "recertify: resident arm certified pass by pass",
           report["passed"] and report["campaign"]["arm"] == "resident"
           and all(programmed) and all(c["passed"] for c in report["certificates"]),
           f"{len(report['certificates'])} certificates")
    identity = report["device"]["identity"]
    _check(rows, "stamps: BDF and xclbin UUID in the deployment report",
           identity.get("bdf") == "0000:81:00.1" and bool(identity.get("xclbin_uuid")),
           identity)
    proto = next(s for s in record["steps"] if s["kind"] == "protocol" and s["status"] == "done")
    with open(os.path.join(proto["dir"], "step.json"), "r", encoding="utf-8") as handle:
        step = json.load(handle)
    orders = {}
    for window in step["windows"]:
        orders.setdefault(window["round"], []).append(window["condition"])
    _check(rows, "protocol: rotated condition order per round",
           len({tuple(order) for order in orders.values()}) == len(orders), orders)
    _check(rows, "protocol: every loop window certified",
           all(w.get("passed") for w in step["windows"]))
    _check(rows, "identity: the declared xclbin UUID matched the file (archived)",
           (step.get("xclbin_identity") or {}).get("status") == "archived"
           and step["xclbin_identity"]["uuid_checked"] is True, step.get("xclbin_identity"))
    controls = [w for w in step["windows"] if w["condition"] == "transfer_as_run"]
    _check(rows, "control: the transfer windows never started the kernel",
           bool(controls) and all(w["ap_starts"] == 0 and w["passes"] > 0
                                  and w["bytes_to_device"] > 0 and w["kernel_duty"] == 0.0
                                  for w in controls),
           [(w["passes"], w["ap_starts"]) for w in controls])
    temps = [r for r in estimator.load_jsonl(os.path.join(results, "samples.jsonl"))
             if r.get("ok") and r.get("temps_c")]
    _check(rows, "stamps: temperature on every good sample",
           bool(temps) and all("fpga0" in r["temps_c"] for r in temps), len(temps))
    with open(os.path.join(results, "estimate.json"), "r", encoding="utf-8") as handle:
        result = json.load(handle)
    stock = result["designs"]["stock"]
    for arm in ("as_run", "resident"):
        dyn = stock["arms"][arm]["workload_increment_w"]
        _check(rows, f"estimate: {arm} workload increment recovers the plant's 0.30 W "
                     f"as a NUMBER",
               dyn["outcome"] == estimator.NUMBER and dyn["lo"] <= 0.30 <= dyn["hi"]
               and dyn["quantity"].startswith("workload increment at the card boundary"),
               f"{dyn['point']:.4f} [{dyn['lo']:.4f}, {dyn['hi']:.4f}]")
        energy = stock["arms"][arm]["workload_energy_per_inference_mJ"]
        _check(rows, f"estimate: {arm} workload energy per inference is reported",
               energy is not None and energy["point"] > 0, energy and energy["statement"])
    control = stock["controls"]["as_run"]["transfer_increment_w"]
    _check(rows, "estimate: the transfer control recovers the plant's 0.05 W",
           control["lo"] <= 0.05 <= control["hi"], control["statement"])
    net = stock["arms"]["as_run"]["net_of_transfers_w"]
    _check(rows, "estimate: net of the transfers recovers the plant's 0.25 W",
           net["lo"] <= 0.25 <= net["hi"], net["statement"])
    region = stock.get("user_region_increment_w", {})
    _check(rows, "estimate: the user-region increment recovers the plant's 0.40 W",
           region.get("lo", 1) <= 0.40 <= region.get("hi", 0), region.get("statement"))
    comparison = (stock.get("vivado_comparison") or {}).get("arms", {}).get("as_run", {})
    _check(rows, "estimate: odin_0 and hmss_0 beside the upper bound, no ratio",
           comparison.get("upper_bound_quantity") == "net_of_transfers_w"
           and comparison.get("hmss_0_dynamic_w") == 2.0
           and "ratio" not in json.dumps(stock.get("vivado_comparison")).lower(),
           comparison.get("reading"))
    _check(rows, "estimate: drift reference inside its band",
           result.get("drift", {}).get("inside_band") is True, result.get("drift"))
    with open(os.path.join(results, "pilot.json"), "r", encoding="utf-8") as handle:
        pilot_record = json.load(handle)
    again = estimator.estimate(os.path.join(results, "samples.jsonl"),
                               os.path.join(results, "markers.jsonl"),
                               load_plan(plan_path), pilot_result=pilot_record)
    _check(rows, "estimate: recomputed from the saved records, it is the same",
           json.dumps(again, sort_keys=True) == json.dumps(result, sort_keys=True))

    results = os.path.join(work, "session_no_sensors")
    plan_path = selftest_plan(work, absent=True)
    code = Session(plan_path, results).run()
    with open(os.path.join(results, "session.json"), "r", encoding="utf-8") as handle:
        record = json.load(handle)
    status = {f"{s['kind']}/{s.get('label') or ''}": s["status"] for s in record["steps"]}
    _check(rows, "no sensors: verdict NO_GO before any window",
           record["pilot"]["verdict"] == estimator.NO_GO, record["pilot"]["reasons"])
    _check(rows, "no sensors: recertification still runs, power steps skipped",
           status.get("recertify/") == "done"
           and status.get("pilot/") == "skipped"
           and status.get("protocol/") == "skipped", status)
    _check(rows, "no sensors: no estimate is written",
           not os.path.exists(os.path.join(results, "estimate.json")))


def selftest(results: str) -> int:
    if os.path.isdir(results):
        shutil.rmtree(results)
    os.makedirs(results)
    rows: List[Dict[str, Any]] = []
    print("[selftest] STRUCTURE, NOT SILICON: a fake pyxrt, a synthetic power plant "
          "and the frozen replay; nothing here measures a card.")
    selftest_parser(rows)
    selftest_estimator(rows, results)
    selftest_session(rows, results)
    failed = [row for row in rows if not row["passed"]]
    driver.write_result(results, "selftest_power.json",
                        {"host": sampler.host_stamp(), "rows": rows,
                         "passed": not failed})
    print(f"[selftest] {len(rows) - len(failed)}/{len(rows)} checks passed")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    ses = sub.add_parser("session", help="one allocation: the whole plan")
    ses.add_argument("--plan", required=True)
    ses.add_argument("--results", required=True)
    ses.add_argument("--fake-pyxrt", default=None)
    bud = sub.add_parser("budget", help="the plan's predicted card-seconds")
    bud.add_argument("--plan", required=True)
    des = sub.add_parser("design", help="one design step (a child of session)")
    des.add_argument("--plan", required=True)
    des.add_argument("--step", type=int, required=True)
    des.add_argument("--markers", required=True)
    des.add_argument("--results", required=True)
    des.add_argument("--fake-pyxrt", default=None)
    uni = sub.add_parser("unit", help="one compute unit of an N-unit step (a child of design)")
    uni.add_argument("--plan", required=True)
    uni.add_argument("--step", type=int, required=True)
    uni.add_argument("--cu-index", type=int, required=True)
    uni.add_argument("--fake-pyxrt", default=None)
    ide = sub.add_parser("identity", help="the BDF and loaded xclbin UUID of a device")
    ide.add_argument("--device-index", type=int, default=0)
    ide.add_argument("--fake-pyxrt", default=None)
    ide.add_argument("--out", required=True)
    ide.add_argument("--fake-plant", default=None, help="fake mode: the synthetic plant (JSON)")
    pre = sub.add_parser("preflight", help="owner step 3: staged files against the "
                                           "plan's pins, no card")
    pre.add_argument("--plan", required=True)
    sel = sub.add_parser("selftest", help="parser, estimator and two fake sessions")
    sel.add_argument("--results", required=True)
    options = parser.parse_args(argv)
    try:
        if options.command == "session":
            return Session(options.plan, options.results, fake_pyxrt=options.fake_pyxrt).run()
        if options.command == "budget":
            plan = load_plan(options.plan)
            result = budget(plan)
            for row in result["steps"]:
                print(f"step {row['index']:>2} {row['kind']:<10} {str(row['design'] or ''):<6} "
                      f"when={row['when']:<13} {row['predicted_s']:8.1f} s")
            print(f"fixed {result['fixed_s']:.0f} s; GO path {result['go_path_s']:.0f} s; "
                  f"REDUCED path {result['reduced_path_s']:.0f} s; worst "
                  f"{result['worst_path_s']:.0f} s of {result['budget_s']:.0f} s "
                  f"({'fits' if result['fits'] else 'DOES NOT FIT'})")
            return 0 if result["fits"] else 1
        if options.command == "preflight":
            report = preflight(load_plan(options.plan))
            for name, row in report["designs"].items():
                archived = row.get("archived") or {}
                print(f"{name}: {'ok' if row['ok'] else 'NOT OK'}; xclbin "
                      f"{row['xclbin']['xclbin']} {row['xclbin']['checks'].get('size_bytes')} "
                      f"bytes (archived {archived.get('size_bytes')} bytes, written "
                      f"{archived.get('written')}); bundle {row['bundle'].get('self_hash')}; "
                      f"uuid declared {row['uuid_declared']}; rebuilt {row['rebuilt']}")
                for problem in row["problems"]:
                    print(f"    problem: {problem}")
                print(f"    UUID and date: {row['uuid_command']}")
            print("preflight " + ("ok" if report["ok"] else "NOT OK"))
            return 0 if report["ok"] else 1
        if options.command == "design":
            plan = load_plan(options.plan)
            fake = options.fake_pyxrt or (plan.get("fake") or {}).get("pyxrt")
            return run_design_step(plan, options.step, markers=options.markers,
                                   results=options.results, fake_pyxrt=fake,
                                   plan_path=options.plan)
        if options.command == "unit":
            return run_unit_worker(options.plan, options.step, options.cu_index,
                                   options.fake_pyxrt)
        if options.command == "identity":
            result = run_identity(device_index=options.device_index,
                                  fake_pyxrt=options.fake_pyxrt,
                                  fake_plant=(json.loads(options.fake_plant)
                                              if options.fake_plant else None))
            with open(options.out, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
            print(json.dumps(result, sort_keys=True))
            return 0
        if options.command == "selftest":
            return selftest(options.results)
        parser.print_help()
        return 2
    except (OdinPowerRefusal, driver.OdinDriverError, sampler.OdinTelemetryError) as exc:
        print(f"REFUSING: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
