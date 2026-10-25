#!/usr/bin/env python
"""Requirement census (N9) of the scout's network instances and the torchvision classification zoo.

    python scripts/requirement_census.py scout OUT.json --scout-dir RA/10_wave1/oracles
    python scripts/requirement_census.py zoo-roster OUT.json [--max-params 200000000]
    python scripts/requirement_census.py zoo OUT.json NAME[,NAME...] [--side N] [--verdict-side N]
    python scripts/requirement_census.py check CENSUS.json [CENSUS.json ...] [--pins PINS.json]

CPU only, random weights (seed 0), eval mode, the MVM contract at C0. Each member carries the stage-2 fp64 verdict
of ``probe_witness_conversion`` (column "fp64 agreement (max relative error)") or its refusal by type name, and one
row per capability level C0-C6 (two where D is declared: on-chip write-back, host payload write). Rows C1-C6 are a
requirement placement on the toolchain IR, labelled so. A MAC field the two counters cannot back is absent with its
reason under ``withheld`` (``mapping/capability/mac_split.py``). A member whose census crashes anonymously is
recorded as a loud failure (never as a refusal), so ``check`` fails on it; so is a measured value mismatch.

A verdict judged on another input than the rows carries ``verdict_input_shape`` on the member and on its C0 row; a
zoo member also carries its declared shape and both resolutions. Every census file pins the code that wrote it: the
runner's sha, and ``source_digest`` of the ``mapping/capability`` package and of the whole ``mimarsinan`` tree it
imported; ``check`` fails a file without them and a set of files written by more than one code state.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torchvision  # noqa: E402
import torchvision.models as tvm  # noqa: E402

import mimarsinan  # noqa: E402
from mimarsinan.common.refusal import ToolchainRefusal  # noqa: E402
from mimarsinan.mapping import capability  # noqa: E402
from mimarsinan.mapping.capability.ladder import (  # noqa: E402
    LADDER,
    PLACEMENT_LABEL,
    capability_level,
    payload_writes,
)
from mimarsinan.mapping.capability.mac_split import joined_reasons, mac_statement  # noqa: E402
from mimarsinan.mapping.capability.member import (  # noqa: E402
    AGREEMENT,
    REFUSED,
    VALUE_MISMATCH,
    VERDICT_COLUMN,
    MemberCensus,
    c0_verdict,
    member_census,
)
from mimarsinan.mapping.capability.census import value_bytes  # noqa: E402
from mimarsinan.mapping.capability.pricing_inputs import level_quantities, period_census_of  # noqa: E402
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING  # noqa: E402
from mimarsinan.torch_mapping.converter import probe_stages  # noqa: E402

VALUE_BITS = 8
#: The scout's oracle files the instances are imported from, by sha256 prefix (RA/10_wave1/oracles/SHA256.txt).
SCOUT_SHAS = {"req_census.py": "296ff421dcc7088b", "net_models.py": "a538dbb191e833f5"}
SCOUT_CLASSES = 10
#: The scout's one-member-per-family torchvision subset (tv_zoo.py, sha256 3ed9c8dca1bba326), kept as a column.
TV_ZOO_23 = (
    "alexnet", "vgg11_bn", "resnet18", "resnet50", "resnext50_32x4d", "wide_resnet50_2", "densenet121",
    "squeezenet1_1", "googlenet", "inception_v3", "mobilenet_v2", "mobilenet_v3_small", "mnasnet0_5",
    "shufflenet_v2_x0_5", "efficientnet_b0", "efficientnet_v2_s", "regnet_x_400mf", "regnet_y_400mf",
    "convnext_tiny", "vit_b_16", "swin_t", "swin_v2_t", "maxvit_t",
)
ZOO_MAX_PARAMS = 200_000_000
ZOO_CLASSES = 1000
#: Builder arguments and input sides as the scout's zoo declares them (tv_zoo.py MODELS); every other member: 224.
ZOO_BUILDER_KWARGS = {"googlenet": {"aux_logits": False, "init_weights": True},
                      "inception_v3": {"aux_logits": False, "init_weights": True}}
ZOO_INPUT_SIDE = {"inception_v3": 299}
ZOO_DEFAULT_SIDE = 224
ZOO_CHANNELS = 3
GROUPED_BOUNDARY = "grouped convolution on the host until N5"
#: The provenance fields of every census file: one value each over the files ``check`` reads together.
RECORD_DIGESTS = ("script_sha256_16", "capability_sha256_16", "mimarsinan_src_sha256_16")


def _sha16(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def source_digest(root: str) -> str:
    """sha256 (first 16 hex digits) over every ``*.py`` file under ``root``: for each file in sorted order of its
    POSIX path relative to ``root``, that path, a NUL byte, the file's bytes and a NUL byte."""
    base = Path(root)
    digest = hashlib.sha256()
    for rel in sorted(path.relative_to(base).as_posix() for path in base.rglob("*.py")):
        digest.update(rel.encode() + b"\0" + (base / rel).read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def scout_members(scout_dir: str) -> List[Tuple[str, Callable, tuple, Optional[str]]]:
    """The oracle's ``members()``, imported from the pinned scout files after their shas are checked."""
    for name, sha in SCOUT_SHAS.items():
        found = _sha16(os.path.join(scout_dir, name))
        if found != sha:
            raise SystemExit(f"{name} in {scout_dir} has sha256 {found}, expected {sha}")
    sys.path.insert(0, os.path.abspath(scout_dir))
    with contextlib.redirect_stdout(io.StringIO()):
        oracle = importlib.import_module("req_census")
    return list(oracle.members())


def zoo_builder(name: str) -> Callable[[], torch.nn.Module]:
    return lambda: tvm.get_model(name, **ZOO_BUILDER_KWARGS.get(name, {}))


def zoo_names() -> List[str]:
    return sorted(tvm.list_models(module=tvm))


def zoo_roster(max_params: int, names: Optional[Sequence[str]] = None) -> List[dict]:
    """Every torchvision classification builder with its parameter count; at most ``max_params`` are in the zoo."""
    rows = []
    for name in names if names is not None else zoo_names():
        torch.manual_seed(0)
        params = int(sum(p.numel() for p in zoo_builder(name)().parameters()))
        rows.append({"name": name, "params": params, "in_zoo": params <= max_params, "in_tv_zoo_23": name in TV_ZOO_23})
    return rows


def _is_refusal(cls: Optional[type]) -> Optional[bool]:
    return None if cls is None else issubclass(cls, ToolchainRefusal)


def _verdict_json(census: MemberCensus) -> dict:
    v = census.verdict
    return {"column": v.column, "status": v.status, "relative_error": v.relative_error, "bound": v.bound,
            "stages": [list(s) for s in v.stages], "refusal": v.refusal, "reason": v.reason,
            "refusal_is_toolchain_refusal": _is_refusal(v.refusal_type), "root_refusal": v.root_refusal,
            "root_refusal_is_toolchain_refusal": _is_refusal(v.root_refusal_type),
            "value_mismatch": v.value_mismatch, "missing_stages": list(v.missing_stages)}


def _share(part: Optional[int], total: Optional[int]) -> Optional[float]:
    return None if part is None or not total else round(part / total, 6)


def _row_json(row, census: MemberCensus, verdict_input_shape: Optional[Sequence[int]] = None) -> dict:
    """One level row; a MAC field its counters cannot back is absent, its reason under ``withheld`` (a share names
    the reasons of its host MACs and of its total, as ``joined_reasons`` joins them). The C0 row carries
    ``verdict_input_shape`` when its verdict was judged on another input than the row's IR."""
    counters = census.mac_counters
    statement = mac_statement(row, counters)
    withheld = dict(statement.withheld)
    period = period_census_of(row, bytes_of_cells=lambda cells: value_bytes(cells, VALUE_BITS))
    data = {
        "level": row.level, "title": capability_level(row.level).title, "payload_write": row.payload_write,
        "label": row.label, "host_ops": row.host_ops, "host_kinds": row.host_kinds,
        "neural_segments": row.neural_segments, "host_segments": row.host_segments,
        "dependency_neural_segments": row.dependency_neural_segments,
        "dependency_host_segments": row.dependency_host_segments,
        "boundary_elements": {"entry": row.boundary.entry, "carry_in": row.boundary.carry_in,
                              "carry_out": row.boundary.carry_out, "readout": row.boundary.readout},
        "boundary_bytes_8bit": row.boundary_bytes, "host_macs_by_kind": dict(statement.host_macs_by_kind),
        "core_mac_floor": row.core_mac_floor, "stated_boundaries": [],
        "dynamic_payload_bytes_8bit": row.dynamic_payload_bytes, "lane_ops": row.lane_ops,
        "element_ops_by_kind": row.element_ops_by_kind, "latency_steps": row.latency_steps,
        "quantities": level_quantities(row, counters=counters),
        "write_census": {"rule": period.label,
                         "writing_passes": sum(1 for p in period.passes if p.written_cells_backing_store),
                         "written_cells_backing_store_per_period": sum(
                             p.written_cells_backing_store for p in period.passes),
                         "payload_bytes_per_period_8bit_weights": sum(p.payload_bytes for p in period.passes),
                         "written_cells_producer_per_inference": sum(p.written_cells_producer for p in period.passes),
                         "producer_payload_bytes_8bit": row.dynamic_payload_bytes},
    }
    if statement.host_macs is not None:
        data["host_macs"] = statement.host_macs
    share = _share(statement.onchip_macs, statement.total_macs)
    if share is not None:
        data["onchip_mac_share"] = share
    elif "onchip_macs" in withheld:
        withheld["onchip_mac_share"] = withheld["onchip_macs"]
    for stated in statement.stated_boundaries:
        entry: dict = {"boundary": stated.boundary}
        if stated.host_macs is not None:
            entry["host_macs"] = stated.host_macs
        boundary_share = _share(stated.host_macs, statement.total_macs)
        if boundary_share is not None:
            entry["host_mac_share"] = boundary_share
        else:
            entry["withheld"] = joined_reasons(stated.withheld, statement.total_withheld)
        data["stated_boundaries"].append(entry)
    data["withheld"] = withheld
    if statement.total_withheld:
        data["mac_total_withheld"] = statement.total_withheld
    if counters is not None and not counters.agree:
        data["mac_counters_disagree"] = {"reason": counters.host_withheld, "count_host_ops": counters.ir_host_macs,
                                         "estimate_onchip_fraction_host": counters.estimator_host_macs}
    if row.level == "C0":
        data[VERDICT_COLUMN] = census.verdict.relative_error
        if verdict_input_shape is not None:
            data["verdict_input_shape"] = list(verdict_input_shape)
    return data


def _rowless(census: MemberCensus, verdict_input_shape: Optional[Sequence[int]] = None) -> List[dict]:
    """The level rows of a member without a census: each names its refusal, or the verdict that left no rows."""
    verdict = census.verdict
    if census.census_refusal is not None or verdict.status == REFUSED:
        key, reason = "refused", census.census_refusal or verdict.root_refusal
    else:
        key, reason = "no_census", verdict.status
    rows = [{"level": level.name, "title": level.title, "payload_write": write,
             "label": None if level.name == "C0" else PLACEMENT_LABEL, key: reason}
            for level in LADDER for write in payload_writes(level)]
    if verdict_input_shape is not None:
        rows[0]["verdict_input_shape"] = list(verdict_input_shape)
    return rows


def _peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)


def census_record(model, input_shape, num_classes: int, verdict_shape=None) -> dict:
    """One member's record: the verdict at ``verdict_shape`` (default ``input_shape``), the rows at ``input_shape``.

    The process peak RSS is read after each, so a verdict judged on a smaller input shows whether the census alone
    fits. A loud failure is captured here (a script boundary), never relabelled a refusal.
    """
    started = time.perf_counter()
    verdict_shape = tuple(verdict_shape or input_shape)
    peaks = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            verdict = c0_verdict(model, verdict_shape, packaging=MVM_PACKAGING)
            peaks["after_verdict"] = _peak_rss_gb()
            census = member_census(model, input_shape, num_classes, packaging=MVM_PACKAGING, value_bits=VALUE_BITS,
                                   verdict=verdict)
            peaks["after_census"] = _peak_rss_gb()
    except Exception as crash:  # noqa: BLE001 - recorded as a loud failure and failed by ``check``
        lines = str(crash).strip().splitlines()
        return {"loud_failure": f"{type(crash).__name__}: {lines[0][:300] if lines else ''}",
                "wall_s": round(time.perf_counter() - started, 1), "peak_rss_gb": peaks}
    counters = census.mac_counters
    marked = list(verdict_shape) if verdict_shape != tuple(input_shape) else None
    record = {"verdict": _verdict_json(census),
              "rows": ([_row_json(row, census, marked) for row in census.rows] if census.rows
                       else _rowless(census, marked)),
              "wall_s": round(time.perf_counter() - started, 1), "peak_rss_gb": peaks}
    if marked is not None:
        record["verdict_input_shape"] = marked
    if census.census_refusal is not None:
        record["census_refusal"] = census.census_refusal
    if counters is not None:
        floor = census.rows[0].core_mac_floor if census.rows else None
        record["mac_counters"] = {
            "count_host_ops": counters.ir_host_macs, "estimate_onchip_fraction_host": counters.estimator_host_macs,
            "estimate_onchip_fraction_total": counters.estimator_total_macs, "agree": counters.agree,
            "has_multi_input_op": counters.has_multi_input_op, "reason": counters.reason,
            "ir_core_mac_floor": floor,
            "total_withheld": None if floor is None else counters.total_withheld(floor)}
    return record


def _header(kind: str) -> dict:
    return {"kind": kind, "script_sha256_16": _sha16(os.path.abspath(__file__)),
            "capability_sha256_16": source_digest(os.path.dirname(os.path.abspath(capability.__file__))),
            "mimarsinan_src_sha256_16": source_digest(os.path.dirname(os.path.abspath(mimarsinan.__file__))),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "value_bits": VALUE_BITS,
            "verdict_column": VERDICT_COLUMN,
            "ladder": [[level.name, level.title, sorted(level.classes)] for level in LADDER], "members": {}}


def run_scout(out: str, scout_dir: str) -> None:
    doc = _header("scout")
    doc["scout_shas"] = SCOUT_SHAS
    for key, factory, shape, note in scout_members(scout_dir):
        torch.manual_seed(0)
        record = census_record(factory().eval(), shape, SCOUT_CLASSES)
        doc["members"][key] = {"family": "scout", "note": note, "input_shape": list(shape), **record}
        _write(out, doc)
        print(key, _summary(record), flush=True)


def resolution_note(declared: int, side: int) -> str:
    return "declared" if side == declared else f"reduced from {declared} to {side}"


def _square(side: int) -> Tuple[int, int, int]:
    return (ZOO_CHANNELS, int(side), int(side))


def zoo_sides(declared: int, side_override: Optional[int] = None, verdict_side: Optional[int] = None,
              ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Dict[str, str]]:
    """``(census shape, verdict shape, resolution)`` of a zoo member declared at ``declared``: the rows at
    ``side_override`` (default the declared side), the verdict at ``verdict_side`` (default the census side), each
    resolution marked against the declared side."""
    side = side_override or declared
    vside = verdict_side or side
    return _square(side), _square(vside), {"census": resolution_note(declared, side),
                                           "verdict": resolution_note(declared, vside)}


def run_zoo(out: str, names: Sequence[str], side_override: Optional[int] = None,
            verdict_side: Optional[int] = None) -> None:
    """Each member at its declared side (``ZOO_INPUT_SIDE``, else ``ZOO_DEFAULT_SIDE``), or at ``side_override``; its
    verdict at ``verdict_side`` when stage 2 does not fit at the census side. Both resolutions are recorded."""
    doc = _header("zoo")
    for name in names:
        declared = ZOO_INPUT_SIDE.get(name, ZOO_DEFAULT_SIDE)
        census_shape, verdict_shape, resolution = zoo_sides(declared, side_override, verdict_side)
        torch.manual_seed(0)
        model = zoo_builder(name)().eval()
        record = census_record(model, census_shape, ZOO_CLASSES, verdict_shape=verdict_shape)
        doc["members"][name] = {"family": "torchvision", "input_shape": list(census_shape),
                                "declared_input_shape": list(_square(declared)), "resolution": resolution,
                                "params": int(sum(p.numel() for p in model.parameters())),
                                "in_tv_zoo_23": name in TV_ZOO_23, **record}
        _write(out, doc)
        print(name, _summary(record), flush=True)


def _summary(record: dict) -> str:
    if "loud_failure" in record:
        return record["loud_failure"]
    if "census_refusal" in record:
        return f"census refused: {record['census_refusal'][:200]}"
    verdict = record["verdict"]
    shown = verdict["relative_error"] if verdict["status"] in (AGREEMENT, VALUE_MISMATCH) else verdict["root_refusal"]
    return f"{verdict['status']} {shown} ({record['wall_s']} s, peak {record['peak_rss_gb']} GB)"


def _write(out: str, doc: dict) -> None:
    with open(out, "w") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)


def _verdict_problems(name: str, verdict: dict) -> List[str]:
    """An agreement carries every required stage; a refusal names a ToolchainRefusal subclass at its root; a
    measured value mismatch and an incomplete verdict are failures, never refusals."""
    status = verdict["status"]
    if status == AGREEMENT:
        found = {stage for stage, _ in verdict.get("stages") or ()}
        missing = [stage for stage in probe_stages(MVM_PACKAGING) if stage not in found]
        return [f"{name}: fp64 agreement without the stage(s) {missing}"] if missing else []
    if status == VALUE_MISMATCH:
        return [f"{name}: value mismatch (relative error {verdict['relative_error']} above bound "
                f"{verdict.get('bound')}): an exactness failure, not a capability refusal"]
    if status != REFUSED:
        return [f"{name}: {status} (missing {verdict.get('missing_stages')})"]
    problems = []
    if not (verdict["refusal"] and verdict["refusal_is_toolchain_refusal"]):
        problems.append(f"{name}: refused without a ToolchainRefusal subclass name")
    if not (verdict.get("root_refusal") and verdict.get("root_refusal_is_toolchain_refusal")):
        problems.append(f"{name}: refused with a root that is not a ToolchainRefusal ({verdict.get('root_refusal')})")
    if verdict.get("value_mismatch"):
        problems.append(f"{name}: a value mismatch recorded as a refusal")
    return problems


def _row_problems(name: str, row: dict, counters: Optional[dict]) -> List[str]:
    problems = []
    if (row["level"] == "C0") == bool(row["label"]):
        problems.append(f"{name} {row['level']}: label {row['label']!r}")
    if row["level"] != "C0" and VERDICT_COLUMN in row:
        problems.append(f"{name} {row['level']}: an exactness verdict above C0")
    quantities = row.get("quantities") or {}
    if "onchip_macs" in quantities and quantities["onchip_macs"] < row.get("core_mac_floor", 0):
        problems.append(f"{name} {row['level']}: onchip_macs {quantities['onchip_macs']} below the cores' MAC "
                        f"floor {row['core_mac_floor']}")
    if counters is not None and not counters["agree"] and ("host_macs" in quantities or "host_macs" in row):
        problems.append(f"{name} {row['level']}: host_macs stated although the two host counters disagree")
    for entry in row.get("stated_boundaries") or ():
        where = f"{name} {row['level']} stated boundary {entry.get('boundary')!r}"
        if "host_mac_share" in entry and row.get("mac_total_withheld"):
            problems.append(f"{where}: a host MAC share over a withheld total ({row['mac_total_withheld']})")
        if "host_mac_share" in entry and "host_macs" not in entry:
            problems.append(f"{where}: a host MAC share without its host MACs")
        if "host_mac_share" not in entry and not entry.get("withheld"):
            problems.append(f"{where}: neither a host MAC share nor the reason it is withheld")
    return problems


def _resolution_problems(name: str, member: dict) -> List[str]:
    """The verdict's input is marked wherever it differs from the rows' (member and C0 row), and a zoo member's two
    resolution notes match its declared, census and verdict shapes."""
    problems = []
    census_shape = member.get("input_shape")
    marked = member.get("verdict_input_shape")
    if marked is not None and marked == census_shape:
        problems.append(f"{name}: verdict_input_shape {marked} repeats the census shape")
    c0 = member["rows"][0] if member.get("rows") else {}
    if c0.get("verdict_input_shape") != marked:
        problems.append(f"{name}: C0 row verdict_input_shape {c0.get('verdict_input_shape')} against the member's "
                        f"{marked}")
    resolution = member.get("resolution")
    if resolution is None:
        return problems
    declared = member.get("declared_input_shape")
    if not declared or not census_shape:
        return problems + [f"{name}: a resolution without declared_input_shape and input_shape"]
    for part, shape in (("census", census_shape), ("verdict", marked or census_shape)):
        expected = "declared" if list(shape) == list(declared) else resolution_note(declared[-1], shape[-1])
        if resolution.get(part) != expected:
            problems.append(f"{name}: {part} resolution {resolution.get(part)!r} at shape {shape}, declared "
                            f"{declared}: expected {expected!r}")
    return problems


def _provenance_problems(docs: Sequence[Tuple[str, dict]]) -> List[str]:
    """Every census file names the code that wrote it, and the files read together come from one code state."""
    problems = []
    for key in RECORD_DIGESTS:
        problems.extend(f"{path}: no {key}" for path, doc in docs if not doc.get(key))
        values = sorted({doc[key] for _, doc in docs if doc.get(key)})
        if len(values) > 1:
            problems.append(f"files from {len(values)} code states ({key}: {values})")
    return problems


def _reading(verdict: dict) -> str:
    """A member's pinned reading: the agreement, or the root refusal's type name, or the status."""
    if verdict["status"] == REFUSED:
        return str(verdict.get("root_refusal"))
    return str(verdict["status"])


def _pin_problems(docs: Sequence[Tuple[str, dict]], pins_path: str) -> List[str]:
    """Every member's reading equals its pinned one, and every pinned member is present (CL4's zoo re-pin)."""
    with open(pins_path) as handle:
        pins = json.load(handle)["members"]
    read = {name: _reading(member["verdict"]) for _, doc in docs for name, member in doc["members"].items()
            if "verdict" in member}
    problems = [f"{name}: not pinned" for name in sorted(read) if name not in pins]
    problems += [f"{name}: pinned {pins[name]}, reads {read[name]}" for name in sorted(read)
                 if name in pins and pins[name] != read[name]]
    return problems + [f"{name}: pinned but absent" for name in sorted(pins) if name not in read]


def check(paths: Sequence[str], pins: Optional[str] = None) -> int:
    """Acceptance over census JSONs: verdict per C0 row, refusals by name, labels, rows per level, MAC counters;
    with ``pins`` (a ``{"members": {name: reading}}`` file), each member's reading against its pinned one."""
    problems: List[str] = []
    converting: Dict[str, float] = {}
    roots: Dict[str, int] = {}
    mismatches = 0
    total = 0
    expected_rows = [(level.name, write) for level in LADDER for write in payload_writes(level)]
    docs = []
    for path in paths:
        with open(path) as handle:
            docs.append((path, json.load(handle)))
    problems.extend(_provenance_problems(docs))
    if pins is not None:
        problems.extend(_pin_problems(docs, pins))
    for path, doc in docs:
        total += len(doc["members"])
        for name, member in doc["members"].items():
            if "loud_failure" in member:
                problems.append(f"{name}: loud failure {member['loud_failure']}")
                continue
            if "census_refusal" in member:
                problems.append(f"{name}: census refused by name ({member['census_refusal']})")
            verdict = member["verdict"]
            if verdict["column"] != VERDICT_COLUMN:
                problems.append(f"{name}: verdict column {verdict['column']!r}")
            problems.extend(_verdict_problems(name, verdict))
            if verdict["status"] == AGREEMENT:
                converting[name] = verdict["relative_error"]
            elif verdict["status"] == REFUSED:
                roots[verdict.get("root_refusal")] = roots.get(verdict.get("root_refusal"), 0) + 1
            mismatches += verdict["status"] == VALUE_MISMATCH
            rows = [(row["level"], row["payload_write"]) for row in member["rows"]]
            if rows != expected_rows:
                problems.append(f"{name}: rows {rows}")
            problems.extend(_resolution_problems(name, member))
            counters = member.get("mac_counters")
            for row in member["rows"]:
                problems.extend(_row_problems(name, row, counters))
            if counters is not None and not counters["agree"]:
                # CL4 (plan §11.1): since G1's input_shapes rule the counters agree on every row, multi-input too.
                problems.append(f"{name}: MAC counters disagree"
                                + (" (a multi-input op)" if counters["has_multi_input_op"] else ""))
    worst = max(converting.values(), default=None)
    print("Code state: " + ", ".join(f"{key} {sorted({str(doc.get(key)) for _, doc in docs})}"
                                     for key in RECORD_DIGESTS))
    print(f"Of {total} members, {len(converting)} convert with fp64 agreement at most {worst} "
          f"(stage-2 identity value program); exact at atol 0 is CL4's integer column.")
    print(f"Refused by name (root refusal): {sum(roots.values())} {dict(sorted(roots.items(), key=str))}")
    print(f"Value mismatches (exactness failures, not refusals): {mismatches}")
    for problem in problems:
        print("FAIL", problem)
    return 1 if problems else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("scout", "zoo-roster", "zoo", "check"))
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--scout-dir")
    parser.add_argument("--max-params", type=int, default=ZOO_MAX_PARAMS)
    parser.add_argument("--pins", help="check: a pinned reading per member (tests/fixtures/requirement_census/"
                                       "zoo_pins.json), each member's reading compared with it")
    parser.add_argument("--side", type=int, help="zoo: run at this input side instead of the declared one (marked)")
    parser.add_argument("--verdict-side", type=int,
                        help="zoo: judge the stage-2 verdict at this side, the census at the census side (marked)")
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args.paths, pins=args.pins)
    out = args.paths[0]
    if args.command == "scout":
        if not args.scout_dir:
            parser.error("scout needs --scout-dir")
        run_scout(out, args.scout_dir)
    elif args.command == "zoo-roster":
        roster = zoo_roster(args.max_params)
        _write(out, {"max_params": args.max_params, "listed": len(roster),
                     "in_zoo": sum(r["in_zoo"] for r in roster), "members": roster, **{
                         k: v for k, v in _header("zoo-roster").items() if k in ("torch", "torchvision")}})
    else:
        run_zoo(out, [n for chunk in args.paths[1:] for n in chunk.split(",") if n], args.side, args.verdict_side)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
